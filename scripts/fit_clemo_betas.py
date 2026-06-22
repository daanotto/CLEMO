#!/usr/bin/env python3
"""
Regenerates `data/overview_betas_KS_25items.csv` -- the cached LR and CLEMO
surrogate coefficients used by `notebooks/02_knapsack_main.ipynb` (when
`USE_CACHED_CLEMO_BETAS = True`).

Reads the sampled-and-solved data from `data/train_data/` (see
`generate_train_data.py` to regenerate that first) and, for every
(type, instance, batch):

  1. fits the LR benchmark (independent linear regression per output, Eq. 5);
  2. computes the CLEMO loss weights (lambda_C1, lambda_C2) from the LR fit,
     following the 0.5-ratio rule (paper, Section 4, "Setup");
  3. fits CLEMO (Eq. 10) via SLSQP, warm-started at the LR solution.

Usage
-----
    cd scripts
    python fit_clemo_betas.py
    python fit_clemo_betas.py --types 0 1 --instances 0 1 2   # partial run
    python fit_clemo_betas.py --resume                          # skip rows already in the output CSV

Runtime
-------
CLEMO's SLSQP fit is the expensive step here -- on the order of minutes per
(instance, batch) for this 25-item problem size (see
`notebooks/03_knapsack_runtime_ablation.ipynb`, Table 3, for runtime
scaling by problem size). With 4 types x 10 instances x 10 batches = 400
fits, expect a multi-hour run in total. Progress is written incrementally
(one row per fit, flushed after every instance), so `--resume` can pick up
from where a previous run left off.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_DATA_DIR = SCRIPT_DIR.parent / "data" / "train_data"
OUTPUT_PATH = SCRIPT_DIR.parent / "data" / "overview_betas_KS_25items.csv"

# Global experiment settings -- must match notebooks/02_knapsack_main.ipynb
NR_ITEMS = 25
NR_BATCHES = 10
NR_TRGTS = NR_ITEMS + 1
NR_FTRES = 2 * NR_ITEMS
NR_FTRES_INTRCPT = NR_FTRES + 1
FEATURES = list(range(NR_FTRES))

CAB_COLS = [f"Original (c,A,b)_{j}" for j in range(2 * NR_ITEMS + 1)]
SAMPLES_COLS = [f"Samples_{j}" for j in range(2 * NR_ITEMS + 1)]
ACTUALS_COLS = [f"Actuals_{j}" for j in range(NR_TRGTS)]


# --- CLEMO loss functions (Eq. 10), identical to notebooks/02_knapsack_main.ipynb ---

def lss_all(beta, X, Y, W, lgr_obj=1, lgr_cns=1, oneD=True):
    if oneD:
        beta = beta.reshape(NR_FTRES_INTRCPT, NR_TRGTS)
    W = np.asarray(W)
    idx = list(range(len(X)))

    pred = np.matmul(X, beta)
    std_err = np.sum(W[:, None] * np.square(Y - pred))
    obj_err = np.sum(W[idx] * np.square(
        pred[:, 0] - np.sum(pred[:, 1:] * X[:, :NR_ITEMS], axis=1))[idx])
    cns_err = np.sum(W[idx] * np.maximum(
        0, np.sum(pred[:, 1:] * X[:, NR_ITEMS:-1], axis=1) - 1)[idx])

    return std_err + lgr_obj * obj_err + lgr_cns * cns_err


def lss_std(beta, X, Y, W, oneD=True):
    if oneD:
        beta = beta.reshape(NR_FTRES_INTRCPT, NR_TRGTS)
    return np.sum(np.asarray(W)[:, None] * np.square(Y - np.matmul(X, beta)))


def lss_obj(beta, X, Y, W, oneD=True):
    if oneD:
        beta = beta.reshape(NR_FTRES_INTRCPT, NR_TRGTS)
    pred = np.matmul(X, beta)
    return np.sum(np.asarray(W) * np.square(pred[:, 0] - np.sum(pred[:, 1:] * X[:, :NR_ITEMS], axis=1)))


def lss_cns(beta, X, Y, W, oneD=True):
    if oneD:
        beta = beta.reshape(NR_FTRES_INTRCPT, NR_TRGTS)
    pred = np.matmul(X, beta)
    return np.sum(np.asarray(W) * np.maximum(0, np.sum(pred[:, 1:] * X[:, NR_ITEMS:-1], axis=1) - 1))


def lss_all_jac_sg(beta, X, Y, W, lgr_obj=1, lgr_cns=1, oneD=True):
    grad = np.zeros((NR_FTRES_INTRCPT, NR_TRGTS))
    if oneD:
        beta = beta.reshape(NR_FTRES_INTRCPT, NR_TRGTS)
    W = np.asarray(W)

    idx = list(range(len(X)))

    pred = np.matmul(X, beta)
    tmp_std = Y - pred
    tmp_obj = pred[:, 0] - np.sum(pred[:, 1:] * X[:, :NR_ITEMS], axis=1)
    tmp_cns = (np.sum(pred[:, 1:] * X[:, NR_ITEMS:-1], axis=1) > 1).astype(int)

    for k in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            grad[k, j] -= 2 * np.sum(W * X[:, k] * tmp_std[:, j])
            if j == 0:
                grad[k, j] += 2 * lgr_obj * np.sum(W[idx] * X[idx, k] * tmp_obj[idx])
            else:
                grad[k, j] -= 2 * lgr_obj * np.sum(W[idx] * X[idx, k] * X[idx, j - 1] * tmp_obj[idx])
                grad[k, j] += lgr_cns * np.sum(W[idx] * tmp_cns[idx] * X[idx, k] * X[idx, j + NR_ITEMS - 1])

    return grad.flatten()


def get_warm_start_linreg(x, y, w):
    """Independent linear regression per output -- the LR benchmark / warm start (Eq. 5)."""
    models = []
    for j in range(len(y[0])):
        clf = LinearRegression()
        clf.fit(x, [row[j] for row in y], sample_weight=w)
        models.append(clf)

    beta_T = np.ones((len(y[0]), len(FEATURES) + 1))
    for i, model in enumerate(models):
        beta_T[i, :-1] = model.coef_
        beta_T[i, -1] = model.intercept_
    return beta_T.T


def fit_clemo(X, Y, W, lgr_obj, lgr_cns, warm_start=None, region="all", maxiter=1000):
    """Fit the CLEMO surrogate (Eq. 10) via SLSQP. Returns beta, shape (NR_TRGTS, NR_FTRES_INTRCPT)."""
    if warm_start is None:
        warm_start = get_warm_start_linreg(X[:, FEATURES], Y, W).flatten()

    args = (X, Y, W, lgr_obj, lgr_cns, True, region)
    sol = minimize(lss_all, warm_start, args=args, method="SLSQP",
                    jac=lss_all_jac_sg, options={"maxiter": maxiter})
    return np.transpose(sol.x.reshape(NR_FTRES_INTRCPT, NR_TRGTS))


def beta_to_row(beta_T, type_idx, instance_idx, batch_idx, reg_type):
    """Flatten a (NR_TRGTS, NR_FTRES_INTRCPT) beta array into one CSV row, matching the
    schema documented in data/README.md (`beta_{i},{j}` columns)."""
    row = {"Type": f"Type {type_idx}", "Instance": f"Instance {instance_idx}",
           "Batch": f"Batch {batch_idx}", "Regression_Type": reg_type}
    for i in range(beta_T.shape[0]):
        for j in range(beta_T.shape[1]):
            row[f"beta_{i},{j}"] = beta_T[i, j]
    return row


def already_done(existing_df, type_idx, instance_idx, batch_idx, reg_type):
    if existing_df is None:
        return False
    mask = ((existing_df["Type"] == f"Type {type_idx}")
            & (existing_df["Instance"] == f"Instance {instance_idx}")
            & (existing_df["Batch"] == f"Batch {batch_idx}")
            & (existing_df["Regression_Type"] == reg_type))
    return mask.any()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--types", type=int, nargs="+", default=list(range(4)),
                         help="Which KP types to fit (0-3, default: all)")
    parser.add_argument("--instances", type=int, nargs="+", default=list(range(10)),
                         help="Which instance indices to fit (0-9, default: all)")
    parser.add_argument("--maxiter", type=int, default=1000, help="SLSQP iteration budget for CLEMO")
    parser.add_argument("--resume", action="store_true",
                         help="Skip (type, instance, batch, method) rows already present in the output CSV")
    args = parser.parse_args()

    existing_df = None
    rows = []
    if args.resume and OUTPUT_PATH.exists():
        existing_df = pd.read_csv(OUTPUT_PATH)
        rows = existing_df.to_dict("records")
        print(f"Resuming: {len(rows)} rows already present in {OUTPUT_PATH.name}")

    n_total = len(args.types) * len(args.instances)
    n_done = 0
    for type_idx in args.types:
        for instance_idx in args.instances:
            n_done += 1
            csv_path = TRAIN_DATA_DIR / f"instance_data_KS_25items_Type_{type_idx}_Instance_{instance_idx}.csv"
            if not csv_path.exists():
                print(f"ERROR: {csv_path} not found. Run generate_train_data.py first.", file=sys.stderr)
                sys.exit(1)

            print(f"[{n_done}/{n_total}] Type {type_idx}, Instance {instance_idx}")
            df = pd.read_csv(csv_path)

            for batch_idx, batch_group in df.groupby("Batch"):
                X = batch_group[SAMPLES_COLS].to_numpy()
                Y = batch_group[ACTUALS_COLS].to_numpy()
                W = batch_group["Weight"].to_numpy()

                # --- LR (also the warm start for CLEMO) ---
                t0 = time.time()
                if already_done(existing_df, type_idx, instance_idx, batch_idx, "Standard Linear Regression"):
                    print(f"  Batch {batch_idx}: LR already done, skipping (--resume).")
                    lr_beta = None
                else:
                    lr_beta_flat = get_warm_start_linreg(X[:, FEATURES], Y, W)
                    lr_beta_T = np.transpose(lr_beta_flat)
                    rows.append(beta_to_row(lr_beta_T, type_idx, instance_idx, batch_idx,
                                             "Standard Linear Regression"))
                    lr_beta = lr_beta_flat.flatten()
                    print(f"  Batch {batch_idx}: LR fit in {time.time() - t0:.2f}s")

                # --- CLEMO loss weights, from the LR fit ---
                if lr_beta is None:
                    # Need the LR beta to compute lambda weights even if we're
                    # skipping the LR row itself; refit quickly (LR is cheap).
                    lr_beta = get_warm_start_linreg(X[:, FEATURES], Y, W).flatten()

                if already_done(existing_df, type_idx, instance_idx, batch_idx, "CLEMO"):
                    print(f"  Batch {batch_idx}: CLEMO already done, skipping (--resume).")
                    continue

                lcl_lss_std = lss_std(lr_beta, X, Y, W)
                lcl_lss_obj = max(lss_obj(lr_beta, X, Y, W), 1e-3)
                lcl_lss_cns = max(lss_cns(lr_beta, X, Y, W), 1e-3)
                lgr_obj = np.round(0.5 * lcl_lss_std / lcl_lss_obj, 1)
                lgr_cns = np.round(0.5 * lcl_lss_std / lcl_lss_cns, 1)

                # --- CLEMO fit (SLSQP, warm-started at LR) ---
                t0 = time.time()
                clemo_beta_T = fit_clemo(X, Y, W, lgr_obj, lgr_cns, warm_start=lr_beta, maxiter=args.maxiter)
                rows.append(beta_to_row(clemo_beta_T, type_idx, instance_idx, batch_idx, "CLEMO"))
                print(f"  Batch {batch_idx}: CLEMO fit in {time.time() - t0:.1f}s "
                      f"(lambda_obj={lgr_obj}, lambda_cns={lgr_cns})")

            # Checkpoint after every instance (400 batches total across the full run)
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            print(f"  Checkpointed {len(rows)} rows to {OUTPUT_PATH}")

    print("Done.")


if __name__ == "__main__":
    main()
