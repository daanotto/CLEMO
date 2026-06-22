#!/usr/bin/env python3
"""
Regenerates `data/train_data/*.csv` -- the sampled-and-solved Knapsack
Problem (KP) data used by `notebooks/02_knapsack_main.ipynb`.

This is the expensive half of the data pipeline: it requires a working
Gurobi installation (via pyomo's `gurobi_direct` interface) and solves
4 types x 10 instances x 10 batches x 1000 samples = 400,000 continuous
knapsack LPs.

Usage
-----
    cd scripts
    python generate_train_data.py
    python generate_train_data.py --types 0 1 --instances 0 1 2   # partial run
    python generate_train_data.py --resume                          # skip files that already exist

Output
------
One CSV per (type, instance) in `../data/train_data/`, named
`instance_data_KS_25items_Type_{t}_Instance_{i}.csv`, each containing the
10 resampled batches (1000 samples each) for that instance -- see
`../data/README.md` for the exact column schema.

Runtime
-------
On the order of hours, depending on the Gurobi license/machine. Each
(type, instance) is checkpointed independently (one CSV per file), so the
script is safe to interrupt and resume with `--resume`.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition

SCRIPT_DIR = Path(__file__).resolve().parent
INSTANCES_DIR = SCRIPT_DIR.parent / "data" / "instances"
OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "train_data"

KS_TYPES = ["00Uncorrelated", "01WeaklyCorrelated", "02StronglyCorrelated", "03InverseStronglyCorrelated"]

# Global experiment settings -- must match notebooks/02_knapsack_main.ipynb
NR_ITEMS = 25
SAMPLE_SIZE = 1000
NR_BATCHES = 10
SEEDS = list(range(42, 42 + NR_BATCHES))  # one seed per batch, matches the original notebook

NR_TRGTS = NR_ITEMS + 1
NR_FTRES = 2 * NR_ITEMS
FEATURES = list(range(NR_FTRES))

solver = pyo.SolverFactory("gurobi_direct")


def model_KS_cts(vals, output="goal"):
    """Solve the continuous knapsack problem h(theta) with Gurobi via Pyomo.

    Args:
        vals: concatenation (c, w, b) of item values, weights, and budget
            (length 2n+1).
        output: one of {'goal', 'bounded', 'feasibility', 'decision vector', 'all'}.

    Returns:
        Depending on `output`: the optimal objective value, a
        feasibility/boundedness flag, the optimal decision vector, or
        [objective, x_1, ..., x_n].
    """
    n = (len(vals) - 1) // 2

    model = pyo.ConcreteModel("Knapsack continuous model")
    model.x = pyo.Var(range(n), domain=pyo.NonNegativeReals, bounds=(0, 1))
    model.objective = pyo.Objective(expr=sum(vals[i] * model.x[i] for i in range(n)), sense=pyo.maximize)
    model.budget = pyo.Constraint(expr=sum(vals[i + n] * model.x[i] for i in range(n)) <= vals[2 * n])

    result = solver.solve(model)

    if output == "goal":
        return model.objective()
    elif output in ("bounded", "feasibility"):
        return result.solver.termination_condition != TerminationCondition.infeasibleOrUnbounded
    elif output == "decision vector":
        return [pyo.value(model.x[i]) for i in range(n)]
    elif output == "all":
        return [model.objective()] + [pyo.value(model.x[i]) for i in range(n)]
    raise ValueError(f"Unsupported output type: {output!r}")


def sample_perturbations_normal(orig, ftr_index_list, mean=0, var=0.2, size=1000,
                                 feasibility_check=True, bounded_check=True):
    """Draw `size` perturbations of `orig`, keeping feasible & bounded ones.

    Each sampled feature `j` in `ftr_index_list` is perturbed multiplicatively:
    `theta_j -> theta_j + N(mean, (var * theta_j)^2)`. The first row of the
    returned array is always `orig` itself (the present problem).
    """
    samples = [orig]
    while len(samples) < size:
        candidate = copy.deepcopy(orig)
        for j in ftr_index_list:
            candidate[j] += np.random.normal(mean, candidate[j] * var)

        ok = True
        if feasibility_check:
            ok = ok and model_KS_cts(candidate, output="feasibility")
        if bounded_check:
            ok = ok and model_KS_cts(candidate, output="bounded")
        if ok:
            samples.append(np.asarray(candidate))

    return np.asarray(samples)


def std_weight_function(a, b, ftr_index_list, kernel_width=None):
    """RBF kernel weight w_i = exp(-d(a,b)^2 / nu^2), Eq. (11)."""
    d = np.linalg.norm(a - b)
    nu = (0.75 * len(ftr_index_list)) if kernel_width is None else kernel_width
    return np.exp(-(d ** 2) / (2 * nu ** 2))


def get_weights_from_samples(samples, ftr_index_list, width=None):
    """RBF weights of all samples relative to the first sample (theta_0)."""
    origin = samples[0]
    return [std_weight_function(origin, s, ftr_index_list, width) for s in samples]


def load_instance(type_idx, instance_idx):
    """Load and rescale one Pisinger KP instance into theta_0 = (c, w, b=1)."""
    path = INSTANCES_DIR / KS_TYPES[type_idx] / f"s{instance_idx:03d}.kp"
    with open(path) as f:
        lines = f.readlines()

    b = int(lines[2])
    v = [int(line.split()[0]) for line in lines[4:]]
    w = [int(line.split()[1]) for line in lines[4:]]

    # Rescale so the present problem's parameters are well-scaled (matches
    # the original notebook: divide by the instance's nominal budget, x2)
    c = [x / b * 2 for x in v[:NR_ITEMS]]
    a = [x / b * 2 for x in w[:NR_ITEMS]]
    return np.concatenate((c, a, [1.0]))


def generate_instance_csv(type_idx, instance_idx):
    """Sample, solve, and save all 10 batches for one (type, instance)."""
    cab = load_instance(type_idx, instance_idx)
    print(f"  Total scaled weight: {sum(cab[NR_ITEMS:2 * NR_ITEMS]):.3f} (budget = 1.0)")

    records = []
    for batch_idx, seed in enumerate(SEEDS):
        np.random.seed(seed)
        t0 = time.time()

        samples = sample_perturbations_normal(cab, FEATURES, size=SAMPLE_SIZE)
        actuals = [model_KS_cts(s, output="all") for s in samples]
        d_list = [np.linalg.norm(samples[0] - s) for s in samples]
        weights = get_weights_from_samples(samples, FEATURES, width=np.mean(d_list))

        for row_idx in range(SAMPLE_SIZE):
            row = {"Type": type_idx, "Instance": instance_idx, "Batch": batch_idx}
            row.update({f"Original (c,A,b)_{j}": cab[j] for j in range(len(cab))})
            row.update({f"Samples_{j}": samples[row_idx, j] for j in range(samples.shape[1])})
            row.update({f"Actuals_{j}": actuals[row_idx][j] for j in range(len(actuals[row_idx]))})
            row["Weight"] = weights[row_idx]
            records.append(row)

        print(f"  Batch {batch_idx}: {SAMPLE_SIZE} samples solved in {time.time() - t0:.1f}s")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--types", type=int, nargs="+", default=list(range(4)),
                         help="Which KP types to generate (0-3, default: all)")
    parser.add_argument("--instances", type=int, nargs="+", default=list(range(10)),
                         help="Which instance indices to generate (0-9, default: all)")
    parser.add_argument("--resume", action="store_true",
                         help="Skip (type, instance) pairs whose output CSV already exists")
    args = parser.parse_args()

    if not solver.available():
        print("ERROR: Gurobi (via pyomo's gurobi_direct interface) is not available. "
              "Check your Gurobi license/installation.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_total = len(args.types) * len(args.instances)
    n_done = 0
    for type_idx in args.types:
        for instance_idx in args.instances:
            out_path = OUTPUT_DIR / f"instance_data_KS_25items_Type_{type_idx}_Instance_{instance_idx}.csv"
            n_done += 1
            print(f"[{n_done}/{n_total}] Type {type_idx}, Instance {instance_idx} -> {out_path.name}")

            if args.resume and out_path.exists():
                print("  Already exists, skipping (--resume).")
                continue

            df = generate_instance_csv(type_idx, instance_idx)
            df.to_csv(out_path, index=False)
            print(f"  Wrote {len(df)} rows to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
