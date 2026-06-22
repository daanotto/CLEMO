# Data

This directory contains the input data and cached intermediate results used
by the notebooks in `../notebooks/`.

## `instances/`

The Pisinger knapsack benchmark instances used by `02_knapsack_main.ipynb`
and `03_knapsack_runtime_ablation.ipynb`, organized by correlation type:

```
instances/
├── 00Uncorrelated/s000.kp ... s009.kp
├── 01WeaklyCorrelated/s000.kp ... s009.kp
├── 02StronglyCorrelated/s000.kp ... s009.kp
└── 03InverseStronglyCorrelated/s000.kp ... s009.kp
```

Each `.kp` file lists, after a small header, one `value weight` pair per
line. Source: <https://github.com/likr/kplib>; see also the [instance type
descriptions](https://di.ku.dk/forskning/Publikationer/tekniske_rapporter/tekniske-rapporter-2003/03-08.pdf)
(Pisinger, 2003). These files are small (~180 KB total) and committed
as-is.

## `train_data/`

**40 CSV files** (`instance_data_KS_25items_Type_{t}_Instance_{i}.csv`
for `t in 0..3`, `i in 0..9`). Each file contains 10,000 rows: 10 resampled
batches (`Batch` column, 0-9) of 1000 samples each, for one (type, instance)
combination of the 25-item continuous knapsack problem.

Columns:

- `Type`, `Instance`, `Batch`: indices.
- `Original (c,A,b)_0..50`: the present problem's parameters
  `theta_0 = (c, w, b=1)` (51 values: 25 item values, 25 item weights, 1
  budget), repeated on every row.
- `Samples_0..50`: the sampled parameters `theta_i` for this row (same
  layout as `Original (c,A,b)`).
- `Actuals_0..25`: the solver output `h(theta_i) = (objective, x_1, ..., x_25)`.
- `Weight`: the RBF sample weight `w_i` (Eq. 11).

### Regenerating `train_data/`

This is the expensive part of the pipeline -- generating it requires a
working Gurobi license (via `pyomo`'s `gurobi_direct` interface) and solves
`4 types x 10 instances x 10 batches x 1000 samples = 400,000` continuous
knapsack LPs.

To regenerate, run the data-generation cells of `02_knapsack_main.ipynb`
(sections 3-4 in the original exploratory notebook covered sampling +
solving; see git history / `KS_cts_various_types.ipynb` in earlier commits
for the original sampling loop). At a high level, for each `(type,
instance)`:

1. Load the corresponding `instances/0XCorrelated/sXXX.kp` file and rescale
   `(v, w)` so the present problem's parameters `theta_0 = (c, w, b=1)` are
   well-scaled (see `model_KS_cts` and the instance-construction cell in
   `03_knapsack_runtime_ablation.ipynb` for the same pattern at smaller
   sizes).
2. For each of the 10 batches, set `np.random.seed(42 + batch)`, draw 1000
   samples with `sample_perturbations_normal`, solve each with
   `model_KS_cts(..., output='all')`, and compute RBF weights with
   `get_weights_from_samples`.
3. Concatenate `(Type, Instance, Batch, Original (c,A,b), Samples, Actuals,
   Weight)` into the CSV layout described above.

If you only need to *analyze* the existing data (Table 2, Figures D.5-D.8),
you do not need to regenerate this -- `02_knapsack_main.ipynb` loads it
directly.

## `overview_betas_KS_25items.csv`

Cached CLEMO and LR surrogate coefficients (`beta`, flattened),
one row per `(Type, Instance, Batch, Regression_Type)` with
`Regression_Type in {'Standard Linear Regression', 'CLEMO'}`. Columns
`beta_{i},{j}` give `beta[i, j]` for a `(nr_trgts, nr_ftres_intrcpt) = (26,
51)` array (already transposed to match the `Beta` arrays used in
`02_knapsack_main.ipynb`.

This is the **cheap** part to regenerate: with `train_data/` available,
`02_knapsack_main.ipynb` can refit both LR and CLEMO directly (set
`USE_CACHED_CLEMO_BETAS = False`). CLEMO refitting uses SLSQP and takes on
the order of minutes per `(instance, batch)` for this problem size (see
`03_knapsack_runtime_ablation.ipynb`, Table 3, for runtime scaling) --
roughly a few hours for all 400 `(type, instance, batch)` combinations.

This file corresponds to the most recent run (2026-05-22) of the original
exploratory notebooks.
