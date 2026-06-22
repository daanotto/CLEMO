# CLEMO: Coherent Local Explanations for Mathematical Optimization

This repository contains the experiments for the paper *"Coherent Local
Explanations for Mathematical Optimization"* (Daan Otto, Jannis Kurtz,
Ş. İlker Birbil).

CLEMO is a sampling-based method for explaining the objective value and
decision variables of arbitrary exact or heuristic optimization algorithms,
while enforcing that the resulting explanations are **coherent** with the
underlying optimization problem: predicted decisions are feasible, and
their predicted objective value matches the objective function evaluated at
those decisions.

## Repository structure

```
.
├── data/
│   ├── instances/              # Pisinger knapsack benchmark instances
│   ├── train_data/             # cached samples + solver outputs (knapsack)
│   └── overview_betas_KS_25items.csv   # cached LR + CLEMO surrogate coefficients
├── notebooks/
│   ├── 01_shortest_path.ipynb
│   ├── 02_knapsack_main.ipynb
│   ├── 03_knapsack_runtime_ablation.ipynb
│   └── 04_vehicle_routing.ipynb
├── scripts/
│   ├── generate_train_data.py  # regenerates data/train_data/*.csv (Gurobi, hours)
│   └── fit_clemo_betas.py      # regenerates data/overview_betas_KS_25items.csv (hours)
├── figures/                     # example/reference output figures
├── requirements.txt
└── README.md  
```

## Notebooks -> paper sections

| Notebook | Paper section(s) | Reproduces | Runtime | External deps |
|---|---|---|---|---|
| `01_shortest_path.ipynb` | Section 4.1 | Figure 1, Table 1, Figure D.4, Table C.7 | seconds | none |
| `02_knapsack_main.ipynb` | Section 4.2 | Table 2, Figures D.5-D.8 | seconds (cached) / hours (refit) | Gurobi (optional, only if refitting) |
| `03_knapsack_runtime_ablation.ipynb` | Section 4.2, Appendix C | Table 3, Table C.5, Table C.6, Figure 2 | tens of minutes | Gurobi |
| `04_vehicle_routing.ipynb` | Section 4.3 | Table 4, Figure 3, Figure D.9 | ~1-1.5 hours | OR-Tools |

Each notebook is self-contained: it defines its own problem instance,
sampling/loss-function code, and plotting. There is intentional duplication
of small helper functions (e.g. `sig`, `lg1`, `lg2`, RBF weighting) across
notebooks rather than a shared package, so each notebook can be read and run
independently.

### `01_shortest_path.ipynb`

Explains Dijkstra's algorithm on a small graph with a single sensitive
parameter. Fully self-contained, runs in seconds. Good starting point for
understanding the CLEMO loss (Eq. 10) and the coherence regularizers (Eq.
8-9) in their simplest form. Proof of value as a comparison against 
Parametric Optimization

### `02_knapsack_main.ipynb`

The main Knapsack comparison (Table 2): CLEMO vs. independent linear
regression (LR) vs. independent decision tree regressors (DTR), across 4 instance
types x 10 instances x 10 resamples. By default loads cached data from
`data/`; see `data/README.md` for how to regenerate from scratch (requires
solver e.g. Gurobi).

### `03_knapsack_runtime_ablation.ipynb`

Runtime scaling (Table 3), the coherence-regularizer ablation (Table C.5),
a comparison of SciPy optimizers (Table C.6), and the convergence-over-
iterations plot (Figure 2). Requires Gurobi; generates its own (small) data.

### `04_vehicle_routing.ipynb`

Explains the Google OR-Tools heuristic on a 17-node CVRP instance (Table 4,
Figure 3, Figure D.9). Requires OR-Tools; the 1000 OR-Tools solves (5s time
limit each) during dataset creation making this the slowest notebook.

## Setup

```bash
pip install -r requirements.txt
```

Note that Gurobi requires a license (a free academic or trial license is 
sufficientfor the problem sizes used here). OR-Tools and all other dependencies 
are open source.

## Citing

If you use this code, please cite the paper:

```bibtex
@article{otto2025coherent,
  title={Coherent Local Explanations for Mathematical Optimization},
  author={Otto, Daan and Kurtz, Jannis and Birbil, S Ilker},
  journal={arXiv preprint arXiv:2502.04840},
  year={2025}
}
```
