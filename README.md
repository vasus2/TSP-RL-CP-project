# TSP-RL-CP-project
This is a project for CS 498 Algorithmic Engineering. The goal is to reimplement Cappart et al., Combining Reinforcement Learning and Constraint Programming for Combinatorial Optimization, AAAI 2021 for the traveling salesman problem.

## Current Experiment Pipeline

classical heuristic benchmarks:

```bash
python3 src/benchmark.py
```

CP-style branch-and-bound experiments:

```bash
python3 src/cp_experiments.py
```

Generate all baseline and search-statistics plots:

```bash
MPLCONFIGDIR=.mplconfig python3 src/visualize_results.py
```

learned branch-ordering pipeline:

```bash
python3 src/generate_branch_data.py
python3 src/train_branch_model.py
python3 -c 'import sys; sys.path.insert(0, "src"); from cp_experiments import run_learned_policy_experiments; run_learned_policy_experiments()'
MPLCONFIGDIR=.mplconfig python3 src/visualize_results.py
```

Key outputs are saved in `results/`. Trained branch-scoring models are saved in `models/`.
