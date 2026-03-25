# sequence-learning / ELiSe

A biophysically-inspired spiking neural network that learns to generate temporal sequences (music patterns, random spike trains, etc.), based on Kriener et al., 2024. The network uses a dendritic-somatic compartment model with plastic dendritic synapses and fixed-delay recurrent connectivity.

---

## Setup

1. Fork and clone the repo:
   ```bash
   git clone git@github.com:USERNAME/sequence-learning.git
   git remote add upstream git@github.com:unibe-cns/sequence-learning.git
   ```
2. Create and activate a Python environment (Python >= 3.11):
   ```bash
   # venv
   python -m venv --system-site-packages <env_name> && source ./<env_name>/bin/activate
   # or conda
   conda create -n <env_name> python=3.11 && conda activate <env_name>
   ```
3. Install dependencies and the package:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

---

## Repository structure

```
sequence-learning/
├── src/elise/          # Core library (model, data, weights, tracker, …)
├── scripts/
│   ├── run.py                   # Main entry point for a single run
│   ├── train.py                 # Training loop
│   ├── run_experiments_local.py # Local multi-experiment dispatcher
│   ├── run_experiments.py       # Cluster (Slurm) multi-experiment dispatcher
│   ├── config.toml              # Master configuration (all defaults)
│   ├── experiments/             # Per-experiment TOML overlays
│   ├── patterns/                # Input pattern files (.txt)
│   ├── array.sh                 # Slurm array-job script
│   ├── plotting.py              # Visualisation helpers
│   └── utils.py                 # Config loading utilities
└── tests/
```

---

## Configuration

All parameters live in `scripts/config.toml`. The file is divided into sections:

| Section | Controls |
|---|---|
| `experiment_params` | MLflow experiment name, seed, pattern files, transforms, replay learning |
| `noise_params` | Membrane/weight noise, disruption, partial replay |
| `network_params` | Number of visible / latent neurons |
| `weight_params` | Connection probabilities, weight ranges, delay ranges |
| `simulation_params` | Time step, epoch counts, learning rates and decay |
| `pattern_params` | Pattern duration, noise, non-Markovian structure |
| `neuron_params` | Membrane capacitance, reversal potentials, conductances, activation |
| `tracking_params` | Which variables to record and at what frequency |

### Experiment overlays

Rather than editing `config.toml` directly, each named experiment has a small TOML file in `scripts/experiments/` that overrides only the relevant keys. These are **deep-merged** on top of `config.toml` at dispatch time. For example:

```toml
# scripts/experiments/Prelude1.toml
[experiment_params]
mlflow_experiment = "Elise-preludes"
patterns = ["Prelude1"]
seed = 1
```

---

## Running a single experiment

All scripts must be run from inside the `scripts/` directory.

```bash
cd scripts
python run.py
```

`run.py` reads `config.toml` from the current directory, creates a uniquely named MLflow run, executes the full training + replay pipeline, saves artifacts and figures locally, and logs everything to MLflow.

**CLI flags:**

| Flag | Description |
|---|---|
| `--saving` | Upload all artifact pickles (network, dataloader, trackers) to MLflow in addition to the config and network weights |
| `--param_tag <str>` | Attach an extra string tag to the MLflow run (useful for labelling parameter sweeps) |
| `--debug` | Run in debug mode — see below |

Example:
```bash
python run.py --saving --param_tag "lr_sweep"
```

Outputs written to the working directory:
- `artifacts/` — pickled network, dataloader, and tracker objects
- `figures/` — connectivity and weight plots
- `run_id.txt` — MLflow run ID for this run

---

## Running multiple experiments

### Locally (`run_experiments_local.py`)

Dispatches one or more experiment TOML overlays as local subprocesses. Each experiment gets its own subdirectory under `cluster_runs/` with a frozen copy of the code and a baked `config.toml`.

```bash
cd scripts

# Run a single experiment
python run_experiments_local.py Prelude1

# Run all experiments in scripts/experiments/
python run_experiments_local.py

# Run in debug mode (passes --debug to each run.py subprocess)
python run_experiments_local.py Prelude1 --debug
```

The experiment name maps directly to a file in `scripts/experiments/<name>.toml`.

### On the cluster (`run_experiments.py`)

Identical interface but submits Slurm array jobs (`sbatch array.sh`) instead of local subprocesses. `array.sh` launches 20 parallel seeds; only seed 0 saves full artifacts (`--saving`). Debug mode is not available on the cluster.

```bash
cd scripts

# Submit a single experiment to the cluster
python run_experiments.py Prelude1

# Submit all experiments
python run_experiments.py
```

---

## Debug mode

Pass `--debug` to run a fast sanity-check run without touching your main MLflow experiments:

```bash
python run.py --debug
```

What changes in debug mode:
- `training_epochs` is reduced to **2** (from 10)
- `replay_epochs` is reduced to **5** (from 50)
- The MLflow experiment is fixed to **"ELise-tests"** regardless of `config.toml`
- A `"debug"` tag is added to the run

This is useful for quickly verifying that a new configuration or code change runs end-to-end before committing to a full training run. There is also a convenience shell script for this:

```bash
bash besh_tst.sh   # equivalent to: python run.py --debug
```

---

## MLflow

All runs are tracked with [MLflow](https://mlflow.org/). The tracking database lives at `scripts/mlflow.db` and run artifacts are stored under `scripts/mlruns/`.

### Starting the MLflow UI

From the `scripts/` directory, launch the UI server:

```bash
cd scripts
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open your browser at **http://127.0.0.1:5000**.

If port 5000 is already occupied, pick a different port:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

### What is logged per run

| Item | How |
|---|---|
| All `config.toml` parameters (flat) | `mlflow.log_params` |
| Raw `config.toml` file | `mlflow.log_artifact` |
| Per-epoch training / validation / replay metrics | `mlflow.log_metric` |
| Network weights pickle (`network.pkl`) | `mlflow.log_artifact` |
| Full artifact set (when `--saving`) | `mlflow.log_artifact` |
| Connectivity and weight figures | `mlflow.log_figure` |
| Tags: pattern name, group tag, debug flag | `mlflow.set_tag` |

### Finding a run programmatically

Each run writes its MLflow run ID to `run_id.txt` in the working directory. You can use this to reload results later:

```python
import mlflow

with open("cluster_runs/Prelude1/run_id.txt") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient("sqlite:///scripts/mlflow.db")
run = client.get_run(run_id)
```

---

## Contributing

1. Create a branch from `main` for your work.
2. Run the test suite: `pytest`
3. Ensure pre-commit checks pass: `pre-commit run --all-files`
4. Open a pull request against `main`.
