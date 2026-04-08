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


## Contributing

1. Create a branch from `main` for your work.
2. Run the test suite: `pytest`
3. Ensure pre-commit checks pass: `pre-commit run --all-files`
4. Open a pull request against `main`.

## Repository structure

```
sequence-learning/
├── src/elise/          # Core library (model, data, weights, tracker, …)
├── scripts/
│   ├── run.py                   # Main entry point for a single run
│   ├── train.py                 # Training loop
│   ├── run_experiments.py       # Multi-experiment dispatcher (e.g. local)
│   ├── config.toml              # Master configuration (all defaults)
│   ├── experiments/             # Per-experiment TOML overlays
│   ├── patterns/                # Input pattern files (.txt)
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


---

## Custom single experiments

All scripts are run from inside the `scripts/` directory.

### Custom Single Run

```bash
python run.py
```

### Pre-configured Runs

Each experiment described in the paper has a small TOML file in `scripts/experiments/` that overrides only the relevant keys. These are **deep-merged** on top of `config.toml` at dispatch time. `run_experiments.py` snapshots the current scripts into `cluster_runs/<experiment_name>/`, bakes the merged config, and launches `run.py`.

You can either specify a specific experiment name (name of the .toml file) or `--all` to run all experiments:

```bash
cd scripts

# Run a single named experiment
python run_experiments.py Prelude1

# Run all experiments defined in experiments/
python run_experiments.py --all

```

Not specifying either will print an error. There is also a cluster variant of this script (using Slurm `sbatch`) for HPC use, which is not documented here.


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

This is useful for quickly verifying that a new configuration or code change runs end-to-end before committing to a full training run.
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


