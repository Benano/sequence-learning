#!/usr/bin/env python3

import hashlib
from datetime import datetime
from pathlib import Path

import coolname
import mlflow


def hash_file(filepath):
    """Generate a SHA-256 hash of a file."""
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_today_date():
    """Return today's date as a string (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")


def ensure_runs_dir(runs_dir="runs"):
    """Ensure the runs directory exists."""
    runs_path = Path(runs_dir)
    runs_path.mkdir(exist_ok=True)
    return runs_path


def create_run_folder(config_path, pattern_name, seed, runs_dir="runs"):
    """Create a unique run folder based on the config hash and today's date."""

    config_hash = hash_file(config_path)
    today = get_today_date()
    run_folder_name = (
        f"{today}_{pattern_name}_{seed}_{config_hash[:8]}"  # Short hash for readability
    )
    run_path = ensure_runs_dir(runs_dir) / run_folder_name

    return run_path


def make_debug_sim_params(simulation_params):
    simulation_params.training_epochs = 2
    simulation_params.replay_epochs = 5

    return simulation_params


def main(parameter_tag, saving, debug):
    import tomllib as toml
    from pathlib import Path

    import tomli_w as toml_w
    from utils import dict_to_namespace

    path = Path(__file__).parent.resolve()
    scripts_dir = path if (path / "run_experiments.py").exists() else path.parent.parent
    mlflow.set_tracking_uri(f"sqlite:///{scripts_dir / 'mlflow.db'}")

    # Load config — only config.toml is used; experiment.toml is not merged here.
    with open(path / "config.toml", "rb") as f:
        config_dict = toml.load(f)

    full_config = dict_to_namespace(config_dict)

    exp_config = full_config.experiment_params

    rng = np.random.default_rng(exp_config.seed)

    config_path = Path("config.toml").resolve()
    pattern_name = "_".join(exp_config.patterns)

    # Set the seed in the config
    rng = np.random.default_rng(exp_config.seed)
    run_path = Path.cwd()

    # Create directories for artifacts, figures, and patterns
    artifact_path = run_path / "artifacts"
    figure_path = run_path / "figures"
    pattern_path = run_path / "patterns"
    artifact_path.mkdir(exist_ok=True)
    figure_path.mkdir(exist_ok=True)

    tags = [pattern_name]
    if parameter_tag:
        tags.append(parameter_tag)

    experiment_name = exp_config.mlflow_experiment
    if debug:
        experiment_name = "ELise-tests"
        tags.append("debug")

    if full_config.simulation_params and debug:
        full_config.simulation_params = make_debug_sim_params(
            full_config.simulation_params
        )

    run_name = f"{coolname.generate_slug(2)}-{datetime.now().strftime('%m%d-%H%M')}"

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(
        run_name=run_name,
        tags={t: "true" for t in tags},
    )
    mlflow.set_tag("group_tag", str(exp_config.group_tag))

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID: {run_id}")
    # Save run ID to a file
    with open(run_path / "run_id.txt", "w") as f:
        f.write(run_id)

    # Log config file and individual parameters to MLflow
    mlflow.log_artifact(str(config_path), artifact_path="parameters")
    with open(config_path, "rb") as f:
        raw_config = toml.load(f)
    flat_params = {}
    for section, params in raw_config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                flat_params[f"{section}.{key}"] = str(value)
    mlflow.log_params(flat_params)

    # Import your main function (assuming it's in the same directory as run.py)
    from train import main as train_main

    (
        network,
        dataloader,
        train_tracker,
        validation_tracker,
        replay_tracker,
        epoch_tracker,
    ) = train_main(
        full_config,
        pattern_path,
        rng,
    )

    network.save(artifact_path / "network.pkl")
    dataloader.save(artifact_path / "dataloader.pkl")
    validation_tracker.save_dict(artifact_path / "validation_dict.pkl")
    train_tracker.save_dict(artifact_path / "train_dict.pkl")
    replay_tracker.save_dict(artifact_path / "replay_dict.pkl")
    epoch_tracker.save_dict(artifact_path / "epoch_dict.pkl")
    mlflow.log_artifact(str(artifact_path / "network.pkl"))

    # copy the config file to the artifact path
    artifact_config_path = artifact_path / "config.toml"
    with open(artifact_config_path, "wb") as f:
        toml_w.dump(config_dict, f)

    if saving:
        for fname in [
            "network.pkl",
            "dataloader.pkl",
            "replay_dict.pkl",
            "train_dict.pkl",
            "epoch_dict.pkl",
            "validation_dict.pkl",
        ]:
            mlflow.log_artifact(str(artifact_path / fname))
        mlflow.set_tag("full_save", "true")

    from plotting import main as plotting_main

    plotting_main(full_config, run_path, artifact_path, figure_path)

    mlflow.end_run()


if __name__ == "__main__":
    # Get seed from sys using sys.argv
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(
        description="Run ELiSe experiment with a specified random seed."
    )
    parser.add_argument(
        "--param_tag", type=str, default="", help="Parameter tag for mlflow"
    )
    parser.add_argument(
        "--saving", action="store_true", help="Flag for saving artifacts"
    )

    parser.add_argument("--debug", action="store_true", help="Flag for debug mode")
    args = parser.parse_args()

    main(args.param_tag, args.saving, args.debug)
