#!/usr/bin/env python3

import hashlib
import tomllib as toml
from datetime import datetime
from pathlib import Path

import neptune
from neptune.utils import stringify_unsupported


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


def track_config(config, neptune_run):
    """Track the configuration in Neptune."""
    for section, params in config.items():
        for key, value in params.items():
            neptune_run[f"config/{section}/{key}"] = value


def make_debug_sim_params(simulation_params):
    simulation_params.training_cycles = 2
    simulation_params.training_epochs = 2
    simulation_params.replay_cycles = 2
    simulation_params.replay_epochs = 5

    return simulation_params


def main(parameter_tag, saving, debug):
    from elise.config import FullConfig

    print(saving)

    # load experiment config from experiment.toml
    with open("experiment.toml", "rb") as f:
        experiment_config = toml.load(f)

    config_path = Path("config.toml").resolve()
    full_config = FullConfig(config_path)
    pattern_name = "_".join(experiment_config["patterns"])

    # Set the seed in the config
    rng = np.random.default_rng(experiment_config["seed"])
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

    project_name = experiment_config["neptune_project"]

    if debug:
        full_config.simulation_params = make_debug_sim_params(
            full_config.simulation_params
        )
        tags.append("debug")
        neptune_run = neptune.init_run(
            project="elise-neurotma/Elise-tests",
            # custom_run_id=run_path.name[-16:],
            name=run_path.name,
            tags=tags,
        )
    else:
        neptune_run = neptune.init_run(
            project=f"elise-neurotma/{project_name}",
            # custom_run_id=run_path.name[-16:],
            name=run_path.name,
            tags=tags,
        )
        neptune_run["sys/group_tags"].add(experiment_config["group_tag"])

    run_id = neptune_run["sys/id"].fetch()
    print(f"Run ID: {run_id}")  # Print the run ID for reference
    # Save run ID to a file
    with open(run_path / "run_id.txt", "w") as f:
        f.write(run_id)

    # Save config file to Neptune
    neptune_run["parameters/config"].upload("config.toml")

    with open(config_path, "rb") as f:
        raw_config = toml.load(f)
    if neptune_run:
        raw_config = stringify_unsupported(raw_config)
        # Log each parameter
        for section, params in raw_config.items():
            for key, value in params.items():
                neptune_run[f"parameters/{section}/{key}"] = value

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
        experiment_config,
        full_config,
        run_path,
        artifact_path,
        pattern_path,
        neptune_run,
        rng,
    )

    network.save(artifact_path / "network.pkl")
    dataloader.save(artifact_path / "dataloader.pkl")
    validation_tracker.save_dict(artifact_path / "validation_dict.pkl")
    train_tracker.save_dict(artifact_path / "train_dict.pkl")
    replay_tracker.save_dict(artifact_path / "replay_dict.pkl")
    epoch_tracker.save_dict(artifact_path / "epoch_dict.pkl")
    neptune_run["network"].upload(str(artifact_path / "network.pkl"))

    if saving:
        neptune_run["network"].upload(str(artifact_path / "network.pkl"))
        neptune_run["dataloader"].upload(str(artifact_path / "dataloader.pkl"))
        neptune_run["replay_dict"].upload(str(artifact_path / "replay_dict.pkl"))
        neptune_run["train_dict"].upload(str(artifact_path / "train_dict.pkl"))
        neptune_run["epoch_dict"].upload(str(artifact_path / "epoch_dict.pkl"))
        neptune_run["validation_dict"].upload(
            str(artifact_path / "validation_dict.pkl")
        )
        neptune_run["sys/tags"].add("full_save")

    from plotting import main as plotting_main

    plotting_main(full_config, run_path, artifact_path, figure_path, neptune_run)

    neptune_run.stop()


if __name__ == "__main__":
    # Get seed from sys using sys.argv
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(
        description="Run ELiSe experiment with a specified random seed."
    )
    parser.add_argument(
        "--param_tag", type=str, default="", help="Parameter tag for neptune"
    )
    parser.add_argument(
        "--saving", action="store_true", help="Flag for savingd artifacts"
    )

    parser.add_argument(
        "--debug", action="store_true", help="Flag for debug mode (no neptune logging)"
    )
    args = parser.parse_args()

    main(args.param_tag, args.saving, args.debug)
