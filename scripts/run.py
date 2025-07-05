#!/usr/bin/env python3

import hashlib
import shutil
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
    run_path.mkdir(exist_ok=True)

    return run_path


def track_config(config, neptune_run):
    """Track the configuration in Neptune."""
    for section, params in config.items():
        for key, value in params.items():
            neptune_run[f"config/{section}/{key}"] = value


def main(seed, saving):
    from elise.config import FullConfig

    config_path = Path("config.toml").resolve()
    full_config = FullConfig(config_path)
    experiment_params = full_config.experiment_params
    pattern_name = "_".join(experiment_params.patterns)

    # Set the seed in the config
    rng = np.random.default_rng(seed)
    experiment_params.seed = seed

    runs_dir = "runs"
    run_path = create_run_folder(config_path, pattern_name, seed, runs_dir)

    # Create directories for artifacts, figures, and patterns
    artifact_path = run_path / "artifacts"
    figure_path = run_path / "figures"
    pattern_path = run_path / "patterns"
    artifact_path.mkdir(exist_ok=True)
    figure_path.mkdir(exist_ok=True)
    pattern_path.mkdir(exist_ok=True)

    # Copy necessary files to the run directory
    shutil.copy("train.py", run_path)
    shutil.copy("plotting.py", run_path)
    shutil.copy("experiment.py", run_path)
    shutil.copy("config.toml", run_path)
    pattern_folder = Path("patterns").resolve()

    if seed == 0:
        print("SEEEED 0 MOTHERFUCKER")

    for pattern in experiment_params.patterns:
        p_file = pattern_folder / (pattern + ".txt")
        shutil.copy(p_file, pattern_path)

    neptune_run = neptune.init_run(
        project="elise-neurotma/ELiSe",
        # custom_run_id=run_path.name[-16:],
        name=run_path.name,
        tags=[pattern_name],
    )
    neptune_run["sys/group_tags"].add(experiment_params.group_tag)

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

    network, dataloader, train_tracker, validation_tracker = train_main(
        full_config, run_path, artifact_path, pattern_path, neptune_run, rng
    )

    network.save(artifact_path / "network.pkl")
    validation_tracker.save(artifact_path / "validation_tracker.pkl")
    train_tracker.save(artifact_path / "train_tracker.pkl")
    dataloader.save(artifact_path / "dataloader.pkl")

    if seed == 0:
        print("SEEEED 0 MOTHERFUCKER")
        neptune_run["network"].upload(str(artifact_path / "network.pkl"))
        neptune_run["train_tracker"].upload(str(artifact_path / "train_tracker.pkl"))
        neptune_run["validation_tracker"].upload(
            str(artifact_path / "validation_tracker.pkl")
        )
        neptune_run["dataloader"].upload(str(artifact_path / "dataloader.pkl"))

    from experiment import main as experiment_main

    replay_tracker = experiment_main(
        full_config, run_path, artifact_path, pattern_path, neptune_run
    )
    replay_tracker.save(str(artifact_path / "replay_tracker.pkl"))

    if seed == 0:
        neptune_run["replay_tracker"].upload(str(artifact_path / "replay_tracker.pkl"))

    from plotting import main as plotting_main

    plotting_main(full_config, run_path, artifact_path, figure_path, neptune_run)


if __name__ == "__main__":
    # Get seed from sys using sys.argv
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(
        description="Run ELiSe experiment with a specified random seed."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for NumPy RNG"
    )
    parser.add_argument("--saving", default=False, help="Flag for savingd artifacts")
    args = parser.parse_args()

    main(args.seed, args.saving)
