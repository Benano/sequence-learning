#!/usr/bin/env python3

import hashlib
from pathlib import Path
from datetime import datetime
import shutil
import neptune
from elise.config import FullConfig
import tomllib as toml
from neptune.utils import stringify_unsupported

def hash_file(filepath):
    """Generate a SHA-256 hash of a file."""
    with open(filepath, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def get_today_date():
    """Return today's date as a string (YYYY-MM-DD)."""
    return datetime.now().strftime("%Y-%m-%d")

def ensure_runs_dir(runs_dir="runs"):
    """Ensure the runs directory exists."""
    runs_path = Path(runs_dir)
    runs_path.mkdir(exist_ok=True)
    return runs_path

def create_run_folder(config_path, pattern_name, runs_dir="runs"):
    """Create a unique run folder based on the config hash and today's date."""

    config_hash = hash_file(config_path)
    today = get_today_date()
    run_folder_name = f"{today}_{pattern_name}_{config_hash[:8]}"  # Short hash for readability
    run_path = ensure_runs_dir(runs_dir) / run_folder_name
    run_path.mkdir(exist_ok=True)

    return run_path

def track_config(config, neptune_run):
    """Track the configuration in Neptune."""
    for section, params in config.items():
        for key, value in params.items():
            neptune_run[f"config/{section}/{key}"] = value

def main():
    from elise.config import FullConfig

    config_path = Path("config.toml").resolve()
    full_config = FullConfig(config_path)
    experiment_params = full_config.experiment_params

    runs_dir = "runs"
    pattern_name = experiment_params.pattern
    run_path = create_run_folder(config_path, pattern_name, runs_dir)

    # Create directories for artifacts, figures, and patterns
    artifact_path = run_path / "artifacts"
    figure_path = run_path / "figures"
    pattern_path = run_path / "patterns"
    artifact_path.mkdir(exist_ok=True)
    figure_path.mkdir(exist_ok=True)
    pattern_path.mkdir(exist_ok=True)

    # Copy necessary files to the run directory
    shutil.copy('experiment.py', run_path)
    shutil.copy('plotting.py', run_path)
    shutil.copy('config.toml', run_path)
    pattern_folder = Path("patterns").resolve()
    pattern = pattern_folder / (experiment_params.pattern + ".txt")
    shutil.copy(pattern, pattern_path)

    neptune_run = neptune.init_run(
        project="elise-neurotma/ELiSe",
        custom_run_id=run_path.name[-16:],
        name = run_path.name,
        tags=[experiment_params.pattern],
    )

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
    from experiment import main as experiment_main
    experiment_main(full_config, run_path, artifact_path, pattern_path, neptune_run)

    from plotting import main as plotting_main
    plotting_main(full_config, run_path, artifact_path, figure_path, neptune_run)


if __name__ == "__main__":
    main()
