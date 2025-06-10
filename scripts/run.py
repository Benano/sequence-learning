#!/usr/bin/env python3

import hashlib
from pathlib import Path
from datetime import datetime
import shutil
import neptune
from elise.config import FullConfig

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

def create_run_folder(config_path, runs_dir="runs"):
    """Create a unique run folder based on the config hash and today's date."""

    config_hash = hash_file(config_path)
    today = get_today_date()
    run_folder_name = f"{today}_{config_hash[:8]}"  # Short hash for readability
    run_folder = ensure_runs_dir(runs_dir) / run_folder_name
    run_folder.mkdir(exist_ok=True)

    artifact_folder = run_folder / "artifacts"
    artifact_folder.mkdir(exist_ok=True)

    figures_folder = run_folder / "figures"
    figures_folder.mkdir(exist_ok=True)

    pattern_folder = run_folder / "patterns"
    pattern_folder.mkdir(exist_ok=True)

    shutil.copy(config_path, run_folder / config_path.name)

    return run_folder, artifact_folder, figures_folder, pattern_folder

def main():
    # Path to your config.toml (relative to the run script)
    #
    config_path = Path("config.toml").resolve()
    runs_dir = "runs"
    run_path, artifact_path, figure_path, pattern_path = create_run_folder(config_path, runs_dir)
    print(f"Created and using run folder: {run_path}")

    from elise.config import FullConfig
    full_config = FullConfig(config_path)
    experiment_params = full_config.experiment_params

    pattern_folder = Path("patterns").resolve()
    pattern = pattern_folder / (experiment_params.pattern + ".txt")

    # Creating Run folder
    shutil.copy('experiment.py', run_path)
    shutil.copy('plotting.py', run_path)
    shutil.copy(pattern, pattern_path)

    run_id = experiment_params.pattern + "_" + run_path.name

    neptune_run = neptune.init_run(
        project="elise-neurotma/ELiSe",
        custom_run_id=run_id)

    # Import your main function (assuming it's in the same directory as run.py)
    from experiment import main as experiment_main
    experiment_main(full_config, run_path, artifact_path, pattern_path, neptune_run)

    from plotting import main as plotting_main
    plotting_main(full_config, run_path, artifact_path, figure_path, neptune_run)

if __name__ == "__main__":
    main()
