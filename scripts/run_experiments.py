#!/usr/bin/env python3
import shutil
import subprocess
import tomllib
from pathlib import Path

import tomli_w


def deep_merge(base, overrides):
    """Recursively merges overrides into base."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def run_experiment(exp_toml_path, master_cfg_path):
    exp_id = exp_toml_path.stem
    # Use a timestamp or unique ID if you plan to run the same TOML multiple times
    run_dir = Path("cluster_runs") / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. BAKE THE CONFIG
    with open(master_cfg_path, "rb") as f:
        config = tomllib.load(f)
    with open(exp_toml_path, "rb") as f:
        overlay = tomllib.load(f)

    final_config = deep_merge(config, overlay)

    with open(run_dir / "config.toml", "wb") as f:
        tomli_w.dump(final_config, f)

    # 2. SNAPSHOT THE CODE (The "Elegant" Copy)
    # Copy the main simulation script
    shutil.copy("run.py", run_dir / "run.py")
    shutil.copy("train.py", run_dir / "train.py")
    shutil.copy("utils.py", run_dir / "utils.py")  # Add other dependencies here
    shutil.copy("plotting.py", run_dir / "plotting.py")  # Add other dependencies here
    shutil.copy(
        "weight_metrics.py", run_dir / "weight_metrics.py"
    )  # Add other dependencies here
    shutil.copy("array.sh", run_dir / "array.sh")  # Add other dependencies here

    # Copy your library/package entirely
    if Path("elise").exists():
        shutil.copytree("elise", run_dir / "elise", dirs_exist_ok=True)

    # Copy assets
    if Path("patterns").exists():
        shutil.copytree("patterns", run_dir / "patterns", dirs_exist_ok=True)

    # 3. DISPATCH TO CLUSTER
    # Replace this with your sbatch/qsub command if using Slurm/PBS
    subprocess.Popen(["sbatch", "array.sh"], cwd=run_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run neural simulations.")
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Name of the experiment .toml file. If empty, runs all.",
    )

    parser.add_argument(
        "--debug", action="store_true", help="Flag for debug mode (no neptune logging)"
    )
    args = parser.parse_args()

    master_cfg = Path("config.toml")
    exp_folder = Path("experiments")

    if args.experiment:
        # Run specific experiment
        exp_file = exp_folder / f"{args.experiment}.toml"
        if not exp_file.exists():
            print(f"Error: {exp_file} not found.")
        else:
            run_experiment(exp_file, master_cfg)
    else:
        # Run all .toml files in the folder
        print("No specific experiment named. Processing all files in /experiments...")
        for exp_file in exp_folder.glob("*.toml"):
            run_experiment(exp_file, master_cfg)
