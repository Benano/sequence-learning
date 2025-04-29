#!/usr/bin/env python3

"""Run a large function from a module that integrates multiple functionalities."""

from pathlib import Path

import pytest
from example_experiment import main

example_experiment_success = None


@pytest.fixture(scope="session")
def paths():
    path = Path(__file__).parent.resolve()
    artifacts_path = path / "artifacts"
    artifacts_path.mkdir(exist_ok=True)
    return {"path": path, "artifacts_path": artifacts_path}


@pytest.mark.parametrize("numba_disable_jit_val", ["0", "1"])
def test_example_experiment(monkeypatch, paths, numba_disable_jit_val):
    """Test the execution both with and without numba enabled."""
    global example_experiment_success
    monkeypatch.setenv("NUMBA_DISABLE_JIT", numba_disable_jit_val)
    try:
        main(paths["path"], paths["artifacts_path"])
        example_experiment_success = True
    except Exception as e:
        example_experiment_success = False
        pytest.fail(
            f"Smoketest with 'example_experiment failed with exception:\n{e}"  # noqa
        )  # noqa


@pytest.mark.parametrize("fname", ["train_tracker.pkl", "replay_tracker.pkl"])
def test_tracker_results(paths, fname):
    if not example_experiment_success:
        pytest.skip()
    print(f"test tracker result file: {fname}")
    from elise.tracker import Tracker

    fpath = paths["artifacts_path"] / fname
    try:
        data = Tracker.load(fpath)  # noqa
    except FileNotFoundError as e:
        pytest.fail(f"The file {fpath} was not found:\n{e}")  # noqa
    except Exception as e:
        pytest.fail(f"Exception when loading {fpath}:\n{e}")  # noqa


@pytest.mark.parametrize("fname", ["network.pkl"])
def test_load_network(paths, fname):
    if not example_experiment_success:
        pytest.skip()
    print(f"test network state file: {fname}")
    from elise.model import Network

    fpath = paths["artifacts_path"] / fname
    try:
        nw = Network.load(fpath)  # noqa
    except FileNotFoundError as e:
        pytest.fail(f"The file {fpath} was not found:\n{e}")  # noqa
    except Exception as e:
        pytest.fail(f"Exception when loading {fpath}:\n{e}")  # noqa
