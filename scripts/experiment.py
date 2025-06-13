
"""
A simple experiment script
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm
import neptune

from elise.config import FullConfig
from elise.data import Dataloader, MultiHotPattern, Pattern
from elise.model import Network, eq_phi  # noqa
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.stats import mse, window_slider
from elise.tracker import Tracker
from elise.weights import DendriticWeights, SomaticWeights
import ast

def compute_loss(output, target, loss_function):
    metric = window_slider(output, target, loss_function)
    loss = np.min(metric)
    return loss


def load_multi_hot_pattern(filename):
    with open(filename, "r") as file:
        content = file.read().strip()

    pat = ast.literal_eval(f"[{content}]")
    return pat

def load_one_hot_pattern(pattern_file):
    pat = np.loadtxt(pattern_file, delimiter=" ").astype(int).T
    pat = pat[:,pat.any(axis=0)]

    return pat

def load_pattern_flexible(pattern_file):
    """
    Try to load a pattern file as multi-hot; if that fails, load as one-hot.
    Returns a numpy array.
    """
    try:
        # Try multi-hot
        pat = load_multi_hot_pattern(pattern_file)
        pat_type = "multi-hot"

    except (ValueError, SyntaxError):
        # Try one-hot
        pat = load_one_hot_pattern(pattern_file)
        pat_type = "one-hot"

    return pat, pat_type


def main(full_config, run_path, artifact_path, pattern_path, neptune_run):

    experiment_params = full_config.experiment_params
    neuron_params = full_config.neuron_params
    network_params = full_config.network_params
    simulation_params = full_config.simulation_params
    weight_params = full_config.weight_params
    track_params = full_config.tracking_params

    pattern_file = pattern_path / f"{experiment_params.pattern}.txt"
    pattern, pattern_type = load_pattern_flexible(pattern_file)

    if len(pattern) > 300:
        pattern = pattern[:,:250]

    if pattern_type == "multi-hot":
        pattern = MultiHotPattern(
            pattern=pattern,
            duration=simulation_params.pattern_duration,
            width=network_params.num_vis)

    elif pattern_type == "one-hot":
        pattern = Pattern(
            pattern=pattern,
            duration=simulation_params.pattern_duration)

    def to_biounits(x):
        return neuron_params.E_l + x * 20.0

    loader = Dataloader(pattern, pre_transforms=[to_biounits])

    # Network
    rate_buffer = Buffer
    dendritic_weights = DendriticWeights(weight_params)
    somatic_weights = SomaticWeights(weight_params)
    network = Network(
        network_params, neuron_params, dendritic_weights, somatic_weights, rate_buffer
    )

    # Simulator
    optimizer = SimpleUpdater
    dt = simulation_params.dt
    eta_lat = simulation_params.eta_lat
    eta_vis = simulation_params.eta_out
    optimizer_lat = optimizer(eta_lat)
    optimizer_vis = optimizer(eta_vis)
    network.prepare_for_simulation(dt, optimizer_vis, optimizer_lat)

    # Every track params sim_step
    u_target = loader.get_full_pattern(dt)[::track_params.sim_step]
    r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)

    # Sim params
    training_duration = (
        simulation_params.training_cycles * simulation_params.pattern_duration
    )
    validation_duration = (
        simulation_params.validation_cycles * simulation_params.pattern_duration
    )
    replay_duration = (
        simulation_params.replay_cycles * simulation_params.pattern_duration
    )

    # Sim Trackers
    train_tracker = Tracker(network, track_params.vars_train, track_params.sim_step)
    validation_tracker = Tracker(network, track_params.vars_val, track_params.sim_step)
    replay_tracker = Tracker(network, track_params.vars_replay, track_params.sim_step)

    validation_loss_u = []
    validation_loss_r = []

    for epoch in tqdm(range(simulation_params.training_epochs)):
        for t in np.arange(0, training_duration, simulation_params.dt):

            network(u_inp=loader(t))

            # Only track the last two epochs
            if epoch >= simulation_params.training_epochs - 2:
                train_tracker.track(t)

        # Add learning rate decay
        network.optimizer_vis.eta *= 0.95
        network.optimizer_lat.eta *= 0.95

        # Validation
        # if epoch != simulation_params.training_epochs - 1:
        for t in np.arange(0, validation_duration, simulation_params.dt):
            network(u_inp=None)
            validation_tracker.track(t)

        u_out = np.array(validation_tracker["u_visible"])[-2 * len(u_target) :]
        r_out = np.array(validation_tracker["r_visible"])[-2 * len(u_target) :]

        mse_loss_u = compute_loss(u_out, u_target, mse)
        mse_loss_r = compute_loss(r_out, r_target, mse)

        if neptune_run:
            neptune_run["validation_loss_r"].append(mse_loss_r)

        validation_loss_r.append(mse_loss_r)
        validation_loss_u.append(mse_loss_u)

        print(f"Epoch {epoch} -  MSE u: {mse_loss_u}")  # noqa
        print(f"Epoch {epoch} -  MSE r: {mse_loss_r}")  # noqa

    network.save(artifact_path / "network.pkl")

    # replay
    replay_loss_u = []
    replay_loss_r = []
    for epoch in tqdm(range(simulation_params.replay_epochs)):
        for t in np.arange(0, replay_duration, simulation_params.dt):
            network(u_inp=None)
            replay_tracker.track(t)


        u_out = np.array(replay_tracker["u_visible"])[-2 * len(u_target) :]
        r_out = np.array(replay_tracker["r_visible"])[-2 * len(u_target) :]

        mse_loss_u = compute_loss(u_out, u_target, mse)
        mse_loss_r = compute_loss(r_out, r_target, mse)

        if neptune_run:
            neptune_run["replay_loss_r"].append(mse_loss_r)

        replay_loss_r.append(mse_loss_r)
        replay_loss_u.append(mse_loss_u)

    train_tracker.store("r_target", r_target)
    validation_tracker.store("r_target", r_target)
    replay_tracker.store("r_target", r_target)

    validation_tracker.store("mse_loss_r", validation_loss_r)
    validation_tracker.store("mse_loss_u", validation_loss_u)
    replay_tracker.store("mse_loss_r", replay_loss_r)
    replay_tracker.store("mse_loss_u", replay_loss_u)

    train_tracker.save(artifact_path / "train_tracker.pkl")
    replay_tracker.save(artifact_path / "replay_tracker.pkl")
    validation_tracker.save(artifact_path / "validation_tracker.pkl")


if __name__ == "__main__":
    from pathlib import Path
    from elise.config import FullConfig

    path = Path(__file__).parent.resolve()
    artifact_path = path / "artifacts"
    figure_path = path / "figures"
    neptune_run = None
    config_path = path / "config.toml"
    full_config = FullConfig(config_path)

    main(
        full_config,
        run_path=path,
        artifact_path=artifact_path,
        pattern_path=path / "patterns",
        neptune_run=neptune_run
    )
