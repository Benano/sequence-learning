#!/usr/bin/env python3

"""A simple experiment script to test if a minimal example runs through.

Should contain all major features and should be kept somethat up to date.
Works as a sanity check for the API etc. Does not test whether the network
actually learns, does the numerics correctly.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from elise.config import FullConfig
from elise.data import Dataloader, MultiHotPattern
from elise.model import Network, eq_phi  # noqa
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.stats import mse, window_slider
from elise.tracker import Tracker
from elise.weights import DendriticWeights, SomaticWeights


def main(path, artifacts_path):
    # Config
    full_config = FullConfig(path / "smoketest_config.toml")
    neuron_params = full_config.neuron_params
    network_params = full_config.network_params
    simulation_params = full_config.simulation_params
    weight_params = full_config.weight_params
    track_params = full_config.tracking_params

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

    def to_biounits(x):
        return neuron_params.E_l + x * 20.0

    elise = np.loadtxt(path / "fuer_elise_short.txt", skiprows=1, delimiter=",").astype(
        int
    )

    pattern = MultiHotPattern(
        pattern=elise,
        duration=simulation_params.pattern_duration,
        width=network_params.num_vis,
    )
    loader = Dataloader(pattern, pre_transforms=[to_biounits])
    u_target = loader.get_full_pattern(dt)

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

    for epoch in tqdm(range(simulation_params.training_epochs)):
        for t in np.arange(0, training_duration, simulation_params.dt):
            network(u_inp=loader(t))

            # Only record in last epoch
            if epoch == simulation_params.training_epochs - 1:
                train_tracker.track(t)

        # Validation
        if epoch != simulation_params.training_epochs - 1:
            for t in np.arange(0, validation_duration, simulation_params.dt):
                network(u_inp=None)
                validation_tracker.track(t)

            u_out = np.array(validation_tracker["u_visible"])
            mse_loss = np.min(window_slider(u_out, u_target, mse))

        else:
            print("Last epoch, not tracking validation.")

        print(f"Epoch {epoch} -  MSE u: {mse_loss}")  # noqa

    network.save(artifacts_path / "network.pkl")

    # replay
    for t in np.arange(0, replay_duration, simulation_params.dt):
        network(u_inp=None)
        replay_tracker.track(t)

    u_out = np.array(replay_tracker["u_visible"])
    replay_losses = window_slider(u_out, u_target, mse)

    fig, ax = plt.subplots()
    ax.plot(replay_losses)
    ax.set_xlabel("time")
    ax.set_ylabel("mse")

    fig.savefig(artifacts_path / "replay_losses.png")

    train_tracker.save(artifacts_path / "train_tracker.pkl")
    replay_tracker.save(artifacts_path / "replay_tracker.pkl")


if __name__ == "__main__":
    path = Path(__file__).parent.resolve()
    artifacts_path = path / "artifacts"
    artifacts_path.mkdir(exist_ok=True)

    main(path, artifacts_path)
    print("main() runs through.")
