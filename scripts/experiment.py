#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm

from elise.config import FullConfig
from elise.data import DiscreteDataloader
from elise.model import Network, eq_phi
from elise.stats import compute_loss, mse
from elise.tracker import Tracker


def main(full_config, run_path, artifact_path, pattern_path, neptune_run):
    neuron_params = full_config.neuron_params
    simulation_params = full_config.simulation_params
    track_params = full_config.tracking_params

    # Load network
    network = Network.load(
        artifact_path / "network.pkl",
    )
    dataloader = DiscreteDataloader.load(artifact_path / "dataloader.pkl")
    dataloader.online_transforms = []

    # Every track params sim_step
    dt = network.dt
    u_target = dataloader.get_full_pattern(dt)[:: track_params.sim_step]
    r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)

    replay_duration = simulation_params.replay_cycles * dataloader.duration

    replay_tracker = Tracker(track_params.vars_replay, track_params.sim_step)

    # replay
    replay_loss_u = []
    replay_loss_r = []

    # replay
    replay_loss_u = []
    replay_loss_r = []

    first = 25
    partial_replay = False
    if partial_replay:
        network.reset_activity()

    import copy

    for epoch in tqdm(range(simulation_params.replay_epochs)):
        for t in np.arange(0, replay_duration, simulation_params.dt):
            # only the first 32 rows

            if partial_replay:
                u_inp = copy.deepcopy(network.get_val("u", "visible"))
                u_tar = dataloader(t)[:first]
                u_inp[:first] = u_tar

                network(u_inp=u_inp, learn=False)

            else:
                network(u_inp=None, learn=False)

            replay_tracker.track(network, t)

        u_out = np.array(replay_tracker["u_visible"])[-2 * len(u_target) :]
        r_out = np.array(replay_tracker["r_visible"])[-2 * len(u_target) :]

        mse_loss_u = compute_loss(u_out, u_target, mse)
        mse_loss_r = compute_loss(r_out, r_target, mse)

        if neptune_run:
            neptune_run["replay_loss_r"].append(mse_loss_r)

        replay_loss_r.append(mse_loss_r)
        replay_loss_u.append(mse_loss_u)

    replay_tracker.store("r_target", r_target)
    replay_tracker.store("mse_loss_r", replay_loss_r)
    replay_tracker.store("mse_loss_u", replay_loss_u)

    return replay_tracker


if __name__ == "__main__":
    from pathlib import Path

    import neptune

    path = Path(__file__).parent.resolve()
    artifact_path = path / "artifacts"
    figure_path = path / "figures"
    config_path = path / "config.toml"
    full_config = FullConfig(config_path)

    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()

    neptune_run = neptune.init_run(
        project="elise-neurotma/ELiSe",
        custom_run_id=run_id,
        name=path.name,
        tags=full_config.experiment_params.patterns,
    )

    main(
        full_config,
        run_path=path,
        artifact_path=artifact_path,
        pattern_path=path / "patterns",
        neptune_run=neptune_run,
    )
