# import ast
import copy
from collections import defaultdict

import neptune
import numpy as np
from tqdm import tqdm
from utils import load_pattern_flexible

from elise.data import Dataloader, MultiPatternDataloader
from elise.data_transforms import CorrelatedNoise, WhiteNoise
from elise.model import Network, eq_phi  # noqa
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.stats import compute_loss, mse
from elise.tracker import Tracker
from elise.weights import DendriticWeights, SomaticWeights


def main(full_config, run_path, artifact_path, pattern_path, neptune_run, rng):
    experiment_params = full_config.experiment_params
    neuron_params = full_config.neuron_params
    network_params = full_config.network_params
    simulation_params = full_config.simulation_params
    weight_params = full_config.weight_params
    track_params = full_config.tracking_params

    patterns = []
    for pattern_name in experiment_params.patterns:
        pattern_file = pattern_path / f"{pattern_name}.txt"
        pattern = load_pattern_flexible(
            pattern_file,
            simulation_params.pattern_duration,
            simulation_params.pattern_dt,
        )
        patterns.append(pattern)

    def to_biounits(x):
        return neuron_params.E_l + x * 20.0

    online_transforms = []
    if simulation_params.noise_sigma > 0:
        if simulation_params.noise_tau > 0:
            online_transforms.append(
                CorrelatedNoise(
                    simulation_params.noise_sigma,
                    simulation_params.noise_tau,
                    simulation_params.dt,
                )
            )

        else:
            online_transforms.append(WhiteNoise(simulation_params.noise_sigma))

    if len(patterns) > 1:
        dataloader = MultiPatternDataloader(
            patterns=patterns,
            pre_transform=[to_biounits],
            online_transform=online_transforms,
        )
        dummyloader = MultiPatternDataloader(patterns)
    else:
        dataloader = Dataloader(
            patterns[0],
            pre_transforms=[to_biounits],
            online_transforms=online_transforms,
        )
        dummyloader = Dataloader(patterns[0])

    # Set seed to experiment_params.seed
    np.random.seed(experiment_params.seed)

    import matplotlib.pyplot as plt

    # Create imshow of pattern
    target_pattern = dataloader.get_full_pattern(simulation_params.dt)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(target_pattern.T, aspect="auto", cmap="gray", interpolation="none")
    ax.set_title("Input Pattern")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neurons")
    # clean_target_pattern = dummyloader.get_full_pattern(simulation_params.dt)

    if neptune_run:
        neptune_run["pattern"].upload(fig)

    # Network
    rate_buffer = Buffer
    dendritic_weights = DendriticWeights(weight_params, rng)
    somatic_weights = SomaticWeights(weight_params, rng)
    network_params.num_vis = dataloader.width
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

    u_target = dummyloader.get_full_pattern(dt)[:: track_params.sim_step]
    r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)

    # Sim params
    training_duration = simulation_params.training_cycles * dataloader.duration
    validation_duration = simulation_params.validation_cycles * dataloader.duration
    replay_duration = simulation_params.replay_cycles * dataloader.duration

    # Sim Trackers
    train_tracker = Tracker(track_params.vars_train, track_params.sim_step)
    validation_tracker = Tracker(track_params.vars_val, track_params.sim_step)
    replay_tracker = Tracker(track_params.vars_replay, track_params.sim_step)

    c_t = 0.0
    losses = defaultdict(list)
    for epoch in tqdm(range(simulation_params.training_epochs)):
        for t in np.arange(0, training_duration, simulation_params.dt):
            network(u_inp=dataloader(t))

            # Only last epoch
            c_t = c_t + simulation_params.dt
            if epoch >= simulation_params.training_epochs - 1:
                train_tracker.track(network, c_t)

        # Add learning rate decay
        network.optimizer_vis.eta *= simulation_params.eta_decay
        network.optimizer_lat.eta *= simulation_params.eta_decay

        # Validation
        if epoch != simulation_params.training_epochs - 1:
            for t in np.arange(0, validation_duration, simulation_params.dt):
                c_t = c_t + simulation_params.dt
                network(u_inp=None, learn=False)
                validation_tracker.track(network, c_t)

            u_out = np.array(validation_tracker["u_visible"])[-2 * len(u_target) :]
            r_out = np.array(validation_tracker["r_visible"])[-2 * len(u_target) :]

            mse_loss_u = compute_loss(u_out, u_target, mse)
            mse_loss_r = compute_loss(r_out, r_target, mse)

            if isinstance(dataloader, MultiPatternDataloader):
                widths = dataloader.widths
                c_width = 0
                for i in range(len(patterns)):
                    mse_loss_r = compute_loss(
                        r_out[:, c_width : c_width + widths[i]],
                        r_target[:, c_width : c_width + widths[i]],
                        mse,
                    )
                    c_width += widths[i]
                    neptune_run[f"validation_loss_pat_{i}"].append(mse_loss_r)
                    losses[f"validation_loss_pat_{i}"].append(mse_loss_r)
            else:
                pass

            if neptune_run:
                neptune_run["validation_loss_r"].append(mse_loss_r)

            losses["validation_loss_r"].append(mse_loss_r)
            losses["validation_loss_u"].append(mse_loss_u)

    train_tracker.store("r_target", r_target)
    validation_tracker.store("r_target", r_target)
    validation_tracker.store("losses", losses)

    if isinstance(dataloader, MultiPatternDataloader):
        first = dataloader.widths[0]
    else:
        first = 10

    partial_replay = True
    if partial_replay:
        network.reset_activity()

    # Create dictionary to store losses that uses list as value
    losses = defaultdict(list)

    replay_network = copy.deepcopy(network)

    for epoch in tqdm(range(simulation_params.replay_epochs)):
        for t in np.arange(0, replay_duration, simulation_params.dt):
            if partial_replay and epoch <= 1:
                u_inp = copy.deepcopy(replay_network.get_val("u", "visible"))
                u_tar = dataloader(t)[:first]
                u_inp[:first] = u_tar

                replay_network(u_inp=u_inp, learn=False)

            else:
                replay_network(u_inp=None, learn=False)

            replay_tracker.track(replay_network, t)

        u_out = np.array(replay_tracker["u_visible"])[-2 * len(u_target) :]
        r_out = np.array(replay_tracker["r_visible"])[-2 * len(u_target) :]

        mse_loss_u = compute_loss(u_out, u_target, mse)
        mse_loss_r = compute_loss(r_out, r_target, mse)

        if neptune_run:
            neptune_run["replay_loss_r"].append(mse_loss_r)

        if isinstance(dataloader, MultiPatternDataloader):
            widths = dataloader.widths
            c_width = 0
            for i in range(len(patterns)):
                mse_loss_r = compute_loss(
                    r_out[:, c_width : c_width + widths[i]],
                    r_target[:, c_width : c_width + widths[i]],
                    mse,
                )
                c_width += widths[i]
                neptune_run[f"replay_loss_pat_{i}"].append(mse_loss_r)
                losses[f"replay_loss_pat_{i}"].append(mse_loss_r)

        losses["replay_loss_r"].append(mse_loss_r)
        losses["replay_loss_u"].append(mse_loss_u)

    replay_tracker.store("losses", losses)
    replay_tracker.store("r_target", r_target)

    return network, dataloader, train_tracker, validation_tracker, replay_tracker


if __name__ == "__main__":
    from pathlib import Path

    from elise.config import FullConfig

    path = Path(__file__).parent.resolve()
    artifact_path = path / "artifacts"
    figure_path = path / "figures"
    config_path = path / "config.toml"
    full_config = FullConfig(config_path)

    #    with open("run_id.txt", "r") as f:
    #        run_id = f.read().strip()

    neptune_run = neptune.init_run(
        project="elise-neurotma/ELiSe",
        #        custom_run_id=run_id,
        name=path.name,
        tags=full_config.experiment_params.patterns,
    )

    rng = np.random.default_rng(full_config.experiment_params.seed)

    main(
        full_config,
        run_path=path,
        artifact_path=artifact_path,
        pattern_path=path / "patterns",
        neptune_run=neptune_run,
        rng=rng,
    )
