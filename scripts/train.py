# import ast
import copy
from collections import defaultdict

import neptune
import numpy as np
from tqdm import tqdm
from utils import load_pattern_flexible
from weight_metrics import analyze_connectivity_metrics

from elise.data import (
    Dataloader,
    MultiPatternDataloader,
    RandomCopiedNonMarkovianPattern,
    RandomPattern,
    RandomSampledNonMarkovianPattern,
)
from elise.data_transforms import (
    AddNothing,
    ChunkSplit,
    ChunkStep,
    ColorNotes,
    CorrelatedNoise,
    Silence,
    WhiteNoise,
)
from elise.model import Network, eq_phi  # noqa
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.stats import compute_loss, mse
from elise.tracker import Tracker
from elise.weights import DendriticWeights, RandomSomaticWeights, SomaticWeights


def main(
    full_config,
    pattern_path,
    neptune_run,
    rng,
):
    experiment_params = full_config.experiment_params
    neuron_params = full_config.neuron_params
    network_params = full_config.network_params
    simulation_params = full_config.simulation_params
    pattern_params = full_config.pattern_params
    weight_params = full_config.weight_params
    track_params = full_config.tracking_params

    patterns = []
    for pattern_name in experiment_params.patterns:
        if pattern_name == "random":
            if pattern_params.non_markov_type == "none":
                pattern_rng = copy.deepcopy(rng)
                pattern = RandomPattern(
                    width=network_params.num_vis,
                    duration=pattern_params.pattern_duration,
                    rng=pattern_rng,
                    dt=pattern_params.pattern_dt,
                )
                patterns.append(pattern)
            elif pattern_params.non_markov_type == "sampled":
                pattern_rng = copy.deepcopy(rng)
                pattern = RandomSampledNonMarkovianPattern(
                    width=network_params.num_vis,
                    duration=pattern_params.pattern_duration,
                    rng=pattern_rng,
                    dt=pattern_params.pattern_dt,
                    nmk=pattern_params.non_markov,
                )
                patterns.append(pattern)
            elif pattern_params.non_markov_type == "copied":
                pattern_rng = copy.deepcopy(rng)
                pattern = RandomCopiedNonMarkovianPattern(
                    width=network_params.num_vis,
                    duration=pattern_params.pattern_duration,
                    rng=pattern_rng,
                    dt=pattern_params.pattern_dt,
                    nmk=pattern_params.non_markov,
                )
                patterns.append(pattern)
        else:
            pattern_file = pattern_path / f"{pattern_name}.txt"
            pattern = load_pattern_flexible(
                pattern_file,
                pattern_params.pattern_duration,
                pattern_params.pattern_dt,
            )
            patterns.append(pattern)

    def to_biounits(x):
        return neuron_params.E_l + x * 20.0

    def harder_softer(x):
        return x * np.random.uniform(0.3, 1, x.shape)

    pre_transform_dict = {
        "to_biounits": to_biounits,
        "harder_softer": harder_softer,
        "color": ColorNotes(),
        "silence": Silence(),
        "chunk_split": ChunkSplit(num_chunks=3),
        "chunk_step": ChunkStep(),
        "nothing": AddNothing(50),
    }

    online_transform_dict = {
        "noise_corr": CorrelatedNoise(
            pattern_params.noise_sigma,
            pattern_params.noise_tau,
            simulation_params.dt,
        ),
        "noise_white": WhiteNoise(pattern_params.noise_sigma),
    }

    pre_transforms = []
    for transform in experiment_params.pre_transforms:
        pre_transforms.append(pre_transform_dict[transform])

    online_transforms = []
    for transform in experiment_params.online_transforms:
        online_transforms.append(online_transform_dict[transform])

    if len(patterns) > 1:
        dataloader = MultiPatternDataloader(
            patterns=patterns,
            pre_transform=pre_transforms,
            online_transform=online_transforms,
        )
    else:
        dataloader = Dataloader(
            patterns[0],
            pre_transforms=pre_transforms,
            online_transforms=online_transforms,
        )

    import matplotlib.pyplot as plt

    # Create imshow of pattern
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(pattern[:].T, aspect="auto", cmap="gray", interpolation="none")
    ax.set_title("Input Pattern")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neurons")

    if neptune_run:
        neptune_run["pattern"].upload(fig)

    # close fig
    fig.clf()

    # Spawn 4 independent child SeedSequences
    child_seeds = rng.bit_generator._seed_seq.spawn(4)
    child_rngs = [np.random.default_rng(s) for s in child_seeds]

    # If fixed seed available, override the corresponding spawned RNG
    d_den_rng = (
        np.random.default_rng(weight_params.d_den_seed)
        if weight_params.d_den_seed != -1
        else child_rngs[0]
    )

    d_som_rng = (
        np.random.default_rng(weight_params.d_som_seed)
        if weight_params.d_som_seed != -1
        else child_rngs[1]
    )

    w_den_rng = (
        np.random.default_rng(weight_params.w_den_seed)
        if weight_params.w_den_seed != -1
        else child_rngs[2]
    )

    w_som_rng = (
        np.random.default_rng(weight_params.w_som_seed)
        if weight_params.w_som_seed != -1
        else child_rngs[3]
    )

    # Network
    rate_buffer = Buffer
    dendritic_weights = DendriticWeights(
        weight_params, rng_w=w_den_rng, rng_d=d_den_rng
    )

    somatic_weight_types = {
        "developed": SomaticWeights,
        "random": RandomSomaticWeights,
    }

    weight_type = somatic_weight_types[weight_params.weight_type]
    somatic_weights = weight_type(weight_params, rng_w=w_som_rng, rng_d=d_som_rng)
    network_params.num_vis = dataloader.width
    # if network_params.load_in:
    #     network = Network.load("network.pkl")
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

    u_target = dataloader.get_full_pattern(dt, online_transforms=False)[
        :: track_params.sim_step
    ]
    r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)

    u_target_real = dataloader.get_full_pattern(dt, num=10, online_transforms=True)[
        :: track_params.sim_step
    ]
    r_target_real = eq_phi(u_target_real, neuron_params.a, neuron_params.b)

    # Sim params
    training_duration = simulation_params.training_cycles * dataloader.duration
    validation_duration = simulation_params.validation_cycles * dataloader.duration
    replay_duration = simulation_params.replay_cycles * dataloader.duration

    # Sim Trackers
    train_tracker = Tracker(track_params.vars_train, track_params.sim_step)
    validation_tracker = Tracker(track_params.vars_val, track_params.sim_step)
    replay_tracker = Tracker(track_params.vars_replay, track_params.sim_step)
    epoch_tracker = Tracker(track_params.vars_epoch, 1)

    c_t = 0.0
    nr_epochs = simulation_params.training_epochs
    losses = defaultdict(list)

    for epoch in tqdm(range(nr_epochs)):
        epoch_tracker.track(network, c_t)
        for t in np.arange(0, training_duration, simulation_params.dt):
            network(u_inp=dataloader(t))

            if track_params.track_training:
                c_t = c_t + simulation_params.dt
                first_patterns = 5 * dataloader.duration
                if t < first_patterns and t > dataloader.duration:
                    train_tracker.track(network, c_t)
            else:
                if epoch > nr_epochs - 1 and t > training_duration - (
                    2 * dataloader.duration
                ):
                    train_tracker.track(network, c_t)

        # Add learning rate decay
        network.optimizer_vis.eta *= simulation_params.eta_decay
        network.optimizer_lat.eta *= simulation_params.eta_decay

        # Validation
        if epoch != simulation_params.training_epochs - 1:
            for t in np.arange(0, validation_duration, simulation_params.dt):
                v_c_t = copy.deepcopy(c_t)
                v_c_t = c_t + simulation_params.dt
                network(u_inp=None, learn=False)
                validation_tracker.track(network, v_c_t)

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
                    if neptune_run:
                        neptune_run[f"validation_loss_pat_{i}"].append(mse_loss_r)
                        losses[f"validation_loss_pat_{i}"].append(mse_loss_r)
            else:
                pass

            if neptune_run:
                neptune_run["validation_loss_r"].append(mse_loss_r)

            losses["validation_loss_r"].append(mse_loss_r)
            losses["validation_loss_u"].append(mse_loss_u)

    train_tracker.store("r_target", r_target)
    train_tracker.store("r_target_real", r_target_real)
    validation_tracker.store("r_target", r_target)
    validation_tracker.store("losses", losses)

    som_w_metrics = analyze_connectivity_metrics(
        somatic_weights.weight_matrix, network.num_vis, cycles=False
    )
    den_w_metrics = analyze_connectivity_metrics(
        dendritic_weights.weight_matrix, network.num_vis, cycles=False
    )
    neptune_run["weight_metrics/dendric"] = den_w_metrics
    neptune_run["weight_metrics/somatic"] = som_w_metrics

    if isinstance(dataloader, MultiPatternDataloader):
        first = dataloader.widths[0]
    else:
        first = 5

    if experiment_params.partial_replay:
        network.reset_activity()

    # Create dictionary to store losses that uses list as value
    losses = defaultdict(list)
    replay_network = copy.deepcopy(network)
    disruption = experiment_params.disruption

    for epoch in tqdm(range(simulation_params.replay_epochs)):
        for t in np.arange(0, replay_duration, simulation_params.dt):
            if (
                disruption != 0
                and epoch == 0
                and t > dataloader.duration
                and t < 2 * dataloader.duration
            ):
                replay_network.set_visible_activity(disruption)
            else:
                pass

            if (
                experiment_params.partial_replay
                and epoch == 1
                and t < dataloader.duration / 4
            ):
                u_inp = copy.deepcopy(replay_network.get_val("u", "visible"))
                u_tar = dataloader(t)[:first]
                u_inp[:first] = u_tar
                replay_network(u_inp=u_inp, learn=experiment_params.replay_learning)
            else:
                replay_network(u_inp=None, learn=experiment_params.replay_learning)

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
                if neptune_run:
                    neptune_run[f"replay_loss_pat_{i}"].append(mse_loss_r)
                    losses[f"replay_loss_pat_{i}"].append(mse_loss_r)

        losses["replay_loss_r"].append(mse_loss_r)
        losses["replay_loss_u"].append(mse_loss_u)

    replay_tracker.store("losses", losses)
    replay_tracker.store("r_target", r_target)

    return (
        network,
        dataloader,
        train_tracker,
        validation_tracker,
        replay_tracker,
        epoch_tracker,
    )


if __name__ == "__main__":
    import tomllib as toml
    from pathlib import Path

    from utils import deep_merge, dict_to_namespace

    path = Path(__file__).parent.resolve()

    # 1. Load the merged config (this is the one created by the Runner script)
    with open(path / "config.toml", "rb") as f:
        config_dict = toml.load(f)

    with open(path / "experiment.toml", "rb") as f:
        experiment_config = toml.load(f)

    full_config = deep_merge(config_dict, experiment_config)

    # 2. Convert to the dot-notation object
    full_config = dict_to_namespace(config_dict)

    neptune_run = neptune.init_run(
        project="elise-neurotma/Elise-tests",
        #        custom_run_id=run_id,
        name=path.name,
        tags=full_config.patterns,
    )

    rng = np.random.default_rng(full_config.seed)

    main(
        full_config,  # Now you can use full_config.neuron_params.E_l again!
        run_path=path,
        neptune_run=neptune_run,
        rng=rng,
    )
