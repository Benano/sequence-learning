import copy
from collections import defaultdict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from tqdm import tqdm
from utils import load_pattern_flexible

from elise.data import (
    Dataloader,
    MultiPatternDataloader,
    RandomCopiedNonMarkovianPattern,
    RandomPattern,
    RandomSampledNonMarkovianPattern,
    ShuffleDataloader,
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


def load_patterns(experiment_params, pattern_params, network_params, pattern_path, rng):
    """Load or generate all training patterns specified in experiment_params."""
    patterns = []
    for pattern_name in experiment_params.patterns:
        if pattern_name == "random":
            pattern_rng = np.random.default_rng(rng.integers(2**63))
            if pattern_params.non_markov_type == "none":
                pattern = RandomPattern(
                    width=network_params.num_vis,
                    duration=pattern_params.pattern_duration,
                    rng=pattern_rng,
                    dt=pattern_params.pattern_dt,
                )
            elif pattern_params.non_markov_type == "sampled":
                pattern = RandomSampledNonMarkovianPattern(
                    width=network_params.num_vis,
                    duration=pattern_params.pattern_duration,
                    rng=pattern_rng,
                    dt=pattern_params.pattern_dt,
                    nmk=pattern_params.non_markov,
                )
            elif pattern_params.non_markov_type == "copied":
                pattern = RandomCopiedNonMarkovianPattern(
                    width=network_params.num_vis,
                    duration=pattern_params.pattern_duration,
                    rng=pattern_rng,
                    dt=pattern_params.pattern_dt,
                    nmk=pattern_params.non_markov,
                )
        else:
            pattern_file = pattern_path / f"{pattern_name}.txt"
            pattern = load_pattern_flexible(
                pattern_file,
                pattern_params.pattern_duration,
                pattern_params.pattern_dt,
            )

        patterns.append(pattern)

    return patterns


def build_dataloader(
    patterns, experiment_params, pattern_params, simulation_params, neuron_params
):
    """Construct a dataloader with the configured pre- and online transforms."""

    def to_biounits(x):
        return neuron_params.E_l + x * pattern_params.biounits_scale

    pre_transform_dict = {
        "to_biounits": to_biounits,
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

    pre_transforms = [pre_transform_dict[t] for t in experiment_params.pre_transforms]
    online_transforms = [
        online_transform_dict[t] for t in experiment_params.online_transforms
    ]

    dataloader_type = getattr(experiment_params, "dataloader_type", "multi")

    if len(patterns) > 1:
        if dataloader_type == "shuffle":
            t_max = (
                max(simulation_params.training_cycles, simulation_params.replay_cycles)
                * patterns[0].duration
                + 1.0
            )
            return ShuffleDataloader(
                pattern=patterns,
                t_max=t_max,
                pre_transforms=pre_transforms,
                online_transforms=online_transforms,
                block_size=getattr(experiment_params, "shuffle_block_size", 1),
            )

        elif dataloader_type == "stacked":
            return MultiPatternDataloader(
                patterns=patterns,
                pre_transform=pre_transforms,
                online_transform=online_transforms,
            )
    else:
        return Dataloader(
            patterns[0],
            pre_transforms=pre_transforms,
            online_transforms=online_transforms,
        )


def log_pattern_figure(dataloader, pattern_params):
    """Log an imshow of the training pattern to MLflow."""
    plot_pattern = dataloader.get_full_pattern(
        dt=pattern_params.pattern_dt, online_transforms=False
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(plot_pattern.T, aspect="auto", cmap="gray", interpolation="none")
    ax.set_title("Input Pattern")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neurons")
    mlflow.log_figure(fig, "pattern.png")
    # plt.show()
    fig.clf()


def setup_network(weight_params, network_params, neuron_params, rng):
    """Initialize dendritic and somatic weights, then build the network."""
    # Spawn 4 independent child SeedSequences — one per weight/delay matrix
    child_seeds = rng.bit_generator._seed_seq.spawn(4)
    child_rngs = [np.random.default_rng(s) for s in child_seeds]

    # Use fixed seeds if provided, otherwise use the spawned RNGs
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

    return Network(
        network_params, neuron_params, dendritic_weights, somatic_weights, rate_buffer
    )


def log_per_pattern_losses(r_out, r_target, widths, prefix, epoch, losses):
    """Compute and log per-pattern MSE losses for multi-pattern dataloaders."""
    c_width = 0
    for i, w in enumerate(widths):
        loss = compute_loss(
            r_out[:, c_width : c_width + w],
            r_target[:, c_width : c_width + w],
            mse,
        )
        c_width += w
        mlflow.log_metric(f"{prefix}_loss_pat_{i}", loss, step=epoch)
        losses[f"{prefix}_loss_pat_{i}"].append(loss)


def nudge_network(
    network,
    dataloader,
    nr_nudging_cycles: int,
    proportion_width: float = 1.0,
    proportion_height: float = 1.0,
    learn: bool = False,
) -> None:
    """Drive the network with a (possibly partial) pattern for nr_nudging_cycles
    complete presentations before free-recall measurement.

    :param proportion_width: Fraction of each cycle's duration during which the nudge
        signal is applied. After this window the network runs freely (u_inp=None).
    :param proportion_height: Fraction of visible neurons (from index 0) that receive
        the nudge signal. The remaining neurons always run freely.
    """
    n_nudge = int(proportion_height * network.num_vis)
    dt = network.dt
    duration = dataloader.duration
    nudge_cutoff = proportion_width * duration

    for _ in range(nr_nudging_cycles):
        for t in np.arange(0, duration, dt):
            if t < nudge_cutoff:
                pat_t = dataloader(t)
                u_inp = network.get_val("u", "visible")
                u_inp[:n_nudge] = pat_t[:n_nudge]
                network(u_inp=u_inp, learn=learn)
            else:
                network(u_inp=None, learn=learn)


def run_training(
    network,
    dataloader,
    simulation_params,
    neuron_params,
    noise_params,
    track_params,
    targets,
    trackers,
):
    """Run the supervised training loop with periodic validation."""
    u_target, r_target, r_target_real = targets
    train_tracker, validation_tracker, epoch_tracker = trackers

    c_t = 0.0
    nr_epochs = simulation_params.training_epochs
    losses = defaultdict(list)
    training_duration = simulation_params.training_cycles * dataloader.duration
    validation_duration = simulation_params.validation_cycles * dataloader.duration

    w_noise = CorrelatedNoise(
        noise_params.w_noise_sigma, noise_params.w_noise_tau, simulation_params.dt
    )
    u_noise = CorrelatedNoise(
        noise_params.u_noise_sigma, noise_params.u_noise_tau, simulation_params.dt
    )

    for epoch in tqdm(range(nr_epochs)):
        if hasattr(dataloader, "reshuffle"):
            dataloader.reshuffle()
        epoch_tracker.track(network, c_t)

        # record = []
        # for t in np.arange(0, training_duration, simulation_params.dt):
        #     record.append(dataloader(t))

        # record = np.array(record)
        # fig, ax = plt.subplots()
        # ax.imshow(record.T, aspect='auto', interpolation='none')
        # plt.show()
        # breakpoint()

        for t in np.arange(0, training_duration, simulation_params.dt):
            network(
                u_inp=dataloader(t),
                w_noise_gen=w_noise,
                u_noise_gen=u_noise,
                learn=True,
            )

            if track_params.track_training:
                c_t = c_t + simulation_params.dt
                first_patterns = (
                    track_params.early_tracking_patterns * dataloader.duration
                )
                if dataloader.duration < t < first_patterns:
                    train_tracker.track(network, c_t)
            else:
                if epoch > nr_epochs - 1 and t > training_duration - (
                    2 * dataloader.duration
                ):
                    train_tracker.track(network, c_t)

        # Learning rate decay
        network.optimizer_vis.eta *= simulation_params.eta_decay
        network.optimizer_lat.eta *= simulation_params.eta_decay

        # Validation (skipped on final epoch — network state is needed for replay)
        if epoch != simulation_params.training_epochs - 1:
            for en, pattern_dl in enumerate(dataloader):
                nudge_network(
                    network,
                    pattern_dl,
                    simulation_params.validation_cue_cycles,
                )

                u_target = pattern_dl.get_full_pattern(simulation_params.dt)[
                    :: track_params.sim_step
                ]
                r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)

                for t in np.arange(0, validation_duration, simulation_params.dt):
                    v_c_t = c_t + simulation_params.dt
                    network(u_inp=None, learn=False)
                    validation_tracker.track(network, v_c_t)

                u_out = np.array(validation_tracker["u_visible"])[-2 * len(u_target) :]
                r_out = np.array(validation_tracker["r_visible"])[-2 * len(u_target) :]
                mse_loss_u = compute_loss(u_out, u_target, mse)
                mse_loss_r = compute_loss(r_out, r_target, mse)

                mlflow.log_metric(f"val_loss_r_pat_{en}", mse_loss_r, step=epoch)
                losses[f"val_loss_u_pat_{en}"].append(mse_loss_u)
                losses[f"val_loss_r_pat_{en}"].append(mse_loss_r)

    train_tracker.store("r_target", r_target)
    train_tracker.store("r_target_real", r_target_real)
    validation_tracker.store("r_target", r_target)
    validation_tracker.store("losses", losses)


def run_replay(
    network,
    dataloader,
    simulation_params,
    neuron_params,
    noise_params,
    experiment_params,
    track_params,
    targets,
    replay_trackers,
):
    replay_duration = simulation_params.replay_cycles * dataloader.duration
    replay_network = copy.deepcopy(network)

    for en, pattern_dl in enumerate(dataloader):
        replay_network.reset_activity()
        replay_tracker = replay_trackers[en]
        losses = defaultdict(list)

        u_target = pattern_dl.get_full_pattern(simulation_params.dt)[
            :: track_params.sim_step
        ]
        r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)
        replay_tracker.store("r_target", r_target)

        nudge_network(
            replay_network,
            pattern_dl,
            simulation_params.replay_cue_cycles,
            learn=experiment_params.replay_learning,
        )

        for epoch in tqdm(range(simulation_params.replay_epochs)):
            for t in np.arange(0, replay_duration, simulation_params.dt):
                replay_network(u_inp=None, learn=experiment_params.replay_learning)
                replay_tracker.track(replay_network, t)

            u_out = np.array(replay_tracker["u_visible"])[-2 * len(u_target) :]
            r_out = np.array(replay_tracker["r_visible"])[-2 * len(u_target) :]

            mse_loss_r = compute_loss(r_out, r_target, mse)
            mse_loss_u = compute_loss(u_out, u_target, mse)

            mlflow.log_metric(f"replay_loss_r_pat_{en}", mse_loss_r, step=epoch)
            losses["replay_loss_r"].append(mse_loss_r)
            losses["replay_loss_u"].append(mse_loss_u)

        replay_tracker.store("losses", losses)


def main(full_config, pattern_path, rng):
    """Run a full training and replay experiment.

    Args:
        full_config: Dot-notation config namespace (from dict_to_namespace).
        pattern_path: Path to directory containing pattern .txt files.
        rng: NumPy random Generator for reproducibility.

    Returns:
        Tuple of (network, dataloader, train_tracker, validation_tracker,
                  replay_tracker, epoch_tracker).
    """
    experiment_params = full_config.experiment_params
    noise_params = full_config.noise_params
    neuron_params = full_config.neuron_params
    network_params = full_config.network_params
    simulation_params = full_config.simulation_params
    pattern_params = full_config.pattern_params
    weight_params = full_config.weight_params
    track_params = full_config.tracking_params

    patterns = load_patterns(
        experiment_params, pattern_params, network_params, pattern_path, rng
    )
    dataloader = build_dataloader(
        patterns, experiment_params, pattern_params, simulation_params, neuron_params
    )
    log_pattern_figure(dataloader, pattern_params)

    network_params.num_vis = dataloader.width
    network = setup_network(weight_params, network_params, neuron_params, rng)
    network.prepare_for_simulation(
        simulation_params.dt,
        SimpleUpdater(simulation_params.eta_out),
        SimpleUpdater(simulation_params.eta_lat),
    )

    dt = simulation_params.dt
    u_target = dataloader.get_full_pattern(dt, online_transforms=False)[
        :: track_params.sim_step
    ]
    r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)
    u_target_real = dataloader.get_full_pattern(dt, num=10, online_transforms=True)[
        :: track_params.sim_step
    ]
    r_target_real = eq_phi(u_target_real, neuron_params.a, neuron_params.b)
    targets = (u_target, r_target, r_target_real)

    train_tracker = Tracker(track_params.vars_train, track_params.sim_step)
    validation_tracker = Tracker(track_params.vars_val, track_params.sim_step)
    replay_trackers = [
        Tracker(track_params.vars_replay, track_params.sim_step) for _ in patterns
    ]
    epoch_tracker = Tracker(track_params.vars_epoch, 1)

    run_training(
        network,
        dataloader,
        simulation_params,
        neuron_params,
        noise_params,
        track_params,
        targets,
        (train_tracker, validation_tracker, epoch_tracker),
    )
    run_replay(
        network,
        dataloader,
        simulation_params,
        neuron_params,
        noise_params,
        experiment_params,
        track_params,
        targets,
        replay_trackers,
    )

    return (
        network,
        dataloader,
        train_tracker,
        validation_tracker,
        replay_trackers,
        epoch_tracker,
    )


if __name__ == "__main__":
    import tomllib as toml
    from pathlib import Path

    from utils import deep_merge, dict_to_namespace

    path = Path(__file__).parent.resolve()

    with open(path / "config.toml", "rb") as f:
        config_dict = toml.load(f)

    with open(path / "experiment.toml", "rb") as f:
        experiment_config = toml.load(f)

    full_config = deep_merge(config_dict, experiment_config)
    full_config = dict_to_namespace(
        full_config
    )  # was: dict_to_namespace(config_dict) — bug fix

    rng = np.random.default_rng(full_config.seed)

    main(
        full_config,
        pattern_path=path / "patterns",
        rng=rng,
    )
