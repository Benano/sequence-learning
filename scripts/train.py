import ast

import neptune
import numpy as np
from tqdm import tqdm

from elise.data import Dataloader, MultiHotPattern, Pattern
from elise.model import Network, eq_phi  # noqa
from elise.optimizer import SimpleUpdater
from elise.rate_buffer import Buffer
from elise.stats import compute_loss, mse
from elise.tracker import Tracker
from elise.weights import DendriticWeights, SomaticWeights


def load_multi_hot_pattern(filename):
    with open(filename, "r") as file:
        content = file.read().strip()

    pat = ast.literal_eval(f"[{content}]")
    return pat


def load_one_hot_pattern(pattern_file):
    pat = np.loadtxt(pattern_file, delimiter=" ").astype(int).T
    pat = pat[:, pat.any(axis=0)]

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


class WhiteNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x):
        noise = np.random.normal(0, self.sigma, x.shape[0])
        return x + noise


def main(full_config, run_path, artifact_path, pattern_path, neptune_run, rng):
    experiment_params = full_config.experiment_params
    neuron_params = full_config.neuron_params
    network_params = full_config.network_params
    simulation_params = full_config.simulation_params
    weight_params = full_config.weight_params
    track_params = full_config.tracking_params

    if len(experiment_params.patterns) > 1:
        patterns = []
        pattern_types = []
        for pattern_name in experiment_params.patterns:
            pattern_file = pattern_path / f"{pattern_name}.txt"
            pattern, pattern_type = load_pattern_flexible(pattern_file)

            patterns.append(pattern)
            pattern_types.append(pattern_type)

        # Check that patterns are all the same type
        if len(set(pattern_types)) > 1:
            raise ValueError("All patterns must be of the same type for stacking.")
        min_length = min([len(p) for p in patterns])
        patterns = [p[:min_length] for p in patterns]
        full_pattern = np.hstack(patterns)

    else:
        pattern_file = pattern_path / f"{experiment_params.patterns[0]}.txt"
        full_pattern, pattern_type = load_pattern_flexible(pattern_file)

    if len(full_pattern) > 300:
        full_pattern = full_pattern[:250, :]

    if pattern_type == "multi-hot":
        pattern = MultiHotPattern(
            pattern=full_pattern,
            duration=simulation_params.pattern_duration,
            width=network_params.num_vis,
        )

    elif pattern_type == "one-hot":
        pattern = Pattern(
            pattern=full_pattern, duration=simulation_params.pattern_duration
        )

    def to_biounits(x):
        return neuron_params.E_l + x * 20.0

    def add_white_noise(x):
        noise = np.random.normal(0, simulation_params.noise_sigma, x.shape[0])
        return x + noise

    online_transforms = []
    if simulation_params.noise_sigma > 0:
        online_transforms.append(WhiteNoise(simulation_params.noise_sigma))

    dataloader = Dataloader(
        pattern, pre_transforms=[to_biounits], online_transforms=online_transforms
    )

    # Set seed to experiment_params.seed
    np.random.seed(experiment_params.seed)

    # get for random numbers for the seeds of the weights

    # Network
    rate_buffer = Buffer
    dendritic_weights = DendriticWeights(weight_params, rng)
    somatic_weights = SomaticWeights(weight_params, rng)
    network_params.num_vis = pattern.shape[1]
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
    u_target = dataloader.get_full_pattern(dt)[:: track_params.sim_step]
    r_target = eq_phi(u_target, neuron_params.a, neuron_params.b)

    # Sim params
    training_duration = (
        simulation_params.training_cycles * simulation_params.pattern_duration
    )
    validation_duration = (
        simulation_params.validation_cycles * simulation_params.pattern_duration
    )

    # Sim Trackers
    train_tracker = Tracker(track_params.vars_train, track_params.sim_step)
    validation_tracker = Tracker(track_params.vars_val, track_params.sim_step)

    validation_loss_u = []
    validation_loss_r = []

    c_t = 0.0
    for epoch in tqdm(range(simulation_params.training_epochs)):
        for t in np.arange(0, training_duration, simulation_params.dt):
            network(u_inp=dataloader(t))

            # Only track the last two epochs
            c_t = c_t + simulation_params.dt
            if epoch >= simulation_params.training_epochs - 2:
                train_tracker.track(network, c_t)

        # Add learning rate decay
        network.optimizer_vis.eta *= simulation_params.eta_decay
        network.optimizer_lat.eta *= simulation_params.eta_decay

        # Validation
        if epoch != simulation_params.training_epochs - 1:
            for t in np.arange(0, validation_duration, simulation_params.dt):
                c_t = c_t + simulation_params.dt
                network(u_inp=None)
                validation_tracker.track(network, c_t)

            u_out = np.array(validation_tracker["u_visible"])[-2 * len(u_target) :]
            r_out = np.array(validation_tracker["r_visible"])[-2 * len(u_target) :]

            mse_loss_u = compute_loss(u_out, u_target, mse)
            mse_loss_r = compute_loss(r_out, r_target, mse)

            if neptune_run:
                neptune_run["validation_loss_r"].append(mse_loss_r)

            validation_loss_r.append(mse_loss_r)
            validation_loss_u.append(mse_loss_u)

            # print(f"Epoch {epoch} -  MSE u: {mse_loss_u}")  # noqa
            # print(f"Epoch {epoch} -  MSE r: {mse_loss_r}")  # noqa

    train_tracker.store("r_target", r_target)
    validation_tracker.store("r_target", r_target)
    validation_tracker.store("mse_loss_r", validation_loss_r)
    validation_tracker.store("mse_loss_u", validation_loss_u)

    return network, dataloader, train_tracker, validation_tracker


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
