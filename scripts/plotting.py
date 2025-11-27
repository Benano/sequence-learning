#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection

# def plot_weights_simple(weight_matrix):


def plot_activity_in_time(train_output, replay_output, dt):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Process train_output
    x_train = train_output[:, 0]
    y_train = train_output[:, 1]
    points_train = np.array([x_train, y_train]).T.reshape(-1, 1, 2)
    segments_train = np.concatenate([points_train[:-1], points_train[1:]], axis=1)

    # Process replay_output
    x_replay = replay_output[:, 0]
    y_replay = replay_output[:, 1]
    points_replay = np.array([x_replay, y_replay]).T.reshape(-1, 1, 2)
    segments_replay = np.concatenate([points_replay[:-1], points_replay[1:]], axis=1)

    # Create two separate LineCollections
    lc_train = LineCollection(
        segments_train, linewidths=2, color="blue", label="training"
    )
    lc_replay = LineCollection(
        [], linewidths=2, color="red", label="replay"
    )  # Start empty

    line_train = ax.add_collection(lc_train)
    line_replay = ax.add_collection(lc_replay)

    # Set plot limits
    border = 0.1
    ax.set_xlim(
        min(np.min(x_train), np.min(x_replay)) - border,
        max(np.max(x_train), np.max(x_replay) + border),
    )
    ax.set_ylim(
        min(np.min(y_train), np.min(y_replay)) - border,
        max(np.max(y_train), np.max(y_replay) + border),
    )

    train_frames = len(train_output)
    total_frames = train_frames + len(replay_output)

    def update(frame):
        frame = frame * 8  # Update every 5th frame
        if frame < train_frames:
            lc_train.set_segments(segments_train[:frame])
            lc_replay.set_segments([])  # Clear replay line
        else:
            lc_train.set_segments(segments_train)  # Complete train line
            replay_frame = frame - train_frames
            lc_replay.set_segments(segments_replay[:replay_frame])

        return line_train, line_replay

    ani = animation.FuncAnimation(
        fig, update, frames=total_frames // 8, blit=True, interval=dt
    )

    ax.legend()

    return ani


def plot_activity_target_match(
    train_output, replay_output, train_target, start_time, step
):
    # Create a figure with two subplots, sharing the x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 4), sharex=True)

    # Concatenate train and val output
    full_output = np.concatenate((train_output, replay_output), axis=1)

    # Determine overall min and max values
    vmin = min(np.min(full_output), np.min(train_target))
    vmax = max(np.max(full_output), np.max(train_target))

    colormap = plt.get_cmap("Blues")

    # Calculate x-ticks
    xticks = np.arange(start_time, start_time + full_output.shape[1] * step, 500) / 1000

    # Plot simulation output
    im1 = ax1.imshow(
        full_output,
        aspect="auto",
        interpolation="none",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_ylabel(r"Neuron")
    ax1.set_title(r"Output")
    ax1.axvline(train_target.shape[1], color="red", linestyle="--")

    # Plot target output
    ax2.imshow(
        train_target,
        aspect="auto",
        interpolation="none",
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_xlabel(r"$t[s]$")
    ax2.set_ylabel(r"Neuron")
    ax2.set_title(r"Target")
    ax2.axvline(train_target.shape[1], color="red", linestyle="--")

    # Add x-ticks based on start time and step
    ax2.set_xticks(np.arange(0, full_output.shape[1], 500 / step))
    ax2.set_xticklabels([f"{x: .1f}" for x in xticks])

    # Add a single colorbar for both plots
    cbar = fig.colorbar(im1, ax=[ax1, ax2])
    cbar.set_label("Firing rate")

    return fig


def plot_activity_match(replay_output, epoch_len, target):
    full_output = replay_output
    len_target = target.shape[1]
    full_target = np.tile(target, (1, (full_output.shape[1] // len_target) + 1))
    # match the length of the target to the output
    full_target = full_target[:, : full_output.shape[1]]

    thresh = 0.4

    # Output rates > threshold = 1, else 0 not using where
    full_output = (full_output > thresh).astype(int)
    full_target = (full_target > thresh).astype(int)

    # Create a new RGB image (white background)
    height, width = full_target.shape
    result = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Create masks for different conditions
    only_target = np.logical_and(full_target == 1, full_output == 0)
    only_output = np.logical_and(full_output == 1, full_target == 0)
    both = np.logical_and(full_output == 1, full_target == 1)

    # Set colors
    result[only_target] = [0, 255, 255]  # Cyan for only target
    result[only_output] = [255, 0, 0]  # Red for only output
    result[both] = [0, 0, 0]  # Black for both

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(result, aspect="auto", interpolation="none")
    # ax.axvline(len_target, color="red", linestyle="--")
    ax.set_ylabel("Output neuron")
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xlabel("Time (ms)")

    return fig


def plot_principal_components(u_latent, target):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = u_latent.T  # Assuming u_latent is your data for PCA

    # Standardize
    data_scaled = StandardScaler().fit_transform(data)

    # PCA
    pca = PCA(n_components=10)
    pca.fit_transform(data_scaled)

    ts_pca = pca.transform(data_scaled)
    n_steps = ts_pca.shape[0]
    indices = np.arange(n_steps)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    pattern = np.argmax(target, axis=1)
    # concatenate target once
    pattern = np.tile(pattern, (1, 2))

    # Use a colormap, e.g., 'viridis' or 'jet'
    # scatter = ax.scatter(ts_pca[:, 0], ts_pca[:, 1], ts_pca[:, 2],
    #                      c=pattern, cmap='Accent', marker='o', s=2)
    scatter = ax.scatter(
        ts_pca[:, 0], ts_pca[:, 1], ts_pca[:, 2], c=indices, cmap="jet", marker="o", s=2
    )

    # Add colorbar to show the mapping
    fig.colorbar(scatter, ax=ax, label="Index")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    return fig


def save_fig(fig, name, path, neptune_run, dpi=300):
    fig.savefig(path / name, dpi=dpi)
    if neptune_run:
        neptune_run[f"figures/{name}"].upload(fig)
    plt.close(fig)
    print(f"Saved figure {name} to {path}")


def load_pkl(path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_weights(epoch_tracker):
    # plot of the weights across time
    weights = epoch_tracker["dendritic_weights_all"]
    weights = np.array(weights)

    # flatten weights (time, neurons, neurons) to 2D
    n_time, n_neurons, _ = weights.shape
    weights_flat = weights.reshape(n_time, n_neurons * n_neurons)

    # Plot with 3 stacked subplots
    # 1st showing raw weights
    # 2nd showing summed positive and negative weights
    # 3rd showing amount of weight change compared to previous time step
    mean_weights = np.mean(weights_flat, axis=1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(weights_flat)
    axes[0].plot(mean_weights, color="black", linestyle="--", label="Mean Weights")
    axes[0].set_title("Dendritic Weights Over Time")
    axes[0].set_ylabel("Weight Value")
    axes[0].set_xlabel("Time Step")
    # 2nd subplot
    sum_positive = np.sum(weights_flat * (weights_flat > 0), axis=1)
    sum_negative = np.sum(weights_flat * (weights_flat < 0), axis=1)
    sum_all = np.sum(weights_flat, axis=1)
    axes[1].plot(sum_positive, label="Sum Positive Weights", color="red")
    axes[1].plot(sum_negative, label="Sum Negative Weights", color="blue")
    axes[1].plot(sum_all, label="Mean Weights", color="black", linestyle="--")
    axes[1].set_title("Sum of Positive and Negative Weights Over Time")
    axes[1].set_ylabel("Sum of Weights")
    axes[1].legend()
    # 3rd subplot
    weight_change = np.linalg.norm(np.diff(weights_flat, axis=0), axis=1)
    axes[2].plot(weight_change, color="green")
    axes[2].set_title("Weight Change Over Time")
    axes[2].set_ylabel("Weight Change (L2 Norm)")
    axes[2].set_xlabel("Time Step")
    plt.tight_layout()

    return fig


def main(full_config, run_path, artifact_path, figure_path, neptune_run):
    from elise.config import FullConfig

    train = load_pkl(artifact_path / "train_dict.pkl")
    val = load_pkl(artifact_path / "validation_dict.pkl")
    replay = load_pkl(artifact_path / "replay_dict.pkl")
    epoch = load_pkl(artifact_path / "epoch_dict.pkl")

    full_config = FullConfig(run_path / "config.toml")
    sim_params = full_config.simulation_params
    pattern_params = full_config.pattern_params
    track_params = full_config.tracking_params

    pattern_duration = pattern_params.pattern_duration * pattern_params.pattern_dt
    dt = sim_params.dt
    sim_step = track_params.sim_step

    last_train = int(2 * pattern_duration / dt / sim_step)
    train_output = train["u_visible"][-last_train:]
    replay_output = replay["u_visible"][: last_train * 4]

    last_train = int(pattern_duration / dt / sim_step)
    first_replay = 1 * int(pattern_duration / dt / sim_step)

    train_output = train["u_visible"][-last_train:].T
    replay_output = replay["u_visible"][: first_replay * 5].T
    train_target = train["u_inp_visible"][-last_train:].T
    dpi = 300
    start_time = (
        pattern_params.pattern_duration
        * (sim_params.training_cycles - 1)
        * sim_params.training_epochs
    )
    step = sim_params.dt * track_params.sim_step

    fig = plot_activity_target_match(
        train_output, replay_output, train_target, start_time, step
    )
    save_fig(fig, "activity_match.png", figure_path, neptune_run, dpi)

    first_replay = 2 * int(pattern_duration / dt / sim_step)
    train_output = train["r_visible"][-last_train:].T
    replay_output = replay["r_visible"][:first_replay].T
    train_target = val["r_target"][0].T
    # hidden_activity = train["r_latent"][-last_train:].T

    # PCA
    # latent_activity = replay["r_latent"][-2 * last_train :].T
    # fig = plot_principal_components(latent_activity, target=train_target.T)
    # save_fig(fig, "PCA.png", figure_path, neptune_run)

    replay_output = replay["r_visible"][:first_replay].T
    epoch_len = int(pattern_duration / dt / sim_step)
    fig = plot_activity_match(replay_output, epoch_len, train_target)
    save_fig(fig, "activity_match_replay.png", figure_path, neptune_run, dpi)

    fig = plot_weights(epoch)
    save_fig(fig, "weights_over_time.png", figure_path, neptune_run, dpi)

    dpi = 100
    gif = False
    if gif:
        ani = plot_activity_in_time(train_output, replay_output, dt)
        writer = PillowWriter(fps=30)
        ani.save("figs/activity.gif", writer)


if __name__ == "__main__":
    from pathlib import Path

    import neptune

    from elise.config import FullConfig

    path = Path(__file__).parent.resolve()
    artifact_path = path / "artifacts"
    figure_path = path / "figures"
    config_path = path / "config.toml"
    full_config = FullConfig(config_path)

    with open("run_id.txt", "r") as f:
        run_id = f.read().strip()

    neptune_run = neptune.init_run(
        project="elise-neurotma/Elise-tests",
        custom_run_id=run_id,
        name=path.name,
        tags=full_config.experiment_params.patterns,
    )

    main(
        full_config,
        path,
        artifact_path,
        figure_path,
        neptune_run,
    )
