#!/usr/bin/env python3
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnchoredText
from weight_metrics import analyze_connectivity_metrics

from elise.model import Network, eq_phi  # noqa


def plot_connectivity_network(weight_matrix, num_vis: int = None):
    """Plot connectivity matrix as a network graph."""

    # Create directed graph from matrix (weight_matrix[post, pre] = 1 means pre -> post)
    G = nx.from_numpy_array(weight_matrix, create_using=nx.DiGraph())

    # Plot with labels, colors, and arrows
    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw(
        G,
        pos=nx.spring_layout(G),
        with_labels=True,
        node_color="lightblue",
        node_size=20,
        arrows=False,
        font_size=8,
        arrowsize=20,
        edge_color="gray",
        width=1.5,
    )
    plt.title("Connectivity Network")

    metrics = analyze_connectivity_metrics(weight_matrix, num_vis)

    # ADD METRICS BOX IN LOWER RIGHT
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.3f}")
        elif isinstance(v, bool):
            lines.append(f"{k}: {'Yes' if v else 'No'}")
        else:
            lines.append(f"{k}: {v}")
    text = "\n".join(lines)

    at = AnchoredText(
        text,
        loc="lower right",
        prop=dict(size=9),
        frameon=True,
        borderpad=0.5,
        bbox_to_anchor=(1.0, 0.0),  # Align to lower right
        bbox_transform=ax.transAxes,
    )
    ax.add_artist(at)

    return fig


# Usage


def plot_weights_grid(network):
    dendritic_weights = network.dendritic_weights
    somatic_weights = network.somatic_weights

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im1 = axes[0].imshow(
        dendritic_weights,
        aspect="auto",
        interpolation="none",
        cmap="bwr",
        vmin=-np.max(np.abs(dendritic_weights)),
        vmax=np.max(np.abs(dendritic_weights)),
    )
    axes[0].set_title("Dendritic Weights")
    fig.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(
        somatic_weights,
        aspect="auto",
        interpolation="none",
        cmap="bwr",
        vmin=-np.max(np.abs(somatic_weights)),
        vmax=np.max(np.abs(somatic_weights)),
    )
    axes[1].set_title("Somatic Weights")
    fig.colorbar(im2, ax=axes[1])
    plt.tight_layout()

    return fig


def plot_dendritic_weights_animation(epoch_tracker):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    ax[0].imshow(
        epoch_tracker["dendritic_weights_all"][0, :, :],
        aspect="auto",
        interpolation="none",
        cmap="bwr",
        vmin=-np.max(np.abs(epoch_tracker["dendritic_weights_all"])),
        vmax=np.max(np.abs(epoch_tracker["dendritic_weights_all"])),
    )

    weight_changes = np.diff(epoch_tracker["dendritic_weights_all"], axis=0)
    # stack to have same number of frames
    weight_changes = np.vstack((weight_changes[0:1, :, :], weight_changes))
    ax[1].imshow(
        weight_changes[0, :, :],
        aspect="auto",
        interpolation="none",
        cmap="bwr",
        vmin=-np.max(np.abs(weight_changes)),
        vmax=np.max(np.abs(weight_changes)),
    )

    def update(frame):
        ax[0].clear()
        ax[0].imshow(
            epoch_tracker["dendritic_weights_all"][frame, :, :],
            aspect="auto",
            interpolation="none",
            cmap="bwr",
            vmin=-np.max(np.abs(epoch_tracker["dendritic_weights_all"])),
            vmax=np.max(np.abs(epoch_tracker["dendritic_weights_all"])),
        )
        ax[1].clear()
        ax[1].imshow(
            weight_changes[frame, :, :],
            aspect="auto",
            interpolation="none",
            cmap="bwr",
            vmin=-np.max(np.abs(weight_changes)),
            vmax=np.max(np.abs(weight_changes)),
        )

        return ax

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(epoch_tracker["dendritic_weights_all"]),
        blit=True,
        interval=500,
    )

    plt.show()

    return ani


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


def plot_activity_lines(train_target, train_output, replay_output, dt):
    target_len = train_target.shape[1]
    last_train = train_output[:, -target_len:]
    first_three_replay = replay_output[:, : target_len * 3]
    time_axis_train = np.arange(last_train.shape[1]) * dt
    time_axis_replay = np.arange(first_three_replay.shape[1]) * dt
    time_axis_full = np.arange(last_train.shape[1] + first_three_replay.shape[1]) * dt

    full_target = np.concatenate(
        (train_target, train_target, train_target, train_target), axis=1
    )

    fig, ax = plt.subplots(train_target.shape[0], 1, figsize=(12, 8), sharex=True)
    for neuron_idx in range(train_target.shape[0]):
        # Plot target for full duration
        ax[neuron_idx].plot(
            time_axis_full,
            full_target[
                neuron_idx, : last_train.shape[1] + first_three_replay.shape[1]
            ],
            color="gray",
            linestyle="--",
            label="Target" if neuron_idx == 0 else "",
        )

        # Plot last training output
        ax[neuron_idx].plot(
            time_axis_train,
            last_train[neuron_idx, :],
            color="blue",
            label="Training Output" if neuron_idx == 0 else "",
        )
        # Plot first three replay outputs
        ax[neuron_idx].plot(
            time_axis_replay + time_axis_train[-1] + dt,
            first_three_replay[neuron_idx, :],
            color="red",
            label="Replay Output" if neuron_idx == 0 else "",
        )
        ax[neuron_idx].set_ylabel(f"Neuron {neuron_idx}")
        if neuron_idx == 0:
            ax[neuron_idx].legend()

    plt.xlabel("Time (ms)")
    plt.show()


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


def save_fig(fig, name, path, dpi=300):
    fig.savefig(path / name, dpi=dpi)
    mlflow.log_figure(fig, f"figures/{name}")
    plt.close(fig)
    print(f"Saved figure {name} to {path}")


def load_pkl(path):
    import pickle

    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def plot_weights_in_time(epoch_tracker, validation_tracker, config):
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

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
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

    nr_timesteps = len(mean_weights)
    losses = validation_tracker["losses"][0]["validation_loss_r"]
    axes[3].set_title("Validation Loss Over Time")
    axes[3].set_ylabel("Loss")
    axes[3].set_xlabel("Time Step")
    axes[3].set_xlim(0, nr_timesteps)
    axes[3].plot(losses)

    plt.tight_layout()

    return fig


def plot_errors(train_error, replay_error):
    begin_train_error = train_error[:1000, :10]
    train_error = train_error[-1000:, :10]
    replay_error = replay_error[:1000, :10]
    replay_end = replay_error[-1000:, :10]

    concat_errors = np.concatenate(
        (begin_train_error, train_error, replay_error, replay_end), axis=0
    )

    # concatenate all errors
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(concat_errors)
    ax.set_title("Training and Replay Errors Over Time")
    ax.axvline(
        len(train_error) + len(begin_train_error),
        color="red",
        linestyle="--",
        label="Start of Replay",
    )
    ax.axvline(
        len(begin_train_error),
        color="green",
        linestyle="--",
        label="End of Initial Training",
    )
    ax.axvline(
        len(begin_train_error) + len(train_error) + len(replay_error),
        color="orange",
        linestyle="--",
        label="End of Replay",
    )
    ax.legend()
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Error")
    plt.show()


def main(full_config, run_path, artifact_path, figure_path):
    import tomllib as toml
    from pathlib import Path

    from utils import dict_to_namespace

    train = load_pkl(artifact_path / "train_dict.pkl")
    val = load_pkl(artifact_path / "validation_dict.pkl")
    replay = load_pkl(artifact_path / "replay_dict.pkl")
    epoch = load_pkl(artifact_path / "epoch_dict.pkl")
    network = load_pkl(artifact_path / "network.pkl")

    path = Path(__file__).parent.resolve()

    # 1. Load the merged config (this is the one created by the Runner script)
    with open(path / "config.toml", "rb") as f:
        config_dict = toml.load(f)
    full_config = dict_to_namespace(config_dict)

    sim_params = full_config.simulation_params
    pattern_params = full_config.pattern_params
    track_params = full_config.tracking_params

    pattern_duration = pattern_params.pattern_duration * pattern_params.pattern_dt
    dt = sim_params.dt
    sim_step = track_params.sim_step

    last_train = int(2 * pattern_duration / dt / sim_step)
    train_output = train["r_visible"][-last_train:]
    replay_output = replay["r_visible"][: last_train * 4]

    last_train = int(pattern_duration / dt / sim_step)
    first_replay = 1 * int(pattern_duration / dt / sim_step)

    train_output = train["r_visible"][-last_train:].T
    replay_output = replay["r_visible"][: first_replay * 5].T
    train_target_u = train["u_inp_visible"][-last_train:].T
    train_target = eq_phi(train_target_u, a=0.3, b=-58.0)

    dpi = 300
    start_time = (
        pattern_params.pattern_duration
        * (sim_params.training_cycles - 1)
        * sim_params.training_epochs
    )
    step = sim_params.dt * track_params.sim_step

    fig = plot_weights_grid(network)
    save_fig(fig, "weights_grid.png", figure_path, dpi)

    fig = plot_activity_target_match(
        train_output, replay_output, train_target, start_time, step
    )
    save_fig(fig, "activity_match.png", figure_path, dpi)

    fig = plot_weights_in_time(epoch, val, full_config)
    save_fig(fig, "weights_over_time.png", figure_path, dpi)

    first_replay = 2 * int(pattern_duration / dt / sim_step)
    train_output = train["r_visible"][-last_train:].T
    replay_output = replay["r_visible"][:first_replay].T
    train_target = val["r_target"][0].T
    # hidden_activity = train["r_latent"][-last_train:].T

    # PCA
    # latent_activity = replay["r_latent"][-2 * last_train :].T
    # fig = plot_principal_components(latent_activity, target=train_target.T)
    # save_fig(fig, "PCA.png", figure_path)

    replay_output = replay["r_visible"][:first_replay].T
    epoch_len = int(pattern_duration / dt / sim_step)
    fig = plot_activity_match(replay_output, epoch_len, train_target)
    save_fig(fig, "activity_match_replay.png", figure_path, dpi)

    fig = plot_connectivity_network(network.somatic_weights, num_vis=network.num_vis)
    save_fig(fig, "somatic_connectivity", figure_path, dpi)

    dpi = 100
    gif = False
    if gif:
        ani = plot_activity_in_time(train_output, replay_output, dt)
        writer = PillowWriter(fps=30)
        ani.save("figs/activity.gif", writer)


if __name__ == "__main__":
    import tomllib as toml
    from pathlib import Path

    from utils import dict_to_namespace

    with open("config.toml", "rb") as f:
        config_dict = toml.load(f)

    full_config = dict_to_namespace(config_dict)

    path = Path(__file__).parent.resolve()
    artifact_path = path / "artifacts"
    figure_path = path / "figures"

    main(
        full_config,
        path,
        artifact_path,
        figure_path,
    )
