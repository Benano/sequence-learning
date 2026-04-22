#!/usr/bin/env python3
import matplotlib.pyplot as plt
import mlflow
import networkx as nx
import numpy as np
from matplotlib import animation
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.offsetbox import AnchoredText

from elise.model import Network, eq_phi  # noqa
from elise.weight_metrics import analyze_connectivity_metrics

_mlflow_enabled = True


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


def plot_activity_target_match(replay_dicts, epoch_len):
    n_patterns = len(replay_dicts)
    colormap = plt.get_cmap("Blues")
    col_labels = ["Target", "Replay", "Replay (Epoch 20)"]

    fig, axes = plt.subplots(
        n_patterns,
        3,
        figsize=(12, 3 * n_patterns),
        squeeze=False,
        sharey="row",
        sharex="col",
        width_ratios=(0.5, 1, 1),
    )

    for i, replay in enumerate(replay_dicts):
        target = replay["r_target"][0].T  # (neurons, time)
        replay_output = replay["r_visible"][: epoch_len * 2].T  # (neurons, 2 epochs)
        replay_epoch20 = replay["r_visible"][epoch_len * 20 : epoch_len * 22].T

        vmin = min(np.min(target), np.min(replay_output), np.min(replay_epoch20))
        vmax = max(np.max(target), np.max(replay_output), np.max(replay_epoch20))

        for j, data in enumerate([target, replay_output, replay_epoch20]):
            axes[i, j].imshow(
                data,
                aspect="auto",
                interpolation="none",
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_labels[j], fontsize=11)
            if i == n_patterns - 1:
                axes[i, j].set_xlabel("Time (steps)")

        axes[i, 0].set_ylabel("Output Neuron")

    plt.tight_layout()
    return fig


def plot_pattern_separation_pca(replay_dicts, epoch_len):
    from sklearn.decomposition import PCA

    # One epoch of latent activity per pattern: shape (time, neurons)
    activities = [r["r_latent"][:epoch_len] for r in replay_dicts]
    n_patterns = len(activities)
    colors = plt.get_cmap("tab10").colors

    combined = np.concatenate(activities, axis=0)
    pca = PCA(n_components=3)
    pca.fit(combined)
    projections = [pca.transform(a) for a in activities]

    fig = plt.figure(figsize=(12, 5))

    ax3d = fig.add_subplot(121, projection="3d")
    for i, proj in enumerate(projections):
        ax3d.plot(
            proj[:, 0],
            proj[:, 1],
            proj[:, 2],
            color=colors[i],
            linewidth=1.2,
            alpha=0.8,
            label=f"Pattern {i}",
        )
        ax3d.scatter(*proj[0, :3], color=colors[i], s=40, zorder=5)  # mark start
    ax3d.set_xlabel("PC1")
    ax3d.set_ylabel("PC2")
    ax3d.set_zlabel("PC3")
    ax3d.set_title("Replay trajectories (latent PC space)")
    ax3d.legend()

    ax2d = fig.add_subplot(122)
    for i in range(n_patterns):
        for j in range(i + 1, n_patterns):
            a, b = projections[i], projections[j]
            sim = np.sum(a * b, axis=1) / (
                np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
            )
            label = f"{i} vs {j}" if n_patterns > 2 else None
            ax2d.plot(sim, color="steelblue", label=label)
    ax2d.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2d.set_xlabel("Time (steps)")
    ax2d.set_ylabel("Cosine similarity")
    ax2d.set_title("Pattern similarity over time (PC space)")
    if n_patterns > 2:
        ax2d.legend()

    var_explained = pca.explained_variance_ratio_
    fig.text(
        0.5,
        0.01,
        f"PC1–3 explain {var_explained[:3].sum()*100:.1f}% variance "
        f"({', '.join(f'{v*100:.1f}%' for v in var_explained[:3])})",
        ha="center",
        fontsize=9,
        color="gray",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def plot_neuron_selectivity(replay_dicts, epoch_len):
    mean_rates = np.stack(
        [r["r_latent"][:epoch_len].mean(axis=0) for r in replay_dicts]
    )  # (n_patterns, neurons)

    if len(replay_dicts) == 2:
        rate_0, rate_1 = mean_rates[0], mean_rates[1]
        selectivity = (rate_0 - rate_1) / (rate_0 + rate_1 + 1e-8)
        sort_idx = np.argsort(selectivity)[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].bar(
            np.arange(len(selectivity)),
            selectivity[sort_idx],
            color=["steelblue" if s >= 0 else "tomato" for s in selectivity[sort_idx]],
            width=1.0,
            edgecolor="none",
        )
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xlabel("Neuron (sorted)")
        axes[0].set_ylabel("Selectivity index\n(+1 = Pattern 0, −1 = Pattern 1)")
        axes[0].set_title("Per-neuron selectivity")

        im = axes[1].imshow(
            mean_rates[:, sort_idx],
            aspect="auto",
            interpolation="none",
            cmap="Blues",
        )
        axes[1].set_yticks([0, 1])
        axes[1].set_yticklabels(["Pattern 0", "Pattern 1"])
        axes[1].set_xlabel("Neuron (sorted by selectivity)")
        axes[1].set_title("Mean firing rate during replay")
        fig.colorbar(im, ax=axes[1], label="Mean rate")
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(mean_rates, aspect="auto", interpolation="none", cmap="Blues")
        ax.set_yticks(np.arange(len(replay_dicts)))
        ax.set_yticklabels([f"Pattern {i}" for i in range(len(replay_dicts))])
        ax.set_xlabel("Neuron")
        ax.set_title("Mean firing rate during replay")
        fig.colorbar(im, ax=ax, label="Mean rate")

    plt.tight_layout()
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
    if _mlflow_enabled:
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
    replay_dicts = []
    i = 0
    while (artifact_path / f"replay_dict_{i}.pkl").exists():
        replay_dicts.append(load_pkl(artifact_path / f"replay_dict_{i}.pkl"))
        i += 1
    if not replay_dicts:
        replay_dicts = [load_pkl(artifact_path / "replay_dict.pkl")]
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

    dpi = 300
    epoch_len = int(pattern_params.pattern_duration / dt / sim_step)

    fig = plot_weights_grid(network)
    save_fig(fig, "weights_grid.png", figure_path, dpi)

    fig = plot_weights_in_time(epoch, val, full_config)
    save_fig(fig, "weights_over_time.png", figure_path, dpi)

    fig = plot_activity_target_match(replay_dicts, epoch_len)
    save_fig(fig, "activity_target_match.png", figure_path, dpi)

    fig = plot_pattern_separation_pca(replay_dicts, epoch_len)
    save_fig(fig, "pattern_separation_pca.png", figure_path, dpi)

    fig = plot_neuron_selectivity(replay_dicts, epoch_len)
    save_fig(fig, "neuron_selectivity.png", figure_path, dpi)

    fig = plot_connectivity_network(network.somatic_weights, num_vis=network.num_vis)
    save_fig(fig, "somatic_connectivity.png", figure_path, dpi)


if __name__ == "__main__":
    import tomllib as toml
    from pathlib import Path

    from utils import dict_to_namespace

    _mlflow_enabled = False

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
