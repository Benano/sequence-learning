#!/usr/bin/env python3
import networkx as nx
import numpy as np


def analyze_connectivity_metrics(
    weight_matrix: np.ndarray, num_vis: int, cycles=False
) -> dict:
    """
    Compute various metrics of the connectivity matrix.

    :param weight_matrix: 2D numpy array, shape (num_total, num_total),
                          where weight_matrix[post, pre] = 1 means a connection pre->post.
    :param num_vis: Number of visible neurons, to separate visible and lateral neurons.
    :return: Dictionary with computed metrics.
    """
    num_total = weight_matrix.shape[0]

    # Total connections
    total_connections = np.sum(weight_matrix)

    # Connection density (assuming visible -> lateral connections possible)
    num_lat = num_total - num_vis
    connection_density = total_connections / (num_vis * num_lat)

    # Out-degree: sum across rows for each presynaptic neuron
    out_degrees = np.sum(weight_matrix, axis=0)
    avg_out_degree = np.mean(out_degrees)

    # In-degree for lateral neurons only
    in_degrees = np.sum(weight_matrix[num_vis:, :], axis=1)
    avg_in_degree = np.mean(in_degrees)

    max_in_degree = np.max(in_degrees)

    # Fraction isolated (degree 0) lateral neurons
    frac_isolated = np.sum(in_degrees == 0) / num_lat

    # Sparsity ratio
    sparsity = 1 - (total_connections / (num_total * num_total))

    # Largest singular value
    largest_singular_value = np.linalg.svd(weight_matrix, compute_uv=False)[0]

    # Spectral radius (max absolute eigenvalue)
    spectral_radius = np.max(np.abs(np.linalg.eigvals(weight_matrix)))

    # Build directed graph with NetworkX
    G = nx.from_numpy_array(weight_matrix, create_using=nx.DiGraph)

    # Check if acyclic
    is_acyclic = nx.is_directed_acyclic_graph(G)

    # Count number of cycles by enumerating simple cycles (limited to avoid long runs)
    if cycles:
        w_cycles = list(nx.simple_cycles(G))
        num_cycles = len(w_cycles)
    else:
        num_cycles = None

    metrics = {
        "total_connections": total_connections,
        "connection_density": connection_density,
        "avg_out_degree": avg_out_degree,
        "avg_in_degree": avg_in_degree,
        "max_in_degree": max_in_degree,
        "frac_isolated_lateral": frac_isolated,
        "sparsity": sparsity,
        "largest_singular_value": largest_singular_value,
        "spectral_radius": spectral_radius,
        "is_acyclic": is_acyclic,
        "num_cycles": num_cycles,
    }

    return metrics


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    from elise.config import FullConfig
    from elise.weights import DendriticWeights, RandomSomaticWeights, SomaticWeights

    path = Path(__file__).parent.resolve()
    config_path = path / "config.toml"
    full_config = FullConfig(config_path)

    weight_params = full_config.weight_params
    rng = np.random.default_rng(seed=43)

    n_vis = 10
    n_lat = 50

    dendritic_weights = DendriticWeights(weight_params, rng_w=rng, rng_d=rng)

    somatic_weight_types = {
        "developed": SomaticWeights,
        "random": RandomSomaticWeights,
    }

    weight_type = somatic_weight_types[weight_params.weight_type]
    somatic_weights = weight_type(weight_params, rng_w=rng, rng_d=rng)

    dendritic_weights(num_vis=n_vis, num_lat=n_lat)
    somatic_weights(num_vis=n_vis, num_lat=n_lat)

    num_vis = 2
    metrics = analyze_connectivity_metrics(
        somatic_weights.weight_matrix, num_vis, cycles=True
    )
    print("Somatic Weight Matrix Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # All to all with values beween -1 and 1
    metrics = analyze_connectivity_metrics(dendritic_weights.weight_matrix, num_vis)
    print("\nDendritic Weight Matrix Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # plot heatmaps
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Somatic Weights")
    plt.imshow(somatic_weights.weight_matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.subplot(1, 2, 2)

    plt.title("Dendritic Weights")
    plt.imshow(dendritic_weights.weight_matrix, cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
