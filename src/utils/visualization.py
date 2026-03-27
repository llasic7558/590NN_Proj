"""
Shared visualization utilities for all experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_layer_accuracy_curves(results: dict, save_path: Path = None):
    """
    Plot probing accuracy across layers for each phenomenon.

    Args:
        results: {phenomenon: [ProbeResult per layer]}
        save_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for phenom, layer_results in results.items():
        layers = [r.layer for r in layer_results]
        accs = [r.accuracy for r in layer_results]
        stds = [r.accuracy_std for r in layer_results]

        ax.errorbar(layers, accs, yerr=stds, marker="o", label=phenom, capsize=3)

    ax.set_xlabel("BERT Layer")
    ax.set_ylabel("Probing Accuracy")
    ax.set_title("Layer-wise Probing Accuracy by Phenomenon")
    ax.set_xticks(range(13))
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_probing_heatmap(results: dict, save_path: Path = None):
    """
    Heatmap of probing accuracy: rows=phenomena, columns=layers.
    """
    phenomena = sorted(results.keys())
    n_layers = len(results[phenomena[0]])
    matrix = np.zeros((len(phenomena), n_layers))

    for i, phenom in enumerate(phenomena):
        for j, r in enumerate(results[phenom]):
            matrix[i, j] = r.accuracy

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=range(n_layers),
        yticklabels=phenomena,
        ax=ax, vmin=0.5, vmax=1.0,
    )
    ax.set_xlabel("BERT Layer")
    ax.set_title("Probing Accuracy (Layer × Phenomenon)")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_cka_heatmap(cka_matrix: np.ndarray, save_path: Path = None):
    """
    Heatmap of CKA similarity between adjacent layers.

    Args:
        cka_matrix: (n_layers, n_layers) CKA similarity matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cka_matrix, annot=True, fmt=".2f", cmap="coolwarm",
        xticklabels=range(cka_matrix.shape[1]),
        yticklabels=range(cka_matrix.shape[0]),
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ax.set_title("CKA Similarity Between Layers")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_ablation_impact(ablation_results: dict, save_path: Path = None):
    """
    Bar chart showing probing accuracy drop when each head is ablated.

    Args:
        ablation_results: {(layer, head): accuracy_drop}
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    keys = sorted(ablation_results.keys())
    labels = [f"L{l}H{h}" for l, h in keys]
    drops = [ablation_results[k] for k in keys]

    colors = ["red" if d > 0.02 else "steelblue" for d in drops]
    ax.bar(range(len(labels)), drops, color=colors, alpha=0.7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_ylabel("Accuracy Drop")
    ax.set_title("Attention Head Ablation Impact")
    ax.axhline(y=0.02, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax.legend()

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_token_position_grid(grid: np.ndarray, token_labels: list, save_path: Path = None):
    """
    Heatmap of probing accuracy at layer × token position.

    Args:
        grid: (n_layers, n_positions) accuracy values.
        token_labels: Labels for token positions (e.g., ["subject", "verb", ...]).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        grid, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=token_labels,
        yticklabels=range(grid.shape[0]),
        ax=ax, vmin=0.5, vmax=1.0,
    )
    ax.set_xlabel("Token Position Role")
    ax.set_ylabel("BERT Layer")
    ax.set_title("Probing Accuracy (Layer × Token Position)")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_pca_umap(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    layer: int,
    method: str = "pca",
    save_path: Path = None,
):
    """
    2D PCA or UMAP scatter of hidden states at a given layer.

    Args:
        hidden_states: (n_samples, hidden_dim)
        labels: (n_samples,) binary labels
        layer: layer index for title
        method: "pca" or "umap"
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(hidden_states)
        title_method = "PCA"
    elif method == "umap":
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42)
        coords = reducer.fit_transform(hidden_states)
        title_method = "UMAP"
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="coolwarm", alpha=0.6, s=15,
    )
    ax.set_title(f"{title_method} — Layer {layer}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.colorbar(scatter, ax=ax, label="Grammatical")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()
