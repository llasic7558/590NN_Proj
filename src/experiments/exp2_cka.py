"""
Experiment 2: Adjacent-Layer CKA (Preserve-or-Rewrite Analysis)

Measures representational similarity between adjacent BERT layers using
Linear CKA. Answers whether each layer is actively transforming
representations or simply passing them along.

Key question: A layer can *contain* grammatical information without
having *produced* it. CKA tells us which layers are doing real work.

Steps:
  1. For each sentence, collect [CLS] representations at all 13 layers
  2. Compute CKA(layer_i, layer_i+1) for i in 0..11
  3. Break down by phenomenon — do "rewriting events" (low CKA between
     adjacent layers) coincide with peaks in probing accuracy?
  4. Generate CKA heatmap and adjacent-layer CKA curve

Expected: Sharp CKA drops at middle layers for agreement; later drops
for negation/clause phenomena.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

from src.utils.cka import linear_cka, compute_cka_matrix
from src.utils.visualization import plot_cka_heatmap

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp2_cka"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "exp2"
NUM_LAYERS = 13


def collect_layer_representations(
    hidden_states_dir: Path,
    metadata: list[dict],
    pooling: str = "cls",
) -> list[np.ndarray]:
    """
    Collect representations at each layer across all sentences.

    For each minimal pair, uses the "good" sentence representation
    so we're measuring how BERT represents grammatical language.

    Returns:
        List of 13 arrays, each (n_sentences, 768).
    """
    # Pre-allocate lists for each layer
    layer_lists = [[] for _ in range(NUM_LAYERS)]

    for entry in tqdm(metadata, desc="  Collecting representations"):
        npz = np.load(hidden_states_dir / f"pair_{entry['idx']:04d}.npz")
        good_hidden = npz["good_hidden"]  # (13, seq_len, 768)

        for layer in range(NUM_LAYERS):
            if pooling == "cls":
                vec = good_hidden[layer][0]          # [CLS] token
            elif pooling == "mean":
                vec = good_hidden[layer].mean(axis=0) # mean pool
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            layer_lists[layer].append(vec)

    return [np.array(vecs) for vecs in layer_lists]


def compute_adjacent_cka(layer_reps: list[np.ndarray]) -> list[float]:
    """
    Compute CKA between each pair of adjacent layers.

    Returns:
        List of 12 CKA values: [CKA(0,1), CKA(1,2), ..., CKA(11,12)]
    """
    adjacent = []
    for i in range(len(layer_reps) - 1):
        sim = linear_cka(layer_reps[i], layer_reps[i + 1])
        adjacent.append(sim)
    return adjacent


def find_rewriting_events(adjacent_cka: list[float], threshold: float = 0.05) -> list[int]:
    """
    Identify layers where adjacent CKA drops sharply (rewriting events).

    A rewriting event is where CKA(i, i+1) is significantly lower than
    the running average, indicating the layer is actively transforming
    rather than passing through.

    Returns:
        List of layer indices where rewriting occurs.
    """
    mean_cka = np.mean(adjacent_cka)
    rewriting = []
    for i, cka_val in enumerate(adjacent_cka):
        if cka_val < mean_cka - threshold:
            rewriting.append(i)
    return rewriting


def plot_adjacent_cka_curves(
    all_adjacent: dict,
    save_path: Path = None,
):
    """
    Plot adjacent-layer CKA curves for each phenomenon, overlaid.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for phenom, adj_cka in sorted(all_adjacent.items()):
        transitions = [f"{i}→{i+1}" for i in range(len(adj_cka))]
        ax.plot(range(len(adj_cka)), adj_cka, marker="o", label=phenom, linewidth=2)

    ax.set_xlabel("Layer Transition")
    ax.set_ylabel("CKA Similarity")
    ax.set_title("Adjacent-Layer CKA: Preserve vs. Rewrite")
    ax.set_xticks(range(12))
    ax.set_xticklabels([f"{i}→{i+1}" for i in range(12)], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_cka_vs_probing(
    adjacent_cka: dict,
    exp1_results_path: Path,
    save_path: Path = None,
):
    """
    Overlay adjacent CKA (inverted = rewriting strength) with probing
    accuracy jumps to see if rewriting events coincide with probing gains.
    """
    if not exp1_results_path.exists():
        print("  Skipping CKA vs probing overlay (Exp 1 results not found)")
        return

    with open(exp1_results_path) as f:
        exp1_data = json.load(f)

    phenomena = sorted(set(adjacent_cka.keys()) & set(exp1_data.keys()))
    if not phenomena:
        print("  No overlapping phenomena between CKA and Exp 1")
        return

    fig, axes = plt.subplots(1, len(phenomena), figsize=(6 * len(phenomena), 5), sharey=False)
    if len(phenomena) == 1:
        axes = [axes]

    for ax, phenom in zip(axes, phenomena):
        adj = adjacent_cka[phenom]
        # Rewriting strength = 1 - CKA (higher = more rewriting)
        rewrite_strength = [1.0 - c for c in adj]

        # Probing accuracy jumps between adjacent layers
        probing_accs = [r["accuracy"] for r in exp1_data[phenom]]
        probing_jumps = [probing_accs[i+1] - probing_accs[i] for i in range(12)]

        x = range(12)
        ax2 = ax.twinx()

        ax.bar(x, rewrite_strength, alpha=0.4, color="steelblue", label="Rewriting (1-CKA)")
        ax2.plot(x, probing_jumps, "ro-", linewidth=2, label="Probing Δacc")

        ax.set_xlabel("Layer Transition")
        ax.set_ylabel("Rewriting Strength (1 - CKA)", color="steelblue")
        ax2.set_ylabel("Probing Accuracy Jump", color="red")
        ax.set_title(f"{phenom}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{i}→{i+1}" for i in range(12)], rotation=45, fontsize=7)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.suptitle("Rewriting Events vs. Probing Accuracy Gains", fontsize=13)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def run_cka_experiment(
    hidden_states_dir: Path = None,
    metadata_path: Path = None,
    pooling: str = "cls",
    results_dir: Path = None,
    figures_dir: Path = None,
    results_filename: str = "cka_results.json",
    label: str = None,
):
    """
    Run the full CKA preserve-or-rewrite analysis.

    1. Load metadata, group by phenomenon
    2. Collect layer representations per phenomenon
    3. Compute full CKA matrix and adjacent-layer CKA
    4. Identify rewriting events
    5. Compare against Exp 1 probing peaks
    6. Save results and generate plots

    `results_dir` / `figures_dir` override the default Exp 2 output paths,
    used by Exp 4 to write fine-tuned-model CKA results separately.
    """
    if hidden_states_dir is None:
        hidden_states_dir = PROCESSED_DIR / "hidden_states"
    if metadata_path is None:
        metadata_path = hidden_states_dir / "metadata.json"
    if results_dir is None:
        results_dir = RESULTS_DIR
    if figures_dir is None:
        figures_dir = FIGURES_DIR

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Group by phenomenon
    phenomena = {}
    for entry in metadata:
        phenom = entry["phenomenon"]
        if phenom not in phenomena:
            phenomena[phenom] = []
        phenomena[phenom].append(entry)

    all_adjacent = {}       # {phenom: [12 CKA values]}
    all_cka_matrices = {}   # {phenom: (13,13) matrix}
    all_rewriting = {}      # {phenom: [layer indices]}

    # ── Also compute CKA on ALL data (not split by phenomenon) ──
    print("\n[1/2] Computing CKA on full corpus...")
    full_reps = collect_layer_representations(hidden_states_dir, metadata, pooling)
    full_cka_matrix = compute_cka_matrix(full_reps)
    full_adjacent = compute_adjacent_cka(full_reps)

    print("\n  Full corpus adjacent-layer CKA:")
    print(f"  {'Transition':<12} {'CKA':>8} {'Rewriting':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*10}")
    for i, cka_val in enumerate(full_adjacent):
        rewrite = 1.0 - cka_val
        marker = " ◀ rewriting" if cka_val < np.mean(full_adjacent) - 0.05 else ""
        print(f"  {i:>2} → {i+1:<2}      {cka_val:8.4f} {rewrite:10.4f}{marker}")

    # ── Per-phenomenon CKA ──
    print("\n[2/2] Computing CKA per phenomenon...")
    for phenom, entries in sorted(phenomena.items()):
        print(f"\n{'='*60}")
        print(f"Phenomenon: {phenom} ({len(entries)} pairs)")
        print(f"{'='*60}")

        layer_reps = collect_layer_representations(hidden_states_dir, entries, pooling)

        # Full pairwise CKA matrix
        cka_matrix = compute_cka_matrix(layer_reps)
        all_cka_matrices[phenom] = cka_matrix

        # Adjacent-layer CKA
        adj_cka = compute_adjacent_cka(layer_reps)
        all_adjacent[phenom] = adj_cka

        # Identify rewriting events
        rewriting = find_rewriting_events(adj_cka)
        all_rewriting[phenom] = rewriting

        print(f"\n  Adjacent-layer CKA:")
        print(f"  {'Transition':<12} {'CKA':>8} {'Rewriting':>10}")
        print(f"  {'-'*12} {'-'*8} {'-'*10}")
        for i, cka_val in enumerate(adj_cka):
            rewrite = 1.0 - cka_val
            marker = " ◀ rewriting" if i in rewriting else ""
            print(f"  {i:>2} → {i+1:<2}      {cka_val:8.4f} {rewrite:10.4f}{marker}")

        if rewriting:
            print(f"\n  Rewriting events at transitions: {[f'{i}→{i+1}' for i in rewriting]}")
        else:
            print(f"\n  No significant rewriting events detected")

    # ── Summary ──
    print(f"\n{'='*80}")
    print("SUMMARY: Rewriting Events by Phenomenon")
    print(f"{'='*80}")
    print(f"\n  {'Phenomenon':<15} {'Lowest CKA transition':<25} {'CKA value':>10} {'Rewriting layers'}")
    print(f"  {'-'*15} {'-'*25} {'-'*10} {'-'*20}")
    for phenom in sorted(all_adjacent.keys()):
        adj = all_adjacent[phenom]
        min_idx = int(np.argmin(adj))
        min_val = adj[min_idx]
        rewrite_str = ", ".join(f"{i}→{i+1}" for i in all_rewriting[phenom]) or "none"
        print(f"  {phenom:<15} {min_idx}→{min_idx+1:<22} {min_val:10.4f} {rewrite_str}")

    print(f"\n  Interpretation:")
    print(f"    LOW CKA  = layer is REWRITING (actively transforming representations)")
    print(f"    HIGH CKA = layer is PRESERVING (passing representations through)")
    print(f"    Rewriting events that coincide with probing accuracy jumps (Exp 1)")
    print(f"    indicate layers that are actively CONSTRUCTING grammatical features.")

    # ── Save results ──
    results_data = {
        "full_corpus": {
            "adjacent_cka": full_adjacent,
            "cka_matrix": full_cka_matrix.tolist(),
        },
        "per_phenomenon": {},
    }
    for phenom in sorted(all_adjacent.keys()):
        results_data["per_phenomenon"][phenom] = {
            "adjacent_cka": all_adjacent[phenom],
            "cka_matrix": all_cka_matrices[phenom].tolist(),
            "rewriting_events": all_rewriting[phenom],
        }

    results_path = results_dir / results_filename
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # ── Generate plots ──
    suffix = f"_{label}" if label else ""

    # 1. Full corpus CKA heatmap
    plot_cka_heatmap(full_cka_matrix, save_path=figures_dir / f"full_cka_heatmap{suffix}.png")

    # 2. Per-phenomenon CKA heatmaps
    for phenom, matrix in all_cka_matrices.items():
        plot_cka_heatmap(matrix, save_path=figures_dir / f"cka_heatmap_{phenom}{suffix}.png")

    # 3. Adjacent CKA curves overlaid
    plot_adjacent_cka_curves(all_adjacent, save_path=figures_dir / f"adjacent_cka_curves{suffix}.png")

    # 4. CKA vs probing overlay (only meaningful for the base run)
    exp1_path = Path(__file__).resolve().parents[2] / "results" / "exp1_probing" / "probing_results.json"
    plot_cka_vs_probing(all_adjacent, exp1_path, save_path=figures_dir / f"cka_vs_probing{suffix}.png")

    return results_data


if __name__ == "__main__":
    run_cka_experiment()
