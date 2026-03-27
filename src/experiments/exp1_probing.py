"""
Experiment 1: Layer-wise Probing (Jawahar et al. 2019 Replication)

For each BERT layer (0-12), train a linear probe to classify
grammatical vs. ungrammatical sentences, broken down by phenomenon.

This establishes the foundational map of WHAT is encoded WHERE.

Output:
  - Per-layer accuracy/F1 for each phenomenon
  - Selectivity scores (probe accuracy - control task accuracy)
  - Probing accuracy heatmap (layer × phenomenon)
"""

import json
import numpy as np
from pathlib import Path

from src.probes.linear_probe import LinearProbe, ProbeResult, prepare_probing_data
from src.utils.visualization import plot_probing_heatmap, plot_layer_accuracy_curves

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp1_probing"
NUM_LAYERS = 13  # embedding layer + 12 transformer layers


def run_probing_experiment(
    hidden_states_dir: Path = None,
    metadata_path: Path = None,
    pooling: str = "cls",
    compute_selectivity: bool = True,
):
    """
    Run layer-wise probing across all phenomena.

    Steps:
      1. Load metadata and group by phenomenon
      2. For each phenomenon × layer, train a linear probe
      3. Optionally compute control task selectivity
      4. Save results and generate plots
    """
    if hidden_states_dir is None:
        hidden_states_dir = PROCESSED_DIR / "hidden_states"
    if metadata_path is None:
        metadata_path = hidden_states_dir / "metadata.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Group by phenomenon
    phenomena = {}
    for entry in metadata:
        phenom = entry["phenomenon"]
        if phenom not in phenomena:
            phenomena[phenom] = []
        phenomena[phenom].append(entry)

    probe = LinearProbe(n_folds=5, max_iter=1000)
    all_results = {}  # {phenomenon: [ProbeResult per layer]}

    for phenom, entries in sorted(phenomena.items()):
        print(f"\n{'='*60}")
        print(f"Phenomenon: {phenom} ({len(entries)} pairs)")
        print(f"{'='*60}")

        layer_results = []
        for layer in range(NUM_LAYERS):
            X, y = prepare_probing_data(hidden_states_dir, entries, layer, pooling)

            result = probe.train_and_evaluate(X, y, layer)

            # Control task for selectivity
            if compute_selectivity:
                rng = np.random.RandomState(42)
                y_random = rng.randint(0, 2, size=len(y))
                ctrl_acc = probe.control_task(X, y_random, layer)
                result.control_accuracy = ctrl_acc
                result.selectivity = result.accuracy - ctrl_acc

            layer_results.append(result)
            print(f"  Layer {layer:2d}: acc={result.accuracy:.3f} ± {result.accuracy_std:.3f}  "
                  f"F1={result.f1:.3f}  sel={result.selectivity:.3f}")

        all_results[phenom] = layer_results

    # Save results
    results_data = {}
    for phenom, results in all_results.items():
        results_data[phenom] = [
            {
                "layer": r.layer,
                "accuracy": r.accuracy,
                "accuracy_std": r.accuracy_std,
                "f1": r.f1,
                "f1_std": r.f1_std,
                "control_accuracy": r.control_accuracy,
                "selectivity": r.selectivity,
                "n_samples": r.n_samples,
            }
            for r in results
        ]

    with open(RESULTS_DIR / "probing_results.json", "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'probing_results.json'}")

    # Generate plots
    figures_dir = Path(__file__).resolve().parents[2] / "figures" / "exp1"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_layer_accuracy_curves(all_results, save_path=figures_dir / "layer_accuracy_curves.png")
    plot_probing_heatmap(all_results, save_path=figures_dir / "probing_heatmap.png")

    return all_results


if __name__ == "__main__":
    run_probing_experiment()
