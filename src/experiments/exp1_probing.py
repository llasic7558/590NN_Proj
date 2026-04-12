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
    results_dir: Path = None,
    figures_dir: Path = None,
    results_filename: str = "probing_results.json",
    label: str = None,
):
    """
    Run layer-wise probing across all phenomena.

    Steps:
      1. Load metadata and group by phenomenon
      2. For each phenomenon × layer, train a linear probe
      3. Optionally compute control task selectivity
      4. Save results and generate plots

    `results_dir` / `figures_dir` override the default Exp 1 output paths,
    used by Exp 4 to write fine-tuned-model results without clobbering
    the base-model probing results.
    """
    if hidden_states_dir is None:
        hidden_states_dir = PROCESSED_DIR / "hidden_states"
    if metadata_path is None:
        metadata_path = hidden_states_dir / "metadata.json"
    if results_dir is None:
        results_dir = RESULTS_DIR
    if figures_dir is None:
        figures_dir = Path(__file__).resolve().parents[2] / "figures" / "exp1"

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

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

    # ── Summary tables ──
    print(f"\n{'='*80}")
    print("SUMMARY: Full results table")
    print(f"{'='*80}")
    for phenom, layer_results in sorted(all_results.items()):
        print(f"\n  [{phenom.upper()}] ({layer_results[0].n_samples} samples)")
        print(f"  {'Layer':>5}  {'Accuracy':>8}  {'± Std':>7}  {'F1':>6}  {'Control':>7}  {'Selectivity':>11}")
        print(f"  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*11}")
        for r in layer_results:
            print(f"  {r.layer:5d}  {r.accuracy:8.3f}  {r.accuracy_std:7.3f}  {r.f1:6.3f}  "
                  f"{r.control_accuracy:7.3f}  {r.selectivity:11.3f}")

    # ── Key findings ──
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    for phenom, layer_results in sorted(all_results.items()):
        best = max(layer_results, key=lambda r: r.accuracy)
        worst = min(layer_results, key=lambda r: r.accuracy)

        # Find where accuracy first exceeds 0.6 (above chance)
        first_useful = next((r for r in layer_results if r.accuracy > 0.6), None)

        # Find biggest jump between adjacent layers
        biggest_jump_layer = 0
        biggest_jump = 0
        for i in range(1, len(layer_results)):
            jump = layer_results[i].accuracy - layer_results[i-1].accuracy
            if jump > biggest_jump:
                biggest_jump = jump
                biggest_jump_layer = i

        print(f"\n  [{phenom.upper()}]")
        print(f"    Peak accuracy:     {best.accuracy:.3f} at layer {best.layer}")
        print(f"    Lowest accuracy:   {worst.accuracy:.3f} at layer {worst.layer}")
        print(f"    Best selectivity:  {max(r.selectivity for r in layer_results):.3f} "
              f"at layer {max(layer_results, key=lambda r: r.selectivity).layer}")
        if first_useful:
            print(f"    First useful (>0.6): layer {first_useful.layer} ({first_useful.accuracy:.3f})")
        else:
            print(f"    First useful (>0.6): never reaches 0.6")
        print(f"    Biggest jump:      +{biggest_jump:.3f} from layer {biggest_jump_layer-1}→{biggest_jump_layer}")
        if layer_results[-1].accuracy < best.accuracy - 0.01:
            print(f"    ⚠ Layer 12 drop:   {best.accuracy:.3f} → {layer_results[-1].accuracy:.3f} "
                  f"({best.accuracy - layer_results[-1].accuracy:.3f} decrease from peak)")

    # ── Cross-phenomenon comparison ──
    print(f"\n{'='*80}")
    print("CROSS-PHENOMENON COMPARISON (accuracy at each layer)")
    print(f"{'='*80}")
    phenoms_sorted = sorted(all_results.keys())
    header = f"  {'Layer':>5}" + "".join(f"  {p:>12}" for p in phenoms_sorted)
    print(header)
    print(f"  {'-'*5}" + "".join(f"  {'-'*12}" for _ in phenoms_sorted))
    for layer in range(NUM_LAYERS):
        row = f"  {layer:5d}"
        for phenom in phenoms_sorted:
            acc = all_results[phenom][layer].accuracy
            # Mark the peak layer for each phenomenon
            peak = max(r.accuracy for r in all_results[phenom])
            marker = " ★" if acc == peak else "  "
            row += f"  {acc:10.3f}{marker}"
        print(row)
    print(f"\n  ★ = peak layer for that phenomenon")

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

    results_path = results_dir / results_filename
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""

    plot_layer_accuracy_curves(
        all_results, save_path=figures_dir / f"layer_accuracy_curves{suffix}.png"
    )
    plot_probing_heatmap(
        all_results, save_path=figures_dir / f"probing_heatmap{suffix}.png"
    )

    return all_results


if __name__ == "__main__":
    run_probing_experiment()
