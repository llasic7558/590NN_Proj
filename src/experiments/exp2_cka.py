"""
Experiment 2: Adjacent-Layer CKA (Preserve-or-Rewrite Analysis)
Weeks 3-4

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
from pathlib import Path

from src.utils.cka import linear_cka, compute_cka_matrix
from src.utils.visualization import plot_cka_heatmap

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp2_cka"


def collect_layer_representations(
    hidden_states_dir: Path,
    metadata: list[dict],
    pooling: str = "cls",
) -> list[np.ndarray]:
    """
    Collect representations at each layer across all sentences.

    Returns:
        List of 13 arrays, each (n_sentences, hidden_dim).
    """
    # TODO: Implement — load .npz files, extract per-layer representations
    #   For each pair, use the "good" sentence representation.
    #   Stack across sentences to get (n_sentences, 768) per layer.
    raise NotImplementedError("Implement in weeks 3-4")


def compute_adjacent_cka(layer_reps: list[np.ndarray]) -> list[float]:
    """
    Compute CKA between each pair of adjacent layers.

    Returns:
        List of 12 CKA values: [CKA(0,1), CKA(1,2), ..., CKA(11,12)]
    """
    # TODO: Implement using linear_cka from src.utils.cka
    raise NotImplementedError("Implement in weeks 3-4")


def run_cka_experiment(hidden_states_dir: Path = None, metadata_path: Path = None):
    """
    Run the full CKA preserve-or-rewrite analysis.

    TODO (weeks 3-4):
      1. Load metadata, group by phenomenon
      2. Collect layer representations
      3. Compute full CKA matrix and adjacent-layer CKA
      4. Compare CKA drops against Exp 1 probing peaks
      5. Save results and generate plots
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Implement full experiment
    raise NotImplementedError("Implement in weeks 3-4")


if __name__ == "__main__":
    run_cka_experiment()
