"""
Experiment 3: Attention Head Ablation
Weeks 5-6

Isolates which specific attention heads are *causally responsible*
for grammatical encoding, rather than which layers contain it.

Method: Zero out individual attention heads one at a time and measure
the drop in probing accuracy at the layer where each phenomenon peaks
(as identified in Exp 1).

This moves from correlation (probing) to causal claims about which
components shape grammatical representations.

Steps:
  1. Identify peak probing layers per phenomenon from Exp 1
  2. For each of 144 heads (12 layers × 12 heads):
     a. Zero out the head's attention weights
     b. Re-extract hidden states at the peak layer
     c. Re-run the trained probe
     d. Record accuracy drop
  3. Rank heads by causal importance per phenomenon
  4. Generate ablation impact visualizations

Expected: A small subset of heads in middle layers drive agreement;
different heads may matter for negation vs. clauses.
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import BertModel, BertTokenizer

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp3_ablation"


def ablate_head(model: BertModel, layer: int, head: int):
    """
    Zero out a specific attention head's output weights.

    TODO (weeks 5-6):
      - Access model.encoder.layer[layer].attention.self
      - Zero out the appropriate slice of the output projection
      - Return a context manager or restore function
    """
    raise NotImplementedError("Implement in weeks 5-6")


def measure_ablation_impact(
    model: BertModel,
    tokenizer: BertTokenizer,
    sentences: list[str],
    probe,
    target_layer: int,
    layer_idx: int,
    head_idx: int,
) -> float:
    """
    Ablate one head and measure probing accuracy drop.

    Returns:
        Accuracy drop (baseline_acc - ablated_acc). Positive = head matters.
    """
    # TODO: Implement
    #   1. Get baseline accuracy (probe on unmodified hidden states)
    #   2. Ablate the head
    #   3. Re-extract hidden states
    #   4. Run probe on ablated states
    #   5. Restore the head
    #   6. Return accuracy difference
    raise NotImplementedError("Implement in weeks 5-6")


def run_ablation_experiment(
    exp1_results_path: Path = None,
    hidden_states_dir: Path = None,
):
    """
    Run the full attention head ablation study.

    TODO (weeks 5-6):
      1. Load Exp 1 results to identify peak layers per phenomenon
      2. Load trained probes or retrain at peak layers
      3. For each phenomenon:
         a. For each of 144 heads, measure ablation impact
         b. Rank heads by impact
      4. Save results and generate visualizations
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Implement full experiment
    raise NotImplementedError("Implement in weeks 5-6")


if __name__ == "__main__":
    run_ablation_experiment()
