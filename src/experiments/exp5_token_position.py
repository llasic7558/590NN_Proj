"""
Experiment 5: Layer × Token-Position Grid
Weeks 5-6

Addresses a gap in Exp 1-4: they collapse across token positions,
treating each layer as a single snapshot. Since representations
belong to specific tokens, this experiment tracks whether the
grammatical signal lives at the subject, verb, or elsewhere,
and whether that changes across depth.

Method:
  1. For each minimal pair, identify critical token positions:
     - Subject position
     - Verb position
     - [CLS] position (for comparison)
     - Other positions (averaged)
  2. At each layer, train a probe using *only* the representation
     at that token position
  3. Build a (13 layers × 4 positions) accuracy grid
  4. Visualize how information "moves" across token positions through depth

Expected: Early layers encode grammar at the verb; middle layers
shift the signal to the subject; late layers spread it to [CLS].
"""

import json
import numpy as np
from pathlib import Path

from src.probes.linear_probe import LinearProbe

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp5_token_position"

TOKEN_ROLES = ["cls", "subject", "verb", "other"]


def identify_token_positions(
    tokens: list[str],
    sentence: str,
    phenomenon: str,
) -> dict[str, int]:
    """
    Identify the index of critical tokens in the tokenized sequence.

    TODO (weeks 5-6):
      - [CLS] is always index 0
      - Subject: first noun/pronoun before the main verb
      - Verb: the main verb (or the verb that differs between good/bad)
      - Other: average of remaining positions

    This will need heuristics or a simple POS tagger.

    Returns:
        {"cls": 0, "subject": i, "verb": j, "other": -1}
        where -1 means "average all other positions"
    """
    raise NotImplementedError("Implement in weeks 5-6")


def prepare_position_probing_data(
    hidden_states_dir: Path,
    metadata: list[dict],
    layer: int,
    token_role: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Like prepare_probing_data but uses a specific token position.

    TODO (weeks 5-6):
      1. For each pair, identify token positions via identify_token_positions()
      2. Extract the hidden state at the specified position
      3. Return (X, y) for probing
    """
    raise NotImplementedError("Implement in weeks 5-6")


def run_token_position_experiment(
    hidden_states_dir: Path = None,
    metadata_path: Path = None,
):
    """
    Run the layer × token-position probing grid experiment.

    TODO (weeks 5-6):
      1. Load metadata
      2. For each phenomenon:
         a. For each layer (0-12):
            - For each token role (cls, subject, verb, other):
              * Prepare data and train probe
              * Record accuracy
         b. Build (13, 4) accuracy grid
      3. Save results and generate heatmap
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Implement
    raise NotImplementedError("Implement in weeks 5-6")


if __name__ == "__main__":
    run_token_position_experiment()
