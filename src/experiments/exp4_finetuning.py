"""
Experiment 4: Fine-tuning Comparison (CoLA vs. SST-2)
Weeks 6-7

Asks whether grammar-specific fine-tuning (CoLA) reorganizes the network
differently than semantic fine-tuning (SST-2), and if so, where.

Method:
  1. Fine-tune BERT on CoLA (grammaticality judgments)
  2. Fine-tune BERT on SST-2 (sentiment classification)
  3. Re-run Exp 1 probing on both fine-tuned models
  4. Re-run Exp 2 CKA on both fine-tuned models
  5. Compare probing curves and CKA profiles against base BERT

Expected: CoLA fine-tuning shifts middle-layer geometry in ways
SST-2 fine-tuning does not, providing evidence that grammar-specific
training has a distinct representational signature.
"""

import json
import torch
from pathlib import Path
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp4_finetuning"
CHECKPOINTS_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def finetune_on_task(
    task_name: str,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
):
    """
    Fine-tune BERT on a GLUE task (CoLA or SST-2).

    TODO (weeks 6-7):
      1. Load dataset from HuggingFace (glue/cola or glue/sst2)
      2. Tokenize with BertTokenizer
      3. Set up BertForSequenceClassification
      4. Train with HuggingFace Trainer
      5. Save checkpoint

    Args:
        task_name: "cola" or "sst2"
    """
    raise NotImplementedError("Implement in weeks 6-7")


def compare_finetuned_models(
    base_model_name: str = "bert-base-uncased",
    cola_checkpoint: Path = None,
    sst2_checkpoint: Path = None,
):
    """
    Compare probing and CKA profiles across base, CoLA-tuned, and SST-2-tuned BERT.

    TODO (weeks 6-7):
      1. Load all three models
      2. Extract hidden states for the minimal-pair corpus from each
      3. Run probing (reuse Exp 1 code) on each model's representations
      4. Run CKA (reuse Exp 2 code) on each model's representations
      5. Plot comparison: 3 probing curves overlaid, 3 CKA curves overlaid
      6. Statistical comparison of where the profiles diverge
    """
    raise NotImplementedError("Implement in weeks 6-7")


def run_finetuning_experiment():
    """
    Run the full fine-tuning comparison experiment.

    TODO (weeks 6-7):
      1. Fine-tune on CoLA
      2. Fine-tune on SST-2
      3. Compare all three models
      4. Save results and generate figures
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: Implement
    raise NotImplementedError("Implement in weeks 6-7")


if __name__ == "__main__":
    run_finetuning_experiment()
