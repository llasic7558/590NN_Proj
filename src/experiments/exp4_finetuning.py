"""
Experiment 4: Fine-tuning Comparison (CoLA vs. SST-2)

Asks whether grammar-specific fine-tuning (CoLA) reorganizes the network
differently than semantic fine-tuning (SST-2), and if so, where.

Method:
  1. Fine-tune BERT on CoLA (grammaticality judgments).
  2. Fine-tune BERT on SST-2 (sentiment classification).
  3. Re-extract hidden states for the minimal-pair corpus from each
     fine-tuned model into separate dirs.
  4. Re-run Exp 1 probing and Exp 2 CKA on each set of hidden states.
  5. Plot 3-way overlays (base vs CoLA vs SST-2) and a divergence summary.

Expected (from the proposal): CoLA fine-tuning shifts middle-layer
geometry in ways SST-2 fine-tuning does not, providing evidence that
grammar-specific training has a representational signature distinct
from semantic fine-tuning.
"""

import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset

from src.extraction.extract_states import extract_corpus
from src.experiments.exp1_probing import run_probing_experiment
from src.experiments.exp2_cka import run_cka_experiment

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "exp4_finetuning"
FIGURES_DIR = PROJECT_ROOT / "figures" / "exp4"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

GLUE_TASKS = {
    "cola": {"hf_name": "glue", "config": "cola", "text_key": "sentence"},
    "sst2": {"hf_name": "glue", "config": "sst2", "text_key": "sentence"},
}


# ---------------------------------------------------------------------------
# 1. Fine-tuning
# ---------------------------------------------------------------------------

def finetune_on_task(
    task_name: str,
    model_name: str = "bert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    output_dir: Path = None,
    overwrite: bool = False,
) -> Path:
    """
    Fine-tune BERT on a GLUE task (CoLA or SST-2) and save the checkpoint.

    Returns the path to the saved checkpoint.
    """
    if task_name not in GLUE_TASKS:
        raise ValueError(f"Unsupported task {task_name}; use one of {list(GLUE_TASKS)}")

    spec = GLUE_TASKS[task_name]
    if output_dir is None:
        output_dir = CHECKPOINTS_DIR / f"bert-{task_name}"
    output_dir = Path(output_dir)

    if output_dir.exists() and not overwrite:
        if (output_dir / "config.json").exists():
            print(f"  Found existing checkpoint at {output_dir}, skipping fine-tune.")
            return output_dir

    print(f"\nFine-tuning BERT on {task_name.upper()}")
    print(f"  model={model_name}  epochs={num_epochs}  bs={batch_size}  lr={learning_rate}")

    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    raw = load_dataset(spec["hf_name"], spec["config"])

    def tokenize(batch):
        return tokenizer(
            batch[spec["text_key"]],
            truncation=True,
            max_length=max_length,
        )

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in ("input_ids", "attention_mask", "token_type_ids", "label")]
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_steps=200,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=collator,
    )
    trainer.train()

    if "validation" in tokenized:
        eval_metrics = trainer.evaluate()
        print(f"  Eval on {task_name}: {eval_metrics}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved fine-tuned checkpoint to {output_dir}")

    # Trainer's intermediate dir is no longer needed
    trainer_dir = output_dir / "trainer"
    if trainer_dir.exists():
        shutil.rmtree(trainer_dir, ignore_errors=True)

    return output_dir


# ---------------------------------------------------------------------------
# 2. Per-model probing + CKA
# ---------------------------------------------------------------------------

def run_probing_and_cka_for_model(
    label: str,
    checkpoint_path: Path,
    pooling: str = "cls",
    overwrite_states: bool = False,
):
    """
    For a single (base or fine-tuned) model:
      - Re-extract hidden states for the minimal-pair corpus into a
        label-specific dir.
      - Run Exp 1 probing on those states, saving results under exp4.
      - Run Exp 2 CKA on those states, saving results under exp4.

    `label` is e.g. "base", "cola", "sst2" and is used to name output dirs.
    `checkpoint_path` may be a HuggingFace name (e.g. "bert-base-uncased")
    or a local fine-tuned dir.
    """
    states_dir = PROCESSED_DIR / f"hidden_states_{label}"
    metadata_path = states_dir / "metadata.json"

    if overwrite_states or not metadata_path.exists():
        print(f"\n[{label}] Extracting hidden states from {checkpoint_path}...")
        extract_corpus(
            output_dir=states_dir,
            model_name_or_path=str(checkpoint_path),
            tokenizer_name="bert-base-uncased",
        )
    else:
        print(f"\n[{label}] Re-using existing hidden states at {states_dir}")

    print(f"[{label}] Running probing...")
    run_probing_experiment(
        hidden_states_dir=states_dir,
        metadata_path=metadata_path,
        pooling=pooling,
        results_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        results_filename=f"{label}_probing.json",
        label=label,
    )

    print(f"[{label}] Running CKA...")
    run_cka_experiment(
        hidden_states_dir=states_dir,
        metadata_path=metadata_path,
        pooling=pooling,
        results_dir=RESULTS_DIR,
        figures_dir=FIGURES_DIR,
        results_filename=f"{label}_cka.json",
        label=label,
    )


# ---------------------------------------------------------------------------
# 3. 3-way comparison plotting
# ---------------------------------------------------------------------------

MODEL_LABELS = ["base", "cola", "sst2"]
MODEL_COLORS = {"base": "#444444", "cola": "#1976d2", "sst2": "#d32f2f"}


def _load_probing(label: str) -> dict:
    path = RESULTS_DIR / f"{label}_probing.json"
    with open(path) as f:
        return json.load(f)


def _load_cka(label: str) -> dict:
    path = RESULTS_DIR / f"{label}_cka.json"
    with open(path) as f:
        return json.load(f)


def plot_probing_3way(save_path: Path = None):
    """One subplot per phenomenon, three probing curves overlaid."""
    runs = {label: _load_probing(label) for label in MODEL_LABELS}
    phenomena = sorted(set.intersection(*[set(r.keys()) for r in runs.values()]))

    fig, axes = plt.subplots(1, len(phenomena), figsize=(6 * len(phenomena), 5), sharey=True)
    if len(phenomena) == 1:
        axes = [axes]

    for ax, phenom in zip(axes, phenomena):
        for label in MODEL_LABELS:
            layer_results = runs[label][phenom]
            xs = [r["layer"] for r in layer_results]
            ys = [r["accuracy"] for r in layer_results]
            ax.plot(xs, ys, marker="o", label=label, color=MODEL_COLORS[label], linewidth=2)
        ax.set_title(phenom)
        ax.set_xlabel("BERT Layer")
        ax.set_ylabel("Probing accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.set_xticks(range(13))
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Probing accuracy: base vs CoLA vs SST-2 fine-tuned BERT", fontsize=13)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_cka_3way(save_path: Path = None):
    """Adjacent-layer CKA, base vs CoLA vs SST-2, one subplot per phenomenon."""
    runs = {label: _load_cka(label) for label in MODEL_LABELS}

    phenomena = sorted(set.intersection(
        *[set(r["per_phenomenon"].keys()) for r in runs.values()]
    ))

    fig, axes = plt.subplots(1, len(phenomena), figsize=(6 * len(phenomena), 5), sharey=True)
    if len(phenomena) == 1:
        axes = [axes]

    for ax, phenom in zip(axes, phenomena):
        for label in MODEL_LABELS:
            adj = runs[label]["per_phenomenon"][phenom]["adjacent_cka"]
            xs = list(range(len(adj)))
            ax.plot(xs, adj, marker="o", label=label, color=MODEL_COLORS[label], linewidth=2)
        ax.set_title(phenom)
        ax.set_xlabel("Layer transition")
        ax.set_ylabel("Adjacent CKA")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{i}→{i+1}" for i in xs], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Adjacent-layer CKA: base vs CoLA vs SST-2", fontsize=13)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_divergence_from_base(save_path: Path = None):
    """
    For each phenomenon, plot (cola - base) and (sst2 - base) probing
    accuracy deltas per layer. The proposal predicts CoLA's curve will
    diverge from base in the middle layers more than SST-2's does.
    """
    runs = {label: _load_probing(label) for label in MODEL_LABELS}
    phenomena = sorted(set.intersection(*[set(r.keys()) for r in runs.values()]))

    fig, axes = plt.subplots(1, len(phenomena), figsize=(6 * len(phenomena), 5), sharey=True)
    if len(phenomena) == 1:
        axes = [axes]

    for ax, phenom in zip(axes, phenomena):
        base = np.array([r["accuracy"] for r in runs["base"][phenom]])
        cola = np.array([r["accuracy"] for r in runs["cola"][phenom]])
        sst2 = np.array([r["accuracy"] for r in runs["sst2"][phenom]])
        xs = list(range(len(base)))
        ax.axhline(0, color="black", linewidth=0.8)
        ax.plot(xs, cola - base, marker="o", color=MODEL_COLORS["cola"], label="CoLA − base")
        ax.plot(xs, sst2 - base, marker="s", color=MODEL_COLORS["sst2"], label="SST-2 − base")
        ax.set_title(phenom)
        ax.set_xlabel("BERT Layer")
        ax.set_ylabel("Δ accuracy vs base")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Per-layer probing divergence from base BERT", fontsize=13)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def compare_finetuned_models():
    """Generate the 3-way overlay plots and a small text summary."""
    plot_probing_3way(save_path=FIGURES_DIR / "probing_3way.png")
    plot_cka_3way(save_path=FIGURES_DIR / "cka_3way.png")
    plot_divergence_from_base(save_path=FIGURES_DIR / "probing_divergence.png")

    # Text summary: peak layer per (model, phenomenon)
    runs = {label: _load_probing(label) for label in MODEL_LABELS}
    phenomena = sorted(set.intersection(*[set(r.keys()) for r in runs.values()]))
    print(f"\n{'='*80}")
    print("PEAK PROBING LAYER BY (MODEL, PHENOMENON)")
    print(f"{'='*80}")
    header = f"  {'phenom':<14}" + "".join(f"  {l:>14}" for l in MODEL_LABELS)
    print(header)
    for phenom in phenomena:
        cells = []
        for label in MODEL_LABELS:
            layer_results = runs[label][phenom]
            best = max(layer_results, key=lambda r: r["accuracy"])
            cells.append(f"L{best['layer']:>2} ({best['accuracy']:.3f})")
        print(f"  {phenom:<14}" + "".join(f"  {c:>14}" for c in cells))


# ---------------------------------------------------------------------------
# 4. Top-level driver
# ---------------------------------------------------------------------------

def run_finetuning_experiment(
    skip_finetune: bool = False,
    pooling: str = "cls",
):
    """
    Run the full Exp 4 pipeline.

    1. Fine-tune CoLA (skipped if checkpoint exists)
    2. Fine-tune SST-2 (skipped if checkpoint exists)
    3. For each of {base, cola, sst2}: re-extract hidden states,
       run probing, run CKA.
    4. Generate 3-way comparison figures and a peak-layer summary.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    if not skip_finetune:
        finetune_on_task("cola")
        finetune_on_task("sst2")

    cola_ckpt = CHECKPOINTS_DIR / "bert-cola"
    sst2_ckpt = CHECKPOINTS_DIR / "bert-sst2"

    run_probing_and_cka_for_model("base", "bert-base-uncased", pooling=pooling)
    run_probing_and_cka_for_model("cola", cola_ckpt, pooling=pooling)
    run_probing_and_cka_for_model("sst2", sst2_ckpt, pooling=pooling)

    compare_finetuned_models()


if __name__ == "__main__":
    run_finetuning_experiment()
