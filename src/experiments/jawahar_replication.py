"""
Jawahar et al. (2019) Replication — Sanity Check

Replicates the core probing experiments from:
  "What does BERT learn about the structure of language?" (ACL 2019)

Uses the SAME setup as the original paper:
  - SentEval probing tasks (10 tasks across surface/syntactic/semantic)
  - Mean pooling across tokens (not [CLS])
  - Linear classifier (logistic regression)
  - Probing at all 13 BERT layers

This serves as a go/no-go validation: if our pipeline recovers their
finding that surface→syntax→semantics maps to lower→middle→upper layers,
we can trust our Exp 1-5 results.

Expected results (from the paper):
  - Surface tasks (SentLen, WC): peak at layers 0-2
  - Syntactic tasks (TreeDepth, TopConst, BShift): peak at layers 4-8
  - Semantic tasks (Tense, SubjNum, ObjNum, SOMO, CoordInv): peak at layers 8-12
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "jawahar_replication"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "jawahar_replication"
NUM_LAYERS = 13

# ─── SentEval probing tasks ───
# Maps task name → (category, number of classes)
SENTEVAL_TASKS = {
    "sentence_length":        ("surface",   7),
    "word_content":           ("surface",   1000),
    "tree_depth":             ("syntactic", 8),
    "top_constituents":       ("syntactic", 20),
    "bigram_shift":           ("syntactic", 2),
    "past_present":           ("semantic",  2),
    "subj_number":            ("semantic",  2),
    "obj_number":             ("semantic",  2),
    "odd_man_out":            ("semantic",  2),
    "coordination_inversion": ("semantic",  2),
}


def load_senteval_task(task_name: str, max_samples: int = 10000) -> tuple[list[str], np.ndarray]:
    """
    Load a SentEval probing task from HuggingFace.

    Returns:
        sentences: list of sentence strings
        labels: np.array of integer labels
    """
    ds = load_dataset("tasksource/linguisticprobing", task_name)

    sentences, labels = [], []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for row in ds[split]:
            sentences.append(row["sentence"])
            labels.append(row["label"])
            if len(sentences) >= max_samples:
                break
        if len(sentences) >= max_samples:
            break

    return sentences[:max_samples], np.array(labels[:max_samples])


@torch.no_grad()
def extract_mean_pooled(
    model: BertModel,
    tokenizer: BertTokenizer,
    sentences: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Extract mean-pooled hidden states at all layers for a list of sentences.

    Returns:
        np.array of shape (13, n_sentences, 768)
    """
    all_hidden = [[] for _ in range(NUM_LAYERS)]

    for i in tqdm(range(0, len(sentences), batch_size), desc="  Extracting"):
        batch = sentences[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=128,
        )
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        # Mean pool over non-padding tokens
        mask = attention_mask.unsqueeze(0).unsqueeze(-1).float().to(device)
        # hidden_states: tuple of 13 tensors, each (batch, seq_len, 768)
        for layer_idx, hidden in enumerate(outputs.hidden_states):
            # Mask out padding, then mean
            masked = hidden * mask[0]                    # (batch, seq_len, 768)
            lengths = attention_mask.sum(dim=1, keepdim=True).float().to(device)  # (batch, 1)
            pooled = masked.sum(dim=1) / lengths         # (batch, 768)
            all_hidden[layer_idx].append(pooled.cpu().numpy())

    # Stack: each layer → (n_sentences, 768)
    result = np.array([np.concatenate(layer_list, axis=0) for layer_list in all_hidden])
    return result  # (13, n_sentences, 768)


def probe_task_at_layer(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> tuple[float, float]:
    """
    Train a logistic regression probe with cross-validation.

    Returns:
        (mean_accuracy, std_accuracy)
    """
    # Skip if too few classes represented
    unique = np.unique(y)
    if len(unique) < 2:
        return 0.0, 0.0

    # For high-cardinality tasks (word_content), use fewer folds
    min_class_count = min(np.bincount(y)[np.bincount(y) > 0])
    actual_folds = min(n_folds, min_class_count)
    if actual_folds < 2:
        return 0.0, 0.0

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[val_idx])
        accs.append(accuracy_score(y[val_idx], y_pred))

    return float(np.mean(accs)), float(np.std(accs))


def run_jawahar_replication(max_samples: int = 5000):
    """
    Run the full Jawahar et al. replication.

    For each SentEval task, extracts mean-pooled representations at all
    13 layers and trains a linear probe at each layer.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load BERT
    print("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained(
        "bert-base-uncased", output_hidden_states=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    all_results = {}  # {task_name: {"category": str, "layers": [{layer, acc, std}]}}

    for task_name, (category, n_classes) in SENTEVAL_TASKS.items():
        print(f"\n{'='*60}")
        print(f"Task: {task_name} ({category}, {n_classes} classes)")
        print(f"{'='*60}")

        # Load data
        try:
            sentences, labels = load_senteval_task(task_name, max_samples=max_samples)
        except Exception as e:
            print(f"  ⚠ Could not load task: {e}")
            continue

        print(f"  Loaded {len(sentences)} samples, {len(np.unique(labels))} classes")

        # Extract hidden states (mean pooled)
        hidden = extract_mean_pooled(model, tokenizer, sentences, device)
        # hidden shape: (13, n_sentences, 768)

        # Probe at each layer
        layer_results = []
        for layer in range(NUM_LAYERS):
            X = hidden[layer]  # (n_sentences, 768)
            acc, std = probe_task_at_layer(X, labels)
            layer_results.append({"layer": layer, "accuracy": acc, "accuracy_std": std})
            print(f"  Layer {layer:2d}: acc={acc:.3f} ± {std:.3f}")

        best = max(layer_results, key=lambda r: r["accuracy"])
        print(f"  → Peak: layer {best['layer']} ({best['accuracy']:.3f})")

        all_results[task_name] = {
            "category": category,
            "n_classes": n_classes,
            "n_samples": len(sentences),
            "layers": layer_results,
        }

    # ── Print comparison summary ──
    print(f"\n{'='*80}")
    print("JAWAHAR REPLICATION SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Expected (from paper): surface→layers 0-2, syntax→layers 4-8, semantic→layers 8-12")
    print(f"\n  {'Task':<25} {'Category':<10} {'Peak Layer':>10} {'Peak Acc':>9} {'Match?':>7}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*9} {'-'*7}")

    expected_ranges = {
        "surface":  (0, 3),   # layers 0-2
        "syntactic": (3, 9),  # layers 3-8
        "semantic": (7, 13),  # layers 7-12
    }

    for task_name, data in sorted(all_results.items(), key=lambda x: x[1]["category"]):
        best = max(data["layers"], key=lambda r: r["accuracy"])
        cat = data["category"]
        lo, hi = expected_ranges[cat]
        match = "✓" if lo <= best["layer"] < hi else "✗"
        print(f"  {task_name:<25} {cat:<10} {best['layer']:>10} {best['accuracy']:>9.3f} {match:>7}")

    # ── Save results ──
    with open(RESULTS_DIR / "jawahar_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'jawahar_results.json'}")

    # ── Generate plot ──
    _plot_replication(all_results)

    return all_results


def _plot_replication(all_results: dict):
    """Plot probing accuracy curves grouped by category."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    categories = ["surface", "syntactic", "semantic"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for ax, category in zip(axes, categories):
        color_idx = 0
        for task_name, data in sorted(all_results.items()):
            if data["category"] != category:
                continue
            layers = [r["layer"] for r in data["layers"]]
            accs = [r["accuracy"] for r in data["layers"]]
            ax.plot(layers, accs, marker="o", markersize=4, label=task_name, color=colors[color_idx])
            color_idx += 1

        ax.set_xlabel("BERT Layer")
        ax.set_title(f"{category.capitalize()} Tasks")
        ax.set_xticks(range(13))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Probing Accuracy")
    fig.suptitle("Jawahar et al. (2019) Replication: SentEval Probing Tasks", fontsize=13)
    plt.tight_layout()

    save_path = FIGURES_DIR / "jawahar_replication.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    run_jawahar_replication()
