"""
Experiment 3: Attention Head Ablation

Isolates which specific attention heads are *causally responsible*
for grammatical encoding, rather than which layers contain it.

Method: Zero out individual attention heads one at a time, re-run
sentences through BERT, and measure the drop in probing accuracy
at the peak layer (from Exp 1). This moves from correlation to
causal claims.

Steps:
  1. Identify peak probing layers per phenomenon from Exp 1
  2. Train probes at those peak layers (baseline)
  3. For each of 144 heads (12 layers × 12 heads):
     a. Zero out the head's output projection weights
     b. Re-extract hidden states for all sentences
     c. Evaluate the pre-trained probe on ablated states
     d. Record accuracy drop
     e. Restore the head
  4. Rank heads by causal importance per phenomenon
  5. Generate ablation impact visualizations

Expected: A small subset of heads in middle layers drive agreement;
different heads may matter for negation vs. clauses.
"""

import json
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.probes.linear_probe import LinearProbe, prepare_probing_data

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp3_ablation"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "exp3"
NUM_LAYERS = 12   # transformer layers (not embedding)
NUM_HEADS = 12
HEAD_DIM = 64     # 768 / 12 = 64


def get_peak_layers(exp1_results_path: Path) -> dict:
    """
    Load Exp 1 results and find the peak probing layer for each phenomenon.

    Returns:
        {phenomenon: peak_layer_index}
    """
    with open(exp1_results_path) as f:
        results = json.load(f)

    peak_layers = {}
    for phenom, layer_results in results.items():
        best = max(layer_results, key=lambda r: r["accuracy"])
        peak_layers[phenom] = best["layer"]

    return peak_layers


def train_baseline_probe(
    hidden_states_dir: Path,
    metadata: list[dict],
    layer: int,
    pooling: str = "cls",
) -> tuple[LogisticRegression, float, np.ndarray, np.ndarray]:
    """
    Train a logistic regression probe at the given layer and return it.

    Returns:
        (trained_classifier, baseline_accuracy, X, y)
    """
    X, y = prepare_probing_data(hidden_states_dir, metadata, layer, pooling)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42)
    clf.fit(X, y)
    baseline_acc = accuracy_score(y, clf.predict(X))

    return clf, baseline_acc, X, y


@torch.no_grad()
def extract_with_ablation(
    model: BertModel,
    tokenizer: BertTokenizer,
    sentences: list[str],
    target_layer: int,
    ablate_layer: int,
    ablate_head: int,
    device: torch.device,
    pooling: str = "cls",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Run sentences through BERT with one attention head zeroed out,
    and extract representations at target_layer.

    The ablation is done by hooking into the attention output and
    zeroing the slice corresponding to the target head.

    Returns:
        np.ndarray of shape (n_sentences, 768)
    """
    representations = []

    # Register a hook to zero out the specific head's contribution
    hook_handle = None

    def ablation_hook(module, input, output):
        # output is a tuple: (context_layer, attention_probs)
        # context_layer shape: (batch, seq_len, all_head_size=768)
        # We zero out the slice for the target head
        context = output[0]
        start = ablate_head * HEAD_DIM
        end = start + HEAD_DIM
        context[:, :, start:end] = 0.0
        return (context,) + output[1:]

    # Attach hook to the self-attention output of the target layer
    attn_module = model.encoder.layer[ablate_layer].attention.self
    hook_handle = attn_module.register_forward_hook(ablation_hook)

    try:
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Get hidden states at the target layer
            hidden = outputs.hidden_states[target_layer]  # (batch, seq_len, 768)

            if pooling == "cls":
                vecs = hidden[:, 0, :]           # (batch, 768)
            elif pooling == "mean":
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                vecs = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            representations.append(vecs.cpu().numpy())
    finally:
        hook_handle.remove()

    return np.concatenate(representations, axis=0)


def run_ablation_for_phenomenon(
    model: BertModel,
    tokenizer: BertTokenizer,
    device: torch.device,
    metadata: list[dict],
    hidden_states_dir: Path,
    peak_layer: int,
    phenom: str,
    pooling: str = "cls",
) -> dict:
    """
    Ablate each of 144 heads and measure impact on probing at the peak layer.

    Returns:
        {(layer, head): {"acc_drop": float, "ablated_acc": float, "baseline_acc": float}}
    """
    # Train baseline probe at peak layer
    print(f"  Training baseline probe at peak layer {peak_layer}...")
    clf, baseline_acc, X_base, y = train_baseline_probe(
        hidden_states_dir, metadata, peak_layer, pooling,
    )
    print(f"  Baseline accuracy: {baseline_acc:.4f}")

    # Prepare sentence lists (good + bad interleaved, matching prepare_probing_data)
    sentences = []
    for entry in metadata:
        sentences.append(entry["good"] if "good" in entry else entry.get("good", ""))
        sentences.append(entry["bad"] if "bad" in entry else entry.get("bad", ""))

    # Try loading from metadata.json which has the actual sentences
    meta_path = hidden_states_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            full_meta = json.load(f)
        idx_set = {e["idx"] for e in metadata}
        sentences = []
        for m in full_meta:
            if m["idx"] in idx_set:
                sentences.append(m["good"])
                sentences.append(m["bad"])

    ablation_results = {}

    for layer_idx in tqdm(range(NUM_LAYERS), desc=f"  Ablating layers for {phenom}"):
        for head_idx in range(NUM_HEADS):
            # Extract with this head zeroed out
            X_ablated = extract_with_ablation(
                model, tokenizer, sentences, peak_layer,
                layer_idx, head_idx, device, pooling,
            )

            # Evaluate the same (frozen) probe on ablated representations
            ablated_acc = accuracy_score(y, clf.predict(X_ablated))
            acc_drop = baseline_acc - ablated_acc

            ablation_results[(layer_idx, head_idx)] = {
                "acc_drop": float(acc_drop),
                "ablated_acc": float(ablated_acc),
                "baseline_acc": float(baseline_acc),
            }

    return ablation_results


def plot_ablation_heatmap(
    ablation_results: dict,
    phenom: str,
    save_path: Path = None,
):
    """
    Heatmap of accuracy drop: rows=layers, columns=heads.
    """
    grid = np.zeros((NUM_LAYERS, NUM_HEADS))
    for (layer, head), data in ablation_results.items():
        grid[layer, head] = data["acc_drop"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        grid, annot=True, fmt=".3f", cmap="RdYlBu_r",
        xticklabels=[f"H{h}" for h in range(NUM_HEADS)],
        yticklabels=[f"L{l+1}" for l in range(NUM_LAYERS)],
        ax=ax, center=0,
    )
    ax.set_xlabel("Attention Head")
    ax.set_ylabel("Layer")
    ax.set_title(f"Ablation Impact on {phenom.capitalize()} Probing\n(Accuracy drop: positive = head matters)")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def plot_top_heads(
    all_ablation: dict,
    top_n: int = 15,
    save_path: Path = None,
):
    """
    Bar chart of the top N most important heads across all phenomena.
    """
    phenomena = sorted(all_ablation.keys())
    fig, axes = plt.subplots(1, len(phenomena), figsize=(6 * len(phenomena), 5))
    if len(phenomena) == 1:
        axes = [axes]

    for ax, phenom in zip(axes, phenomena):
        results = all_ablation[phenom]
        sorted_heads = sorted(results.items(), key=lambda x: x[1]["acc_drop"], reverse=True)
        top = sorted_heads[:top_n]

        labels = [f"L{l+1}H{h}" for (l, h), _ in top]
        drops = [d["acc_drop"] for _, d in top]
        colors = ["#d32f2f" if d > 0.02 else "#1976d2" for d in drops]

        ax.barh(range(len(labels)), drops, color=colors, alpha=0.8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Accuracy Drop")
        ax.set_title(f"{phenom.capitalize()}")
        ax.invert_yaxis()
        ax.axvline(x=0.02, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.suptitle(f"Top {top_n} Most Important Attention Heads per Phenomenon", fontsize=13)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()


def run_ablation_experiment(
    exp1_results_path: Path = None,
    hidden_states_dir: Path = None,
    metadata_path: Path = None,
    pooling: str = "cls",
):
    """
    Run the full attention head ablation study.

    1. Load Exp 1 results to identify peak layers per phenomenon
    2. Load BERT model
    3. For each phenomenon, ablate all 144 heads and measure impact
    4. Rank heads, save results, generate visualizations
    """
    if hidden_states_dir is None:
        hidden_states_dir = PROCESSED_DIR / "hidden_states"
    if metadata_path is None:
        metadata_path = hidden_states_dir / "metadata.json"
    if exp1_results_path is None:
        exp1_results_path = Path(__file__).resolve().parents[2] / "results" / "exp1_probing" / "probing_results.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Get peak layers from Exp 1
    print("Loading Exp 1 results to identify peak layers...")
    peak_layers = get_peak_layers(exp1_results_path)
    print(f"  Peak layers: {peak_layers}")

    # 2. Load metadata and group by phenomenon
    with open(metadata_path) as f:
        metadata = json.load(f)

    phenomena = {}
    for entry in metadata:
        phenom = entry["phenomenon"]
        if phenom not in phenomena:
            phenomena[phenom] = []
        phenomena[phenom].append(entry)

    # 3. Load BERT
    print("\nLoading BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained(
        "bert-base-uncased", output_hidden_states=True, output_attentions=True,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # 4. Run ablation per phenomenon
    all_ablation = {}

    for phenom, entries in sorted(phenomena.items()):
        peak = peak_layers.get(phenom)
        if peak is None:
            print(f"\n  Skipping {phenom}: no peak layer found in Exp 1")
            continue

        print(f"\n{'='*60}")
        print(f"Phenomenon: {phenom} ({len(entries)} pairs, peak layer={peak})")
        print(f"{'='*60}")

        results = run_ablation_for_phenomenon(
            model, tokenizer, device, entries,
            hidden_states_dir, peak, phenom, pooling,
        )
        all_ablation[phenom] = results

        # Print top 10 most impactful heads
        sorted_heads = sorted(results.items(), key=lambda x: x[1]["acc_drop"], reverse=True)
        print(f"\n  Top 10 most impactful heads for {phenom}:")
        print(f"  {'Head':<10} {'Acc Drop':>10} {'Ablated Acc':>12} {'Baseline':>10}")
        print(f"  {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
        for (l, h), data in sorted_heads[:10]:
            print(f"  L{l+1:>2}H{h:<2}     {data['acc_drop']:10.4f} {data['ablated_acc']:12.4f} {data['baseline_acc']:10.4f}")

    # ── Cross-phenomenon summary ──
    print(f"\n{'='*80}")
    print("CROSS-PHENOMENON SUMMARY")
    print(f"{'='*80}")
    for phenom, results in sorted(all_ablation.items()):
        sorted_heads = sorted(results.items(), key=lambda x: x[1]["acc_drop"], reverse=True)
        causal_heads = [(l, h) for (l, h), d in sorted_heads if d["acc_drop"] > 0.02]
        print(f"\n  [{phenom.upper()}] (peak layer: {peak_layers[phenom]})")
        print(f"    Causally important heads (>2% acc drop): {len(causal_heads)}")
        if causal_heads:
            print(f"    Heads: {[f'L{l+1}H{h}' for l, h in causal_heads[:10]]}")
            layers_involved = sorted(set(l for l, h in causal_heads))
            print(f"    Layers involved: {[l+1 for l in layers_involved]}")

    # Check for shared causal heads across phenomena
    if len(all_ablation) > 1:
        print(f"\n  Shared causal heads across phenomena:")
        phenom_list = sorted(all_ablation.keys())
        for i, p1 in enumerate(phenom_list):
            for p2 in phenom_list[i+1:]:
                heads1 = {(l,h) for (l,h), d in all_ablation[p1].items() if d["acc_drop"] > 0.02}
                heads2 = {(l,h) for (l,h), d in all_ablation[p2].items() if d["acc_drop"] > 0.02}
                shared = heads1 & heads2
                if shared:
                    print(f"    {p1} ∩ {p2}: {[f'L{l+1}H{h}' for l, h in sorted(shared)]}")
                else:
                    print(f"    {p1} ∩ {p2}: none (phenomenon-specific heads)")

    # ── Save results ──
    save_data = {}
    for phenom, results in all_ablation.items():
        save_data[phenom] = {
            "peak_layer": peak_layers[phenom],
            "heads": {
                f"L{l}_H{h}": data for (l, h), data in results.items()
            },
        }

    with open(RESULTS_DIR / "ablation_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'ablation_results.json'}")

    # ── Generate plots ──
    # 1. Per-phenomenon heatmaps
    for phenom, results in all_ablation.items():
        plot_ablation_heatmap(results, phenom, save_path=FIGURES_DIR / f"ablation_heatmap_{phenom}.png")

    # 2. Top heads bar chart
    plot_top_heads(all_ablation, top_n=15, save_path=FIGURES_DIR / "top_heads.png")

    return all_ablation


if __name__ == "__main__":
    run_ablation_experiment()
