"""
Experiment 5: Layer × Token-Position Grid

Addresses a gap in Exp 1-4: they collapse across token positions,
treating each layer as a single snapshot. Since representations
belong to specific tokens, this experiment tracks whether the
grammatical signal lives at the subject, the verb, or elsewhere,
and whether that changes across depth.

Method:
  1. For each minimal pair, identify critical token positions in
     both the good and bad sentences via spaCy POS/dependency parsing:
       - [CLS] (always index 0)
       - subject (nsubj/nsubjpass head)
       - verb    (ROOT verb, falling back to any VERB/AUX)
       - other   (mean of remaining content positions)
  2. At each layer × token role, train a linear probe using only
     the hidden state at that position.
  3. Build a (13 layers × 4 roles) accuracy grid per phenomenon
     and plot a heatmap.

Expected: early layers encode grammar at the verb/critical token;
middle layers shift the signal to the subject; late layers spread
it to [CLS].
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from transformers import BertTokenizerFast

from src.probes.linear_probe import LinearProbe
from src.utils.visualization import plot_token_position_grid

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "exp5_token_position"
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "exp5"

NUM_LAYERS = 13
TOKEN_ROLES = ["cls", "subject", "verb", "other"]


def load_spacy():
    """Load spaCy en_core_web_sm, downloading it on first use if needed."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")


def find_role_indices(
    nlp,
    tokenizer_fast: BertTokenizerFast,
    sentence: str,
    expected_seq_len: int,
) -> dict | None:
    """
    Identify BERT WordPiece token indices for cls / subject / verb / other.

    Returns a dict {role: index_or_list_of_indices}, or None if subject
    or verb cannot be located. The 'other' entry is a list of indices
    (mean-pooled later).

    `expected_seq_len` is the seq_len of the saved hidden states for this
    sentence; if our re-tokenization disagrees we return None so the pair
    can be skipped.
    """
    doc = nlp(sentence)

    subj_char = None
    verb_char = None
    for token in doc:
        if subj_char is None and token.dep_ in ("nsubj", "nsubjpass"):
            subj_char = (token.idx, token.idx + len(token.text))
        if verb_char is None and token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
            verb_char = (token.idx, token.idx + len(token.text))
    if verb_char is None:
        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                verb_char = (token.idx, token.idx + len(token.text))
                break
    if subj_char is None or verb_char is None:
        return None

    enc = tokenizer_fast(
        sentence,
        truncation=True,
        max_length=128,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    seq_len = len(offsets)
    if seq_len != expected_seq_len:
        return None

    def first_overlapping(char_span):
        cs, ce = char_span
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue  # CLS / SEP / pad
            if not (e <= cs or s >= ce):
                return i
        return None

    subj_idx = first_overlapping(subj_char)
    verb_idx = first_overlapping(verb_char)
    if subj_idx is None or verb_idx is None:
        return None

    # 'other' = all real word tokens that aren't subject or verb
    special_or_skip = {0, subj_idx, verb_idx}
    other = [
        i for i in range(seq_len)
        if i not in special_or_skip and not (offsets[i][0] == 0 and offsets[i][1] == 0)
    ]
    if not other:
        return None

    return {"cls": 0, "subject": subj_idx, "verb": verb_idx, "other": other}


def vec_at_role(hidden_layer: np.ndarray, roles: dict, role: str) -> np.ndarray | None:
    """
    Extract a single vector from a layer's hidden state at a token role.

    For 'other', mean-pools across all listed indices.
    Returns None if the role's index is out of bounds.
    """
    seq_len = hidden_layer.shape[0]
    if role == "other":
        idxs = [i for i in roles["other"] if i < seq_len]
        if not idxs:
            return None
        return hidden_layer[idxs].mean(axis=0)

    idx = roles[role]
    if idx >= seq_len:
        return None
    return hidden_layer[idx]


def build_role_data_for_phenomenon(
    hidden_states_dir: Path,
    entries: list[dict],
    nlp,
    tokenizer_fast: BertTokenizerFast,
) -> tuple[dict, int]:
    """
    Single pass over a phenomenon's pairs. For each (layer, role),
    accumulates (X, y) lists. Skipped pairs are counted.

    Returns:
        (data dict keyed by (layer, role) -> {"X": list, "y": list},
         number of pairs skipped)
    """
    data = {(l, r): {"X": [], "y": []} for l in range(NUM_LAYERS) for r in TOKEN_ROLES}
    n_skipped = 0

    for entry in tqdm(entries, desc="  Extracting role-position vectors"):
        npz_path = hidden_states_dir / f"pair_{entry['idx']:04d}.npz"
        if not npz_path.exists():
            n_skipped += 1
            continue

        npz = np.load(npz_path)
        good_h = npz["good_hidden"]   # (13, sl_good, 768)
        bad_h = npz["bad_hidden"]     # (13, sl_bad, 768)

        good_roles = find_role_indices(nlp, tokenizer_fast, entry["good"], good_h.shape[1])
        bad_roles = find_role_indices(nlp, tokenizer_fast, entry["bad"], bad_h.shape[1])
        if good_roles is None or bad_roles is None:
            n_skipped += 1
            continue

        for layer in range(NUM_LAYERS):
            for role in TOKEN_ROLES:
                gv = vec_at_role(good_h[layer], good_roles, role)
                bv = vec_at_role(bad_h[layer], bad_roles, role)
                if gv is None or bv is None:
                    continue
                data[(layer, role)]["X"].append(gv)
                data[(layer, role)]["y"].append(1)
                data[(layer, role)]["X"].append(bv)
                data[(layer, role)]["y"].append(0)

    return data, n_skipped


def run_token_position_experiment(
    hidden_states_dir: Path = None,
    metadata_path: Path = None,
):
    """
    Run the layer × token-position probing grid for every phenomenon.

    Saves a (13, 4) accuracy grid per phenomenon plus a heatmap figure.
    """
    if hidden_states_dir is None:
        hidden_states_dir = PROCESSED_DIR / "hidden_states"
    if metadata_path is None:
        metadata_path = hidden_states_dir / "metadata.json"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        metadata = json.load(f)

    print("Loading spaCy en_core_web_sm...")
    nlp = load_spacy()
    print("Loading BertTokenizerFast for offset mapping...")
    tokenizer_fast = BertTokenizerFast.from_pretrained("bert-base-uncased")

    phenomena: dict[str, list[dict]] = {}
    for entry in metadata:
        phenomena.setdefault(entry["phenomenon"], []).append(entry)

    probe = LinearProbe(n_folds=5, max_iter=1000)
    all_grids: dict[str, np.ndarray] = {}
    all_skipped: dict[str, int] = {}

    for phenom, entries in sorted(phenomena.items()):
        print(f"\n{'='*60}")
        print(f"Phenomenon: {phenom} ({len(entries)} pairs)")
        print(f"{'='*60}")

        data, n_skipped = build_role_data_for_phenomenon(
            hidden_states_dir, entries, nlp, tokenizer_fast,
        )
        all_skipped[phenom] = n_skipped
        if n_skipped:
            print(f"  Skipped {n_skipped}/{len(entries)} pairs (parse failure or seq mismatch)")

        grid = np.zeros((NUM_LAYERS, len(TOKEN_ROLES)))
        for layer in range(NUM_LAYERS):
            row_strs = []
            for ri, role in enumerate(TOKEN_ROLES):
                X = np.array(data[(layer, role)]["X"])
                y = np.array(data[(layer, role)]["y"])
                if len(y) < 10 or len(set(y)) < 2:
                    grid[layer, ri] = float("nan")
                    row_strs.append(f"{role}=  n/a")
                    continue
                result = probe.train_and_evaluate(X, y, layer)
                grid[layer, ri] = result.accuracy
                row_strs.append(f"{role}={result.accuracy:.3f}")
            print(f"  Layer {layer:2d}: " + "  ".join(row_strs))

        all_grids[phenom] = grid

    # ── Save results ──
    results_data = {
        "token_roles": TOKEN_ROLES,
        "num_layers": NUM_LAYERS,
        "skipped_per_phenomenon": all_skipped,
        "grids": {p: g.tolist() for p, g in all_grids.items()},
    }
    out_path = RESULTS_DIR / "grid_results.json"
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Plots ──
    for phenom, grid in all_grids.items():
        plot_token_position_grid(
            grid,
            token_labels=TOKEN_ROLES,
            save_path=FIGURES_DIR / f"position_grid_{phenom}.png",
        )

    # ── Cross-phenomenon summary ──
    print(f"\n{'='*80}")
    print("PEAK (LAYER, ROLE) PER PHENOMENON")
    print(f"{'='*80}")
    for phenom, grid in sorted(all_grids.items()):
        # Ignore NaN cells when finding the peak
        flat = np.nan_to_num(grid, nan=-1.0)
        peak_layer, peak_role_idx = np.unravel_index(np.argmax(flat), flat.shape)
        print(
            f"  {phenom:<12} peak={grid[peak_layer, peak_role_idx]:.3f} "
            f"at layer={peak_layer}, role={TOKEN_ROLES[peak_role_idx]}"
        )

    return results_data


if __name__ == "__main__":
    run_token_position_experiment()
