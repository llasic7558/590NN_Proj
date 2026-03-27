"""
Week 1-2: Hidden State Extraction Pipeline

Feeds minimal-pair sentences through BERT and records:
  - Hidden states at all 13 layers (embedding + 12 transformer layers)
  - Attention weights from all 12 layers × 12 heads
  - Token-level alignments

Outputs are saved per-phenomenon for downstream experiments.
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"


def load_model(model_name: str = "bert-base-uncased"):
    """Load BERT model and tokenizer, set to eval mode."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Loaded {model_name} on {device}")
    return model, tokenizer, device


@torch.no_grad()
def extract_single(model, tokenizer, sentence: str, device: torch.device) -> dict:
    """
    Extract hidden states and attentions for a single sentence.

    Returns:
        dict with:
            - "tokens": list of token strings
            - "hidden_states": np.array of shape (13, seq_len, 768)
            - "attentions": np.array of shape (12, 12, seq_len, seq_len)
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # hidden_states: tuple of 13 tensors, each (1, seq_len, 768)
    hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (13, 1, seq_len, 768)
    hidden_states = hidden_states.squeeze(1).cpu().numpy()      # (13, seq_len, 768)

    # attentions: tuple of 12 tensors, each (1, 12, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions, dim=0)         # (12, 1, 12, seq_len, seq_len)
    attentions = attentions.squeeze(1).cpu().numpy()             # (12, 12, seq_len, seq_len)

    return {
        "tokens": tokens,
        "hidden_states": hidden_states,
        "attentions": attentions,
    }


def extract_corpus(corpus_path: str = None, output_dir: str = None, batch_log_every: int = 100):
    """
    Extract hidden states for the full minimal-pair corpus.

    Saves one .npz file per sentence pair containing:
        - good_hidden: (13, seq_len_good, 768)
        - bad_hidden:  (13, seq_len_bad, 768)
        - good_attn:   (12, 12, seq_len_good, seq_len_good)
        - bad_attn:    (12, 12, seq_len_bad, seq_len_bad)
    And a metadata JSON with tokens and labels.
    """
    if corpus_path is None:
        corpus_path = PROCESSED_DIR / "minimal_pairs.json"
    if output_dir is None:
        output_dir = PROCESSED_DIR / "hidden_states"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(corpus_path) as f:
        corpus = json.load(f)

    model, tokenizer, device = load_model()

    metadata_all = []

    for i, pair in enumerate(tqdm(corpus, desc="Extracting hidden states")):
        good_out = extract_single(model, tokenizer, pair["good"], device)
        bad_out = extract_single(model, tokenizer, pair["bad"], device)

        # Save hidden states as compressed numpy
        np.savez_compressed(
            output_dir / f"pair_{i:04d}.npz",
            good_hidden=good_out["hidden_states"],
            bad_hidden=bad_out["hidden_states"],
            good_attn=good_out["attentions"],
            bad_attn=bad_out["attentions"],
        )

        metadata_all.append({
            "idx": i,
            "good": pair["good"],
            "bad": pair["bad"],
            "good_tokens": good_out["tokens"],
            "bad_tokens": bad_out["tokens"],
            "phenomenon": pair["phenomenon"],
            "sub_phenomenon": pair["sub_phenomenon"],
        })

        if (i + 1) % batch_log_every == 0:
            print(f"  Processed {i + 1}/{len(corpus)} pairs")

    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata_all, f, indent=2)

    print(f"\nExtraction complete. {len(corpus)} pairs saved to {output_dir}")


if __name__ == "__main__":
    extract_corpus()
