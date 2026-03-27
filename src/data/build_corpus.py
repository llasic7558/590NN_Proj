"""
Week 1-2: Corpus Construction

Builds ~2,000 minimal-pair sentences differing by a single grammatical feature.
Three phenomena:
  1. Subject-verb agreement (singular/plural mismatch)
  2. Negation (presence/absence and scope)
  3. Dependent clauses (relative clauses, embedded subjects)

Each pair: (grammatical sentence, ungrammatical sentence, phenomenon, metadata).

Sources:
  - BLiMP (Warstadt et al. 2020) subsets for agreement and negation
  - Hand-crafted templates for dependent clauses
  - CoLA (Warstadt et al. 2019) filtered examples as supplementary data
"""

import json
import os
import random
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

from datasets import load_dataset

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class MinimalPair:
    """A single minimal pair with metadata."""
    good: str                   # grammatical sentence
    bad: str                    # ungrammatical sentence
    phenomenon: str             # agreement | negation | clause
    sub_phenomenon: str         # e.g. "simple_agreement", "across_rel_clause"
    critical_token_idx: Optional[int] = None  # index of the token that differs
    source: str = "blimp"       # blimp | template | cola


# ---------------------------------------------------------------------------
# 1. BLiMP-based pairs (agreement & negation)
# ---------------------------------------------------------------------------

# BLiMP paradigms we pull from
BLIMP_AGREEMENT_PARADIGMS = [
    "anaphor_number_agreement",
    "distractor_agreement_relational_noun",
    "distractor_agreement_relative_clause",
    "irregular_plural_subject_verb_agreement_1",
    "irregular_plural_subject_verb_agreement_2",
    "regular_plural_subject_verb_agreement_1",
    "regular_plural_subject_verb_agreement_2",
]

BLIMP_NEGATION_PARADIGMS = [
    "sentential_negation_npi_licensor_present",
    "sentential_negation_npi_scope",
]

SUB_PHENOM_MAP = {
    "anaphor_number_agreement": "anaphor_agreement",
    "distractor_agreement_relational_noun": "across_relational_noun",
    "distractor_agreement_relative_clause": "across_rel_clause",
    "irregular_plural_subject_verb_agreement_1": "irregular_plural_1",
    "irregular_plural_subject_verb_agreement_2": "irregular_plural_2",
    "regular_plural_subject_verb_agreement_1": "regular_plural_1",
    "regular_plural_subject_verb_agreement_2": "regular_plural_2",
    "sentential_negation_npi_licensor_present": "negation_npi_licensor",
    "sentential_negation_npi_scope": "negation_npi_scope",
}


def load_blimp_pairs(paradigm: str, phenomenon: str, max_pairs: int = 200) -> list[MinimalPair]:
    """Load minimal pairs from a single BLiMP paradigm via HuggingFace."""
    ds = load_dataset("nyu-mll/blimp", paradigm, split="train")
    pairs = []
    for row in ds:
        if len(pairs) >= max_pairs:
            break
        pairs.append(MinimalPair(
            good=row["sentence_good"],
            bad=row["sentence_bad"],
            phenomenon=phenomenon,
            sub_phenomenon=SUB_PHENOM_MAP.get(paradigm, paradigm),
            source="blimp",
        ))
    return pairs


# ---------------------------------------------------------------------------
# 2. Template-based pairs (dependent clauses)
# ---------------------------------------------------------------------------

SUBJECTS_SG = ["the doctor", "the teacher", "the engineer", "the artist", "the manager",
               "the student", "the chef", "the scientist", "the lawyer", "the nurse"]
SUBJECTS_PL = ["the doctors", "the teachers", "the engineers", "the artists", "the managers",
               "the students", "the chefs", "the scientists", "the lawyers", "the nurses"]
VERBS_SG = ["runs", "walks", "writes", "reads", "speaks", "works", "plays", "sings", "drives", "cooks"]
VERBS_PL = ["run", "walk", "write", "read", "speak", "work", "play", "sing", "drive", "cook"]
RC_OBJECTS = ["the report", "the letter", "the proposal", "the article", "the book"]
ADVERBS = ["quickly", "carefully", "quietly", "eagerly", "slowly"]

CLAUSE_TEMPLATES = [
    # Relative clause intervening between subject and verb
    {
        "good": "{subj_sg} who {verb_pl} {obj} {verb_sg} {adv}.",
        "bad":  "{subj_sg} who {verb_pl} {obj} {verb_pl_main} {adv}.",
        "sub_phenomenon": "rc_subject_verb",
    },
    # Embedded subject in sentential complement
    {
        "good": "{subj_sg} thinks that {subj_pl} {verb_pl} {obj}.",
        "bad":  "{subj_sg} thinks that {subj_pl} {verb_sg_emb} {obj}.",
        "sub_phenomenon": "embedded_subject",
    },
]


def generate_clause_pairs(n: int = 400) -> list[MinimalPair]:
    """Generate minimal pairs for dependent-clause phenomena from templates."""
    pairs = []
    random.seed(42)
    while len(pairs) < n:
        tmpl = random.choice(CLAUSE_TEMPLATES)
        i = random.randrange(len(SUBJECTS_SG))
        j = random.randrange(len(VERBS_SG))
        k = random.randrange(len(VERBS_SG))
        obj = random.choice(RC_OBJECTS)
        adv = random.choice(ADVERBS)
        subj_sg = SUBJECTS_SG[i]
        subj_pl = SUBJECTS_PL[i]
        verb_sg = VERBS_SG[j]
        verb_pl = VERBS_PL[j]
        verb_sg_emb = VERBS_SG[k]
        verb_pl_main = VERBS_PL[k]

        good = tmpl["good"].format(
            subj_sg=subj_sg, subj_pl=subj_pl,
            verb_sg=verb_sg, verb_pl=verb_pl,
            verb_sg_emb=verb_sg_emb, verb_pl_main=verb_pl_main,
            obj=obj, adv=adv,
        )
        bad = tmpl["bad"].format(
            subj_sg=subj_sg, subj_pl=subj_pl,
            verb_sg=verb_sg, verb_pl=verb_pl,
            verb_sg_emb=verb_sg_emb, verb_pl_main=verb_pl_main,
            obj=obj, adv=adv,
        )
        if good != bad:
            pairs.append(MinimalPair(
                good=good, bad=bad,
                phenomenon="clause",
                sub_phenomenon=tmpl["sub_phenomenon"],
                source="template",
            ))
    return pairs


# ---------------------------------------------------------------------------
# 3. Assemble full corpus
# ---------------------------------------------------------------------------

def build_corpus(target_total: int = 2000) -> list[MinimalPair]:
    """
    Build the full minimal-pair corpus.

    Target distribution (~2000 pairs):
      - Agreement: ~800 pairs from BLiMP
      - Negation:  ~400 pairs from BLiMP
      - Clauses:   ~400 pairs from templates
      - Remainder filled by sampling more from existing sources
    """
    print("=== Building Minimal-Pair Corpus ===")
    all_pairs: list[MinimalPair] = []

    # Agreement from BLiMP
    print("\n[1/3] Loading agreement pairs from BLiMP...")
    per_paradigm = 120  # 7 paradigms × 120 ≈ 840
    for paradigm in BLIMP_AGREEMENT_PARADIGMS:
        pairs = load_blimp_pairs(paradigm, "agreement", max_pairs=per_paradigm)
        print(f"  {paradigm}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    # Negation from BLiMP
    print("\n[2/3] Loading negation pairs from BLiMP...")
    for paradigm in BLIMP_NEGATION_PARADIGMS:
        pairs = load_blimp_pairs(paradigm, "negation", max_pairs=200)
        print(f"  {paradigm}: {len(pairs)} pairs")
        all_pairs.extend(pairs)

    # Dependent clauses from templates
    print("\n[3/3] Generating dependent-clause pairs from templates...")
    clause_pairs = generate_clause_pairs(n=400)
    print(f"  Generated {len(clause_pairs)} clause pairs")
    all_pairs.extend(clause_pairs)

    # Trim or pad to target
    random.seed(42)
    if len(all_pairs) > target_total:
        random.shuffle(all_pairs)
        all_pairs = all_pairs[:target_total]

    # Print summary
    from collections import Counter
    counts = Counter(p.phenomenon for p in all_pairs)
    print(f"\n=== Corpus Summary ===")
    print(f"Total pairs: {len(all_pairs)}")
    for phenom, count in sorted(counts.items()):
        print(f"  {phenom}: {count}")

    return all_pairs


def save_corpus(pairs: list[MinimalPair], filename: str = "minimal_pairs.json"):
    """Save corpus to JSON."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    data = [asdict(p) for p in pairs]
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data)} pairs to {out_path}")
    return out_path


def load_corpus(filename: str = "minimal_pairs.json") -> list[MinimalPair]:
    """Load corpus from JSON."""
    path = PROCESSED_DIR / filename
    with open(path) as f:
        data = json.load(f)
    return [MinimalPair(**d) for d in data]


if __name__ == "__main__":
    pairs = build_corpus()
    save_corpus(pairs)
