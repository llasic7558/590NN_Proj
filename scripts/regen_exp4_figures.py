"""
Regenerate Exp 4 figures locally from the embedded outputs of
notebooks/run_pipeline.ipynb, without re-running the GPU pipeline.

The Colab run already produced per-layer probing accuracies and per-
phenomenon adjacent-CKA values; those numbers are inside cell 39's stream
output. This script parses them, writes the JSON files that
`src/experiments/exp4_finetuning.py` expects under
`results/exp4_finetuning/`, then calls the plotting routines to emit
PNGs under `figures/exp4/`.

Run from the project root:

    python3 scripts/regen_exp4_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOK = PROJECT_ROOT / "notebooks" / "run_pipeline.ipynb"
EXP4_CELL_INDEX = 39
RESULTS_DIR = PROJECT_ROOT / "results" / "exp4_finetuning"
FIGURES_DIR = PROJECT_ROOT / "figures" / "exp4"

MODEL_LABELS = ["base", "cola", "sst2"]


def _load_stream_text(notebook_path: Path, cell_index: int) -> str:
    nb = json.loads(notebook_path.read_text())
    cell = nb["cells"][cell_index]
    return "".join(
        "".join(o.get("text", []))
        for o in cell.get("outputs", [])
        if o.get("output_type") == "stream"
    )


_MODEL_RE = re.compile(r"^\[(base|cola|sst2)\]\s+Running\s+(probing|CKA)")
_PHENOM_RE = re.compile(r"^Phenomenon:\s+(\w+)\s+\(\d+ pairs\)")
_LAYER_RE = re.compile(
    r"^\s*Layer\s+(\d+):\s+acc=([\d.]+)\s+±\s+([\d.]+)\s+F1=([\d.]+)\s+sel=(-?[\d.]+)"
)
_CKA_HEADER_RE = re.compile(r"^\s*Adjacent-layer CKA:")
_FULL_CKA_HEADER_RE = re.compile(r"^\s*Full corpus adjacent-layer CKA:")
_CKA_ROW_RE = re.compile(r"^\s*(\d+)\s+→\s+(\d+)\s+([\d.]+)\s+([\d.]+)")


def parse_run(text: str) -> dict:
    """
    Walk the stream text once, returning:

        {
          "base":  {"probing": {phenom: [{"layer": int, "accuracy": float, ...}]},
                    "cka":     {"full":  [12 floats],
                                "per_phenomenon": {phenom: [12 floats]}}},
          "cola":  {...},
          "sst2":  {...},
        }
    """
    out = {m: {"probing": {}, "cka": {"full": None, "per_phenomenon": {}}} for m in MODEL_LABELS}

    model = None
    section = None  # "probing" | "cka"
    phenom = None
    cka_target = None  # "full" | "phenom"

    for raw_line in text.splitlines():
        m = _MODEL_RE.match(raw_line)
        if m:
            model, sec = m.group(1), m.group(2).lower()
            section = "probing" if sec == "probing" else "cka"
            phenom = None
            cka_target = None
            continue

        if model is None:
            continue

        if _PHENOM_RE.match(raw_line):
            phenom = _PHENOM_RE.match(raw_line).group(1)
            cka_target = None
            continue

        if section == "probing":
            lm = _LAYER_RE.match(raw_line)
            if lm and phenom is not None:
                layer = int(lm.group(1))
                acc = float(lm.group(2))
                acc_std = float(lm.group(3))
                f1 = float(lm.group(4))
                sel = float(lm.group(5))
                bucket = out[model]["probing"].setdefault(phenom, [])
                # The summary block at the end repeats Layer 0..12 in a different
                # format that this regex does NOT match, so deduping is normally
                # not needed; still, guard against duplicates.
                if not any(r["layer"] == layer for r in bucket):
                    bucket.append({
                        "layer": layer,
                        "accuracy": acc,
                        "accuracy_std": acc_std,
                        "f1": f1,
                        "f1_std": 0.0,
                        "control_accuracy": acc - sel,
                        "selectivity": sel,
                        "n_samples": 0,
                    })
            continue

        if section == "cka":
            if _FULL_CKA_HEADER_RE.match(raw_line):
                cka_target = "full"
                out[model]["cka"]["full"] = []
                continue
            if _CKA_HEADER_RE.match(raw_line):
                cka_target = "phenom"
                if phenom is not None:
                    out[model]["cka"]["per_phenomenon"][phenom] = []
                continue
            cm = _CKA_ROW_RE.match(raw_line)
            if cm and cka_target is not None:
                cka_val = float(cm.group(3))
                if cka_target == "full":
                    out[model]["cka"]["full"].append(cka_val)
                elif cka_target == "phenom" and phenom is not None:
                    out[model]["cka"]["per_phenomenon"][phenom].append(cka_val)

    return out


def _write_jsons(parsed: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODEL_LABELS:
        probing = parsed[model]["probing"]
        for phenom, layers in probing.items():
            layers.sort(key=lambda r: r["layer"])
        (RESULTS_DIR / f"{model}_probing.json").write_text(json.dumps(probing, indent=2))

        cka = {
            "full_corpus": {"adjacent_cka": parsed[model]["cka"]["full"] or []},
            "per_phenomenon": {
                phenom: {"adjacent_cka": vals}
                for phenom, vals in parsed[model]["cka"]["per_phenomenon"].items()
            },
        }
        (RESULTS_DIR / f"{model}_cka.json").write_text(json.dumps(cka, indent=2))
        print(f"  wrote {model}_probing.json and {model}_cka.json")


def _validate(parsed: dict) -> None:
    for model in MODEL_LABELS:
        probing = parsed[model]["probing"]
        if not probing:
            raise RuntimeError(f"No probing data parsed for {model!r}")
        for phenom, layers in probing.items():
            if len(layers) != 13:
                raise RuntimeError(
                    f"{model}/{phenom} probing has {len(layers)} layers (expected 13)"
                )
        per_phen = parsed[model]["cka"]["per_phenomenon"]
        if not per_phen:
            raise RuntimeError(f"No per-phenomenon CKA parsed for {model!r}")
        for phenom, vals in per_phen.items():
            if len(vals) != 12:
                raise RuntimeError(
                    f"{model}/{phenom} CKA has {len(vals)} transitions (expected 12)"
                )


def main() -> None:
    if not NOTEBOOK.exists():
        raise FileNotFoundError(f"Notebook not found at {NOTEBOOK}")
    print(f"Parsing {NOTEBOOK}...")
    text = _load_stream_text(NOTEBOOK, EXP4_CELL_INDEX)
    parsed = parse_run(text)
    _validate(parsed)
    print(f"Writing JSONs to {RESULTS_DIR}...")
    _write_jsons(parsed)

    print("Generating figures...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from src.experiments.exp4_finetuning import compare_finetuned_models
    compare_finetuned_models()


if __name__ == "__main__":
    main()
