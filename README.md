# Probing the Grammar Machine

A layer-wise mechanistic analysis of grammatical encoding in BERT.

**Authors:** Omar Osman, Luka Lasic — UMass Amherst, CS 590NN

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### If you're using Windows PowerShell

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```


## Project Structure

```
src/
├── data/           # Corpus construction and data loading
├── extraction/     # Hidden state extraction from BERT
├── experiments/    # One module per experiment (Exp 1-5)
├── utils/          # Shared utilities (CKA, visualization, metrics)
├── probes/         # Probing classifiers
configs/            # Experiment configuration files
notebooks/          # Jupyter notebooks for exploration
```

## Running Experiments

```bash
# Step 1: Build corpus
python -m src.data.build_corpus

# Step 2: Extract hidden states
python -m src.extraction.extract_states

# Step 3: Run experiments
python -m src.experiments.exp1_probing
python -m src.experiments.exp2_cka
python -m src.experiments.exp3_ablation
python -m src.experiments.exp4_finetuning
python -m src.experiments.exp5_token_position
```

## Milestones

| Weeks | Milestone |
|-------|-----------|
| 1–2   | Corpus construction; pipeline setup; Jawahar et al. replication |
| 3–4   | Exp 1 & 2: Probing suite; CKA preserve-or-rewrite |
| 5–6   | Exp 3 & 4: Attention ablation; spatial probing grid |
| 6–7   | Exp 5: Fine-tuning comparison |
| 7–8   | Analysis, figures, write-up |
