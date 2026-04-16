# Linux Training Notes

## Folder Layout

Keep code and data in two separate folders:

```text
/workspace/Google-Quest-Challenge
/data/google-quest-challenge
```

On Windows the matching local layout is:

```text
F:\闈㈠challenge\google-quest-challenge
F:\闈㈠challenge\google-quest-data\google-quest-challenge
```

## Install

```bash
cd /workspace/Google-Quest-Challenge
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Smoke Run

```bash
cd /workspace/Google-Quest-Challenge
source .venv/bin/activate
python train.py \
  --config configs/deberta_dual_hierarchical_smoke.yaml \
  --data-dir /data/google-quest-challenge

python predict.py \
  --config configs/deberta_dual_hierarchical_smoke.yaml \
  --data-dir /data/google-quest-challenge \
  --checkpoint-dir artifacts_deberta_smoke/checkpoints \
  --output submission_smoke.csv
```

## Full Training

```bash
cd /workspace/Google-Quest-Challenge
source .venv/bin/activate
python train.py \
  --config configs/deberta_dual_hierarchical_linux.yaml \
  --data-dir /data/google-quest-challenge
```

## Full Inference

```bash
cd /workspace/Google-Quest-Challenge
source .venv/bin/activate
python predict.py \
  --config configs/deberta_dual_hierarchical_linux.yaml \
  --data-dir /data/google-quest-challenge \
  --checkpoint-dir artifacts_deberta/checkpoints \
  --output submission.csv
```

## Notes

- `QUEST_DATA_DIR` can replace `--data-dir` if you prefer environment-based configuration.
- `train.py` reuses completed fold checkpoints when the expected checkpoint file already exists.
- `predict.py` applies rank-based distribution matching by default because the config enables `distribution_matching: true`.
