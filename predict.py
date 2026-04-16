from __future__ import annotations

import argparse
from pathlib import Path

from quest.config import load_config
from quest.pipeline import predict_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a submission CSV for Google QUEST.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the dataset directory. Takes precedence over QUEST_DATA_DIR and config data_dir.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override the local model directory. Takes precedence over QUEST_MODEL_DIR and config model_dir.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints",
        help="Directory containing trained fold checkpoints.",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = predict_pipeline(
        config,
        checkpoint_dir=Path(args.checkpoint_dir),
        output_path=Path(args.output),
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )
    print(summary)


if __name__ == "__main__":
    main()
