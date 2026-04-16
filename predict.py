from __future__ import annotations

import argparse
from pathlib import Path

from quest.config import load_config
from quest.pipeline import predict_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a submission CSV for Google QUEST.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
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
    )
    print(summary)


if __name__ == "__main__":
    main()
