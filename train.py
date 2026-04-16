from __future__ import annotations

import argparse

from quest.config import load_config
from quest.pipeline import train_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Google QUEST dual transformer ensemble.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the dataset directory. Takes precedence over QUEST_DATA_DIR and config data_dir.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a minimal debug loop with fewer folds, epochs, and batches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.debug:
        config.debug = True
    summary = train_pipeline(config, data_dir=args.data_dir)
    print(summary)


if __name__ == "__main__":
    main()
