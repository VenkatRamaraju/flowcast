#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Entrypoint to train a model
"""

# Imports
import argparse
from train.data.load import load, read_from_s3, upload_to_s3
from train.data.transform import transform
from train.model.model import train
from train.model.eval import eval


def main():
    parser = argparse.ArgumentParser(description="Flowcast ETL and training entrypoint")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Run incremental XGBoost training")
    mode.add_argument("--eval", action="store_true", help="Evaluate model on held-out eval file")
    mode.add_argument(
        "--data",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="Run ETL on catalogue slice entries[START:STOP]",
    )
    args = parser.parse_args()

    if args.train:
        train()
        return

    if args.eval:
        eval()
        return

    start, stop = args.data
    entries = load()[start:stop]
    for entry in entries:
        print(f"Downloading {entry[0]}")
        data = read_from_s3(entry)
        transformed_data = transform(data)
        upload_to_s3(transformed_data, entry)


if __name__ == "__main__":
    main()
