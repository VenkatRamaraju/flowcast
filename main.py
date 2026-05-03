#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Entrypoint to train a model
"""

# Imports
import sys
from train.data.load import load, read_from_s3, upload_to_s3
from train.data.transform import transform

# Main
def main():
    # Full catalogue
    entries = load()
    # Optional indices
    argv = sys.argv[1:]
    if argv:
        # CLI index range
        start = int(argv[0])
        stop = int(argv[1]) if len(argv) > 1 else None
        entries = entries[start:stop]

    for entry in entries:
        # Download source
        print(f"Downloading {entry[0]}")
        data = read_from_s3(entry)

        # Normalize frame
        transformed_data = transform(data)

        # Upload result
        upload_to_s3(transformed_data, entry)


if __name__ == "__main__":
    main()
