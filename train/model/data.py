#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Stream CSV objects from S3 as pandas DataFrames
"""

# Imports
from pathlib import Path
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Constants
DEFAULT_BUCKET = "lyft-training-data"


def list_csv_keys(bucket=DEFAULT_BUCKET, prefix=""):
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    keys = []
    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/") or not key.lower().endswith(".csv"):
                continue
            keys.append(key)
    return sorted(keys)


def iter_csv_dataframes(bucket=DEFAULT_BUCKET, prefix=""):
    client = boto3.client("s3")
    keys = list_csv_keys(bucket=bucket, prefix=prefix)
    for key in sorted(keys):
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        yield pd.read_csv(body)


def iter_csv_files(bucket=DEFAULT_BUCKET, prefix=""):
    client = boto3.client("s3")
    keys = list_csv_keys(bucket=bucket, prefix=prefix)
    for key in keys:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        yield key, pd.read_csv(body)


def count_total_rows(bucket=DEFAULT_BUCKET, prefix=""):
    total_rows = 0
    for dataframe in iter_csv_dataframes(bucket=bucket, prefix=prefix):
        total_rows += len(dataframe.index)
    return total_rows


if __name__ == "__main__":
    print(count_total_rows())
