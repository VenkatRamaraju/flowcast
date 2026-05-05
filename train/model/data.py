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
READ_BUCKET = "lyft-training-data-enhanced"
MIXED_BUCKET = "lyft-training-data-mixed"


def list_csv_keys(bucket):
    client = boto3.client("s3")
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket)
    keys = []
    for page in pages:
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/") or not key.lower().endswith(".csv"):
                continue
            keys.append(key)
    return sorted(keys)


def iter_csv_dataframes(bucket):
    client = boto3.client("s3")
    keys = list_csv_keys(bucket)
    for key in sorted(keys):
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        yield pd.read_csv(body)


def iter_csv_files(bucket):
    client = boto3.client("s3")
    keys = list_csv_keys(bucket)
    for key in keys:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        yield key, pd.read_csv(body)


def get_eval_file(bucket):
    client = boto3.client("s3")
    keys = list_csv_keys(bucket)
    key = keys[-1]
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    return key, pd.read_csv(body)


def count_total_rows(bucket):
    total_rows = 0
    for dataframe in iter_csv_dataframes(bucket):
        total_rows += len(dataframe.index)
    return total_rows


def mix_csv_files(
    output_files=100,
    random_state=42,
):
    client = boto3.client("s3")
    keys = list_csv_keys(READ_BUCKET)
    frames = []
    for key in keys:
        print(f"Loading {key}")
        body = client.get_object(Bucket=READ_BUCKET, Key=key)["Body"]
        frames.append(pd.read_csv(body))

    if not frames:
        print("No CSV files found")
        return

    data = pd.concat(frames, ignore_index=True)
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    total_rows = len(data)
    base_rows = total_rows // output_files
    extra_rows = total_rows % output_files
    start = 0

    for i in range(output_files):
        rows_this_file = base_rows + (1 if i < extra_rows else 0)
        stop = start + rows_this_file
        part = data.iloc[start:stop]
        output_key = f"part-{i:03d}.csv"
        print(f"Uploading {output_key} with {len(part):,} rows")
        client.put_object(
            Bucket=MIXED_BUCKET,
            Key=output_key,
            Body=part.to_csv(index=False).encode("utf-8"),
        )
        start = stop
