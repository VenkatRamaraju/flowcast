#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Stream CSV objects from S3 as pandas DataFrames
"""

# Imports
from pathlib import Path
import time
import boto3
import pandas as pd
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Constants
READ_BUCKET = "lyft-training-data-enhanced"
MIXED_BUCKET = "lyft-training-data-mixed"
S3_CONFIG = Config(
    connect_timeout=10,
    read_timeout=180,
    retries={"max_attempts": 10, "mode": "standard"},
)
CSV_READ_ATTEMPTS = 4
CSV_READ_BACKOFF_SECONDS = 3


def make_s3_client():
    return boto3.client("s3", config=S3_CONFIG)


def read_csv_object(client, bucket, key):
    for attempt in range(1, CSV_READ_ATTEMPTS + 1):
        try:
            body = client.get_object(Bucket=bucket, Key=key)["Body"]
            return pd.read_csv(body)
        except (BotoCoreError, ClientError, TimeoutError) as exc:
            if attempt == CSV_READ_ATTEMPTS:
                raise
            wait_seconds = CSV_READ_BACKOFF_SECONDS * attempt
            print(
                f"Retrying {key} after read failure "
                f"({attempt}/{CSV_READ_ATTEMPTS}): {exc}"
            )
            time.sleep(wait_seconds)


def list_csv_keys(bucket):
    client = make_s3_client()
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


def tripdata_key_yyyymm(key):
    basename = key.rsplit("/", 1)[-1]
    if len(basename) >= 6 and basename[:6].isdigit():
        return int(basename[:6])
    return None


def tripdata_key_is_from_jan_2022_or_later(key):
    yyyymm = tripdata_key_yyyymm(key)
    return yyyymm is not None and yyyymm >= 202201


def iter_csv_dataframes(bucket):
    client = make_s3_client()
    keys = list_csv_keys(bucket)
    for key in sorted(keys):
        yield read_csv_object(client, bucket, key)


def iter_csv_files(bucket):
    client = make_s3_client()
    keys = list_csv_keys(bucket)
    for key in keys:
        yield key, read_csv_object(client, bucket, key)


def iter_csv_files_for_keys(bucket, keys):
    client = make_s3_client()
    for key in keys:
        yield key, read_csv_object(client, bucket, key)


def count_total_rows(bucket):
    total_rows = 0
    for dataframe in iter_csv_dataframes(bucket):
        total_rows += len(dataframe.index)
    return total_rows


def mix_csv_files(
    output_files=100,
    random_state=42,
):
    client = make_s3_client()
    all_keys = list_csv_keys(READ_BUCKET)
    keys = [
        k
        for k in all_keys
        if "baywheels" in k.lower() and tripdata_key_is_from_jan_2022_or_later(k)
    ]
    if (n_skip := len(all_keys) - len(keys)) > 0:
        print(f"Skipping {n_skip} CSV file(s) (not Bay Wheels name or before 2022-01)")
    frames = []
    for key in keys:
        print(f"Loading {key}")
        frames.append(read_csv_object(client, READ_BUCKET, key))

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


if __name__ == "__main__":
    mix_csv_files()
    count = count_total_rows(MIXED_BUCKET)
    print(f"Total rows: {count:,}")
