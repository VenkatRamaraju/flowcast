#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Print two conflicting training rows
"""

# Imports
from pathlib import Path
import sys
import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.model.data import MIXED_BUCKET, list_csv_keys

def print_two_conflicting_rows(bucket=MIXED_BUCKET, round_decimals=1):
    load_dotenv(repo_root / ".env")
    client = boto3.client("s3")
    features = [
        "day_of_week",
        "time_bucket",
        "is_weekend",
        "month",
        "is_us_federal_holiday",
        "commute_hours",
        "station_id",
        "temperature",
        "precipitation",
        "wind",
    ]
    target = "net_flow"
    frames = []
    for key in list_csv_keys(bucket):
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
        df = pd.read_csv(body)
        missing = [column for column in features + [target] if column not in df.columns]
        if missing:
            raise ValueError(f"{key} missing columns: {missing}")
        frames.append(df)
        print(f"Loaded {key} | rows: {len(df):,}")
    if not frames:
        raise ValueError("No training files found")

    raw = pd.concat(frames, ignore_index=True)
    work = raw[features + [target]].copy()
    numeric_features = work[features].select_dtypes(include=[np.number]).columns
    for column in numeric_features:
        work[column] = work[column].round(round_decimals)

    grouped = (
        work.groupby(features, dropna=False)[target]
        .agg(["size", "nunique", "min", "max"])
        .reset_index()
    )
    conflicts = grouped[grouped["nunique"] > 1].sort_values(
        ["size", "max", "min"],
        ascending=[False, False, True],
    )
    if conflicts.empty:
        print("No conflicting rows found")
        return pd.DataFrame()

    display_columns = [column for column in ["period_start", "period_end"] + features + [target] if column in raw.columns]
    all_pairs = []
    print()
    print("A couple conflicting row pairs")
    for i, (_, conflict) in enumerate(conflicts.head(2).iterrows(), start=1):
        mask = pd.Series(True, index=work.index)
        for column in features:
            value = conflict[column]
            if pd.isna(value):
                mask &= work[column].isna()
            else:
                mask &= work[column] == value

        matching = raw.loc[mask].copy().sort_values(target)
        first_row = matching.iloc[[0]]
        second_row = matching[matching[target] != first_row.iloc[0][target]].iloc[[0]]
        pair = pd.concat([first_row, second_row], ignore_index=True)
        all_pairs.append(pair[display_columns])
        print()
        print(f"Conflict pair {i}")
        print(pair[display_columns].to_string(index=False))

    return pd.concat(all_pairs, ignore_index=True)

if __name__ == "__main__":
    print_two_conflicting_rows()