#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Evaluate the categorical XGBoost model against held-out files
"""

# Imports
import json
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train.model.data import MIXED_BUCKET
from train.model.model import FEATURES, HELD_OUT_KEYS_PATH, MODEL_PATH, STATION_CATEGORIES_PATH, TARGET, make_dmatrix, normalize_station_ids, read_csv, validate_columns


def read_json(path):
    with path.open() as file:
        return json.load(file)


def load_eval_data(client, bucket, keys):
    frames = []
    for key in keys:
        raw = read_csv(client, bucket, key)
        validate_columns(raw, key)
        frames.append(raw)
        print(f"Loaded eval file: {key} | rows: {len(raw):,}")
    df = pd.concat(frames, ignore_index=True)
    df["station_id"] = normalize_station_ids(df["station_id"])
    return df


def load_booster():
    if not MODEL_PATH.exists():
        print("No model found in artifacts")
        return None
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster


def eval(bucket):
    client = boto3.client("s3")
    held_out_keys = read_json(HELD_OUT_KEYS_PATH)
    station_categories = read_json(STATION_CATEGORIES_PATH)
    df = load_eval_data(client, bucket, held_out_keys)
    y = np.round(df[TARGET].values).astype(np.int64)

    y_min = int(np.min(y))
    y_max = int(np.max(y))
    n_random_options = y_max - y_min + 1

    rng = np.random.default_rng(42)
    random_preds = rng.integers(y_min, y_max + 1, size=len(y), endpoint=False)

    rand_correct = int(np.sum(random_preds == y))
    rand_accuracy = rand_correct / len(y)
    rand_mean_error = float(np.mean(random_preds - y))
    rand_rmse = float(np.sqrt(mean_squared_error(y, random_preds)))
    rand_mae = mean_absolute_error(y, random_preds)

    print(
        f"Held-out net_flow (int): min={y_min:,} max={y_max:,} "
        f"→ {n_random_options:,} choices for uniform random picks"
    )
    print("Random baseline (uniform integer per row, seed=42)")
    print(f"  Accuracy (exact): {rand_correct:,} / {len(y):,}  ({rand_accuracy:.2%})")
    print(f"  Mean error       : {rand_mean_error:.4f}")
    print(f"  RMSE             : {rand_rmse:.4f}")
    print(f"  MAE              : {rand_mae:.4f}")
    print()

    booster = load_booster()
    if booster is None:
        return

    dtest = make_dmatrix(df, station_categories)
    preds = booster.predict(dtest)
    rounded = np.round(preds).astype(int)

    correct = int(np.sum(rounded == y))
    accuracy = correct / len(y)
    raw_mse = mean_squared_error(y, preds)
    raw_rmse = np.sqrt(raw_mse)
    raw_mae = mean_absolute_error(y, preds)

    print("XGBoost model")
    print(f"Rows evaluated : {len(y):,}")
    print(f"Correct (exact): {correct:,} / {len(y):,}  ({accuracy:.2%})")
    print(f"Raw mean sq error  : {raw_mse:.4f}")
    print(f"Raw root mean sq   : {raw_rmse:.4f}")
    print(f"Raw mean abs error : {raw_mae:.4f}")


if __name__ == "__main__":
    eval(MIXED_BUCKET)
