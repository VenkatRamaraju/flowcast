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
from train.model.data import DEFAULT_BUCKET
from train.model.train_categorical import FEATURES, HELD_OUT_KEYS_PATH, MODEL_PATH, STATION_CATEGORIES_PATH, TARGET, make_dmatrix, normalize_station_ids, read_csv, validate_columns


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


def eval():
    booster = load_booster()
    if booster is None:
        return

    client = boto3.client("s3")
    held_out_keys = read_json(HELD_OUT_KEYS_PATH)
    station_categories = read_json(STATION_CATEGORIES_PATH)
    df = load_eval_data(client, DEFAULT_BUCKET, held_out_keys)
    y = df[TARGET].values

    dtest = make_dmatrix(df, station_categories)
    preds = booster.predict(dtest)
    rounded = np.round(preds).astype(int)

    correct = int(np.sum(rounded == y))
    accuracy = correct / len(y)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)

    print(df[TARGET].describe())
    print(df[TARGET].value_counts().head(20))

    print(f"Rows evaluated : {len(y):,}")
    print(f"Correct (exact): {correct:,} / {len(y):,}  ({accuracy:.2%})")
    print(f"Mean sq error  : {mse:.4f}")
    print(f"Root mean sq   : {rmse:.4f}")
    print(f"Mean abs error : {mae:.4f}")


if __name__ == "__main__":
    eval()
