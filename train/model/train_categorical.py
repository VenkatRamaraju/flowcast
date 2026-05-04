#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Train an XGBoost net_flow regressor with categorical station IDs
"""

# Imports
import json
from pathlib import Path
import boto3
import pandas as pd
import xgboost as xgb
from train.model.data import DEFAULT_BUCKET, list_csv_keys

# Constants
FEATURES = ["day_of_week", "time_bucket", "station_id", "temperature", "precipitation", "wind"]
TARGET = "net_flow"
MODEL_COLUMNS = FEATURES + [TARGET]
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model-categorical.ubj"
STATION_CATEGORIES_PATH = ARTIFACTS_DIR / "station-categories.json"
HELD_OUT_KEYS_PATH = ARTIFACTS_DIR / "held-out-keys.json"
TRAIN_FILES = 95
HELD_OUT_FILES = 5
EXPECTED_FILES = TRAIN_FILES + HELD_OUT_FILES
NUM_BOOST_ROUNDS = 4_800

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 8,
    "min_child_weight": 20,
    "learning_rate": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 3.0,
    "reg_alpha": 0.0,
    "max_bin": 256,
    "max_cat_to_onehot": 16,
    "seed": 42,
    "verbosity": 1,
}


def validate_columns(df, key):
    columns = list(df.columns)
    if columns != MODEL_COLUMNS:
        raise ValueError(
            f"{key} has columns {columns}, expected exactly {MODEL_COLUMNS}"
        )


def ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temp_path.replace(path)


def save_model(booster, path):
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    booster.save_model(str(temp_path))
    temp_path.replace(path)


def read_csv(client, bucket, key):
    body = client.get_object(Bucket=bucket, Key=key)["Body"]
    return pd.read_csv(body)


def normalize_station_ids(series):
    cleaned = series.astype("string").str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    as_int = numeric.round().astype("Int64")
    normalized = cleaned.mask(as_int.notna(), as_int.astype("string"))
    return normalized.astype(str)


def load_training_data(client, bucket, keys):
    frames = []
    for i, key in enumerate(keys):
        raw = read_csv(client, bucket, key)
        validate_columns(raw, key)
        frames.append(raw)
        print(f"Loaded file {i + 1:,}/{len(keys):,}: {key} | rows: {len(raw):,}")
    df = pd.concat(frames, ignore_index=True)
    df["station_id"] = normalize_station_ids(df["station_id"])
    return df


def prepare(df, station_categories):
    df = df.copy()
    categories = pd.CategoricalDtype(categories=station_categories)
    df["station_id"] = df["station_id"].astype(categories)
    return df


def make_dmatrix(df, station_categories):
    df = prepare(df, station_categories)
    return xgb.DMatrix(
        df[FEATURES],
        label=df[TARGET],
        enable_categorical=True,
    )


def train(bucket=DEFAULT_BUCKET, prefix=""):
    ensure_artifacts_dir()
    client = boto3.client("s3")
    keys = list_csv_keys(bucket=bucket, prefix=prefix)
    if len(keys) != EXPECTED_FILES:
        print(f"Expected {EXPECTED_FILES} files, got {len(keys)}")
        return

    train_keys = keys[:TRAIN_FILES]
    held_out_keys = keys[TRAIN_FILES:]
    write_json(HELD_OUT_KEYS_PATH, held_out_keys)

    df = load_training_data(client, bucket, train_keys)
    station_categories = sorted(df["station_id"].unique())
    write_json(STATION_CATEGORIES_PATH, station_categories)

    print(f"Training files: {len(train_keys):,}")
    print(f"Held out files: {len(held_out_keys):,}")
    print(f"Training rows: {len(df):,}")
    print(f"Station categories: {len(station_categories):,}")

    dtrain = make_dmatrix(df, station_categories)
    print(f"Training {NUM_BOOST_ROUNDS:,} trees")
    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        verbose_eval=True,
    )

    save_model(booster, MODEL_PATH)
    write_json(STATION_CATEGORIES_PATH, station_categories)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
