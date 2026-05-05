#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Train an XGBoost net_flow regressor incrementally over streamed S3 CSVs
"""

# Imports
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from train.model.data import MIXED_BUCKET, iter_csv_files

# Constants
FEATURES = ["day_of_week", "time_bucket", "station_id", "temperature", "precipitation", "wind"]
TARGET = "net_flow"
MODEL_COLUMNS = FEATURES + [TARGET]
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.ubj"
CHECKPOINT_PATH = ARTIFACTS_DIR / "checkpoint.json"
STATION_MAP_PATH = ARTIFACTS_DIR / "station_map.json"
ROUNDS_PER_FILE = 36
MAX_TOTAL_ROUNDS = 3_600
TEST_SIZE = 0.1

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "max_depth": 9,
    "min_child_weight": 50,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "max_bin": 256,
    "seed": 42,
    "verbosity": 1,
}


def validate_columns(df, key):
    columns = list(df.columns)
    if columns != MODEL_COLUMNS:
        raise ValueError(
            f"{key} has columns {columns}, expected exactly {MODEL_COLUMNS}"
        )


def encode_stations(series, station_map):
    station_values = series.astype(str)
    for val in station_values.unique():
        if val not in station_map:
            station_map[val] = len(station_map)
    return station_values.map(station_map)


def prepare(df, station_map):
    df = df.copy()
    df["station_id"] = encode_stations(df["station_id"], station_map)
    return df


def ensure_artifacts_dir():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path, default):
    if not path.exists():
        return default
    with path.open() as file:
        return json.load(file)


def write_json(path, data):
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temp_path.replace(path)


def artifact_path(state, field, default_path):
    file_name = state.get(field)
    if not file_name:
        return default_path
    return ARTIFACTS_DIR / file_name


def checkpoint_id(state):
    return (
        f"{len(state['processed_keys']):06d}-"
        f"{state['total_rounds']:06d}-"
        f"{state['rows_seen']:012d}"
    )


def load_booster(path):
    if not path.exists():
        return None
    booster = xgb.Booster()
    booster.load_model(str(path))
    return booster


def save_model(booster, path):
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    booster.save_model(str(temp_path))
    temp_path.replace(path)


def save_checkpoint(booster, station_map, state):
    ensure_artifacts_dir()
    artifact_id = checkpoint_id(state)
    model_path = ARTIFACTS_DIR / f"model-{artifact_id}.ubj"
    station_map_path = ARTIFACTS_DIR / f"station-map-{artifact_id}.json"

    save_model(booster, model_path)
    write_json(station_map_path, station_map)

    checkpoint = dict(state)
    checkpoint["checkpoint_version"] = 2
    checkpoint["model_file"] = model_path.name
    checkpoint["station_map_file"] = station_map_path.name
    write_json(CHECKPOINT_PATH, checkpoint)


def load_training_state():
    ensure_artifacts_dir()
    state = read_json(
        CHECKPOINT_PATH,
        {"processed_keys": [], "rows_seen": 0, "total_rounds": 0},
    )
    state.setdefault("processed_keys", [])
    state.setdefault("rows_seen", 0)
    state.setdefault("total_rounds", 0)
    booster = load_booster(artifact_path(state, "model_file", MODEL_PATH))
    station_map = read_json(artifact_path(state, "station_map_file", STATION_MAP_PATH), {})
    return booster, station_map, state


def train():
    booster, station_map, state = load_training_state()
    processed_keys = set(state["processed_keys"])
    print(
        f"Loaded state | processed files: {len(processed_keys):,} | "
        f"rows seen: {state['rows_seen']:,}"
    )

    if booster is not None:
        total_trees = booster.num_boosted_rounds()
        print(f"Resuming from {MODEL_PATH} with {total_trees:,} trees")

    for i, (key, raw) in enumerate(iter_csv_files(MIXED_BUCKET)):
        print("=" * 80)
        if key in processed_keys:
            print(f"Skipping {key}")
            continue

        print(f"Starting file {i + 1}: {key}")
        current_rounds = 0 if booster is None else booster.num_boosted_rounds()
        rounds_left = MAX_TOTAL_ROUNDS - current_rounds
        if rounds_left <= 0:
            print(f"Stopping at {MAX_TOTAL_ROUNDS:,} trees")
            break

        rounds_this_file = min(ROUNDS_PER_FILE, rounds_left)
        print(f"Preparing {len(raw):,} rows")
        validate_columns(raw, key)
        df = prepare(raw, station_map)

        X = df[FEATURES].values
        y = df[TARGET].values

        print("Splitting train/test")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )

        # Incremental train
        print(
            f"Training {rounds_this_file:,} trees | "
            f"current trees: {current_rounds:,}"
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        booster = xgb.train(
            XGB_PARAMS,
            dtrain,
            num_boost_round=rounds_this_file,
            xgb_model=booster,
            verbose_eval=True,
        )

        dtest = xgb.DMatrix(X_test)
        preds = booster.predict(dtest)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        state["processed_keys"].append(key)
        state["rows_seen"] += len(df)
        state["total_rounds"] = booster.num_boosted_rounds()
        print("Saving checkpoint")
        save_checkpoint(booster, station_map, state)

        total_trees = booster.num_boosted_rounds()
        print(
            f"Batch {i + 1} | file: {key} | rows: {len(df):,} | "
            f"rmse: {rmse:.4f} | mae: {mae:.4f} | total trees: {total_trees:,}"
        )

    if booster is None:
        print("No training data found")
        return

    save_model(booster, MODEL_PATH)
    write_json(STATION_MAP_PATH, station_map)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
