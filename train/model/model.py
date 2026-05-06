#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Train an XGBoost net_flow regressor with categorical station IDs
"""

# Imports
import json
from pathlib import Path
import boto3
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from train.model.data import MIXED_BUCKET, list_csv_keys

# Constants
FEATURES = [
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
TARGET = "net_flow"
MODEL_COLUMNS = FEATURES + [TARGET]
ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model-categorical.ubj"
STATION_CATEGORIES_PATH = ARTIFACTS_DIR / "station-categories.json"
HELD_OUT_KEYS_PATH = ARTIFACTS_DIR / "held-out-keys.json"
TRAIN_FILES = 95
HELD_OUT_FILES = 5
EXPECTED_FILES = TRAIN_FILES + HELD_OUT_FILES
NUM_BOOST_ROUNDS = 6_000
VALIDATION_FILES = 5
EARLY_STOPPING_ROUNDS = 300

XGB_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "tree_method": "hist",
    "max_depth": 10,
    "min_child_weight": 2,
    "learning_rate": 0.05,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "reg_alpha": 0.0,
    "max_bin": 256,
    "max_cat_to_onehot": 1,
    "max_cat_threshold": 128,
    "seed": 42,
    "verbosity": 1,
}


def validate_columns(df, key):
    columns = list(df.columns)
    missing = [column for column in MODEL_COLUMNS if column not in columns]
    if missing:
        message = f"{key} has columns {columns}, expected at least {MODEL_COLUMNS}."
        message += (
            f" Missing {missing}. These rows were built with an older transform schema. "
            "Re-run ETL so S3 gets the new columns: "
            "`python main.py --data 0 <N>` over your catalogue slice (see train.data.load.load), "
            "then rebuild and upload `mixed/part-*.csv` to the bucket train() reads from."
        )
        raise ValueError(message)


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


def drop_rows_with_integer_station_ids(df):
    digits_only = df["station_id"].str.fullmatch(r"\d+", na=False)
    return df.loc[~digits_only].reset_index(drop=True)


def load_training_data(client, bucket, keys):
    frames = []
    for i, key in enumerate(keys):
        raw = read_csv(client, bucket, key)
        validate_columns(raw, key)
        frames.append(raw)
        print(f"Loaded file {i + 1:,}/{len(keys):,}: {key} | rows: {len(raw):,}")
    df = pd.concat(frames, ignore_index=True)
    df["station_id"] = normalize_station_ids(df["station_id"])
    df = drop_rows_with_integer_station_ids(df)
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


def print_pre_training_report(df_train, df_validation):
    df = pd.concat([df_train, df_validation], ignore_index=True)
    y = np.round(df[TARGET].values).astype(np.int64)
    y_min = int(y.min())
    y_max = int(y.max())
    rng = np.random.default_rng(42)
    random_y = rng.integers(y_min, y_max + 1, size=len(y), endpoint=False)
    acc = float(np.mean(random_y == y))
    rmse = float(np.sqrt(mean_squared_error(y, random_y)))
    mae = float(mean_absolute_error(y, random_y))
    print(
        f"Rows: {len(df_train):,} fit + {len(df_validation):,} val = {len(df):,}; "
        f"stations {df['station_id'].nunique():,}; "
        f"net_flow min={y_min} max={y_max} mean={float(y.mean()):.2f} std={float(y.std()):.2f}"
    )
    print(
        f"Uniform random int in [{y_min}, {y_max}] vs truth: "
        f"accuracy={acc:.2%}  RMSE={rmse:.4f}  MAE={mae:.4f}"
    )


def train(bucket):
    ensure_artifacts_dir()
    client = boto3.client("s3")
    keys = list_csv_keys(bucket)
    if len(keys) != EXPECTED_FILES:
        print(f"Expected {EXPECTED_FILES} files, got {len(keys)}")
        return
    if TRAIN_FILES <= VALIDATION_FILES:
        print("TRAIN_FILES must be greater than VALIDATION_FILES")
        return

    train_keys = keys[:TRAIN_FILES]
    fit_keys = train_keys[:-VALIDATION_FILES]
    validation_keys = train_keys[-VALIDATION_FILES:]
    held_out_keys = keys[TRAIN_FILES:]
    write_json(HELD_OUT_KEYS_PATH, held_out_keys)

    df_train = load_training_data(client, bucket, fit_keys)
    df_validation = load_training_data(client, bucket, validation_keys)
    station_categories = sorted(
        set(df_train["station_id"].unique()) | set(df_validation["station_id"].unique())
    )
    print_pre_training_report(df_train, df_validation)

    print(f"Training files: {len(train_keys):,}")
    print(f"Fit files: {len(fit_keys):,}")
    print(f"Validation files: {len(validation_keys):,}")
    print(f"Held out files: {len(held_out_keys):,}")
    print(f"Fit rows: {len(df_train):,}")
    print(f"Validation rows: {len(df_validation):,}")
    print(f"Station categories: {len(station_categories):,}")

    dtrain = make_dmatrix(df_train, station_categories)
    dvalidation = make_dmatrix(df_validation, station_categories)
    print(f"Training {NUM_BOOST_ROUNDS:,} trees")
    booster = xgb.train(
        XGB_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dvalidation, "validation")],
        callbacks=[
            xgb.callback.EarlyStopping(
                rounds=EARLY_STOPPING_ROUNDS,
                save_best=True,
            )
        ],
        verbose_eval=50,
    )
    print(f"Best iteration: {booster.best_iteration:,}")
    print(f"Best validation mae: {booster.best_score}")

    save_model(booster, MODEL_PATH)
    write_json(STATION_CATEGORIES_PATH, station_categories)
    print(f"Model saved to {MODEL_PATH}")