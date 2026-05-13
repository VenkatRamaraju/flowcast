#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Train a PyTorch net_flow regressor on mixed CSV data
"""

# Imports
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from src.model.data import MIXED_BUCKET, iter_csv_files_for_keys, list_csv_keys

# Constants
FEATURES = [
    "day_of_week",
    "time_bucket",
    "is_weekend",
    "week_of_year",
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
TRAIN_FILES = 95
VALIDATION_FILES = 5
HELD_OUT_FILES = 5
EXPECTED_FILES = TRAIN_FILES + HELD_OUT_FILES
NN_ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "nn"
MODEL_PATH = NN_ARTIFACTS_DIR / "nn-model.pt"
BEST_METRICS_PATH = NN_ARTIFACTS_DIR / "nn-model-best.json"


class FlowcastNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        dropout_p = 0.1
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.ndim != 2:
            raise ValueError("Expected x with shape [batch_size, num_features]")
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {x.shape[1]}"
            )

        return self.model(x).squeeze(1)


def validate_columns(df, key):
    columns = list(df.columns)
    missing = [column for column in MODEL_COLUMNS if column not in columns]
    if missing:
        message = f"{key} has columns {columns}, expected at least {MODEL_COLUMNS}."
        message += (
            f" Missing {missing}. These rows were built with an older transform schema. "
            "Re-run ETL so S3 gets the new columns: "
            "`python main.py --data 0 <N>` over your catalogue slice (see src.data.load.load), "
            "then rebuild and upload `mixed/part-*.csv` to the bucket train() reads from."
        )
        raise ValueError(message)


def normalize_station_ids(series):
    cleaned = series.astype("string").str.strip()
    numeric = pd.to_numeric(cleaned, errors="coerce")
    is_integer = numeric.notna() & ((numeric % 1) == 0)
    as_int = numeric.where(is_integer).astype("Int64")
    normalized = cleaned.mask(is_integer, as_int.astype("string"))
    return normalized


def drop_rows_with_integer_station_ids(df):
    digits_only = df["station_id"].astype("string").str.fullmatch(r"\d+", na=False)
    return df.loc[~digits_only].reset_index(drop=True)


def iter_training_frames(bucket, keys):
    for i, (key, frame) in enumerate(iter_csv_files_for_keys(bucket, keys)):
        validate_columns(frame, key)
        frame["station_id"] = normalize_station_ids(frame["station_id"])
        frame = drop_rows_with_integer_station_ids(frame)
        print(
            f"Loaded file {i + 1:,}/{len(keys):,}: {key} | rows: {len(frame):,}"
        )
        yield key, frame


def collect_station_categories(bucket, keys):
    station_categories = set()
    total_rows = 0
    for key, frame in iter_training_frames(bucket, keys):
        total_rows += len(frame)
        station_categories.update(frame["station_id"].dropna().unique())
        del frame
        gc.collect()
    if total_rows == 0:
        raise ValueError("No rows found for this split")
    if not station_categories:
        raise ValueError("No station IDs found for this split")
    return sorted(station_categories), total_rows


def prepare(df, station_categories):
    work = df[MODEL_COLUMNS].copy()
    categories = pd.CategoricalDtype(categories=station_categories)
    work["station_id"] = work["station_id"].astype(categories)
    for column in FEATURES:
        if column == "station_id":
            continue
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    rows_before = len(work)
    work = work.dropna(subset=MODEL_COLUMNS).reset_index(drop=True)
    dropped_rows = rows_before - len(work)
    if dropped_rows > 0:
        print(f"Dropped rows: {dropped_rows:,} (missing required values)")
    work["station_id"] = work["station_id"].cat.codes.astype(np.int64)
    return work


def make_arrays(df, station_categories):
    prepared = prepare(df, station_categories)
    if prepared.empty:
        return None
    features = np.ascontiguousarray(
        prepared[FEATURES].to_numpy(dtype=np.float32, copy=True)
    )
    target = np.ascontiguousarray(
        prepared[TARGET].to_numpy(dtype=np.float32, copy=True)
    )
    return features, target


def iter_batches(features, target, batch_size, shuffle, rng):
    row_count = len(target)
    if shuffle:
        order = rng.permutation(row_count)
    else:
        order = None
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        if order is None:
            yield features[start:stop], target[start:stop]
        else:
            batch_rows = order[start:stop]
            yield features[batch_rows], target[batch_rows]


def iter_array_files(bucket, keys, station_categories):
    for key, frame in iter_training_frames(bucket, keys):
        arrays = make_arrays(frame, station_categories)
        del frame
        if arrays is None:
            print(f"Skipped file with no usable rows: {key}")
        else:
            features, target = arrays
            yield key, features, target
            del features
            del target
        gc.collect()


def load_data(bucket):
    keys = list_csv_keys(bucket)
    if len(keys) != EXPECTED_FILES:
        raise ValueError(f"Expected {EXPECTED_FILES} files, got {len(keys)}")
    if TRAIN_FILES <= VALIDATION_FILES:
        raise ValueError("TRAIN_FILES must be greater than VALIDATION_FILES")

    train_keys = keys[:TRAIN_FILES]
    fit_keys = train_keys[:-VALIDATION_FILES]
    validation_keys = train_keys[-VALIDATION_FILES:]
    station_categories, total_rows = collect_station_categories(
        bucket,
        fit_keys + validation_keys,
    )

    print(
        f"Rows: {total_rows:,}; station categories: {len(station_categories):,}"
    )
    return fit_keys, validation_keys, station_categories, len(FEATURES)


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


def read_best_val_mae():
    if not BEST_METRICS_PATH.exists():
        return float("inf")
    with BEST_METRICS_PATH.open() as file:
        data = json.load(file)
    return float(data["best_val_mae"])


def write_best_metrics(best_val_mae, epoch):
    NN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = BEST_METRICS_PATH.with_suffix(f"{BEST_METRICS_PATH.suffix}.tmp")
    with temp_path.open("w") as file:
        json.dump(
            {"best_val_mae": best_val_mae, "epoch": epoch},
            file,
            indent=2,
            sort_keys=True,
        )
    temp_path.replace(BEST_METRICS_PATH)


def pick_batch_size(device):
    if device.type == "cuda":
        return 16384
    if device.type == "mps":
        return 4096
    return 2048


def train(bucket):
    learning_rate = 1e-3
    num_epochs = 20
    seed = 2024
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS detected; using CPU for stability")
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    batch_size = pick_batch_size(device)
    print(f"device={device} batch_size={batch_size:,}")
    print("===== TRAINING START =====")

    torch.manual_seed(seed)
    np.random.seed(seed)

    best_val_mae = read_best_val_mae()
    if best_val_mae < float("inf"):
        print(f"Existing best val_mae: {best_val_mae:.4f} (must beat to save)")

    fit_keys, validation_keys, station_categories, input_dim = load_data(bucket)

    # Build model
    model = FlowcastNet(input_dim=input_dim).to(device)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.L1Loss()

    for epoch in range(num_epochs):
        print(f"===== EPOCH {epoch + 1}/{num_epochs} =====")
        # Train mode
        model.train()
        total_train_loss = 0.0
        total_train_rows = 0
        batch_index = 0

        # Train batches
        for key, features_array, target_array in iter_array_files(
            bucket,
            fit_keys,
            station_categories,
        ):
            print(f"Training file: {key}")
            for features_batch, target_batch in iter_batches(
                features_array,
                target_array,
                batch_size,
                shuffle=True,
                rng=rng,
            ):
                batch_index += 1
                # Move tensors
                features = torch.tensor(features_batch, device=device)
                target = torch.tensor(target_batch, device=device)

                # Forward pass
                prediction = model(features)
                loss = loss_fn(prediction, target)

                optimizer.zero_grad()
                # Backprop step
                loss.backward()
                optimizer.step()

                # Track metrics
                batch_rows = target.size(0)
                total_train_loss += float(loss.item()) * batch_rows
                total_train_rows += batch_rows
                if batch_index % 200 == 0:
                    running_mae = total_train_loss / total_train_rows
                    print(
                        f"----- epoch {epoch + 1} batch {batch_index} "
                        f"train_mae={running_mae:.4f}"
                    )
            del features_array
            del target_array
            gc.collect()
        if total_train_rows == 0:
            raise ValueError("No training rows remain after dropping incomplete rows")

        # Eval mode
        model.eval()
        absolute_error_sum = 0.0
        total_val_rows = 0
        # Disable grads
        with torch.inference_mode():
            # Validate batches
            for key, features_array, target_array in iter_array_files(
                bucket,
                validation_keys,
                station_categories,
            ):
                print(f"Validating file: {key}")
                for features_batch, target_batch in iter_batches(
                    features_array,
                    target_array,
                    batch_size,
                    shuffle=False,
                    rng=rng,
                ):
                    # Move tensors
                    features = torch.tensor(features_batch, device=device)
                    target = torch.tensor(target_batch, device=device)

                    # Forward pass
                    prediction = model(features)

                    # Sum errors
                    absolute_error_sum += float(
                        torch.abs(prediction - target).sum().item()
                    )
                    total_val_rows += target.size(0)
                del features_array
                del target_array
                gc.collect()
        if total_val_rows == 0:
            raise ValueError("No validation rows remain after dropping incomplete rows")

        train_mae = total_train_loss / total_train_rows
        val_mae = absolute_error_sum / total_val_rows
        print(
            f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
            f"train_mae={train_mae:.4f} val_mae={val_mae:.4f}"
        )
        print("===== EPOCH COMPLETE =====")

        if val_mae < best_val_mae:
            save_model(model, MODEL_PATH)
            write_best_metrics(val_mae, epoch + 1)
            best_val_mae = val_mae
            print(f"New best val_mae={val_mae:.4f}")


if __name__ == "__main__":
    train(MIXED_BUCKET)
