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
STATION_FEATURE = "station_id"
EMBEDDED_FEATURE_SPECS = {
    "day_of_week": (7, 0),
    "time_bucket": (96, 1),
    "week_of_year": (52, 1),
    "month": (12, 1),
}
EMBEDDED_FEATURES = list(EMBEDDED_FEATURE_SPECS)
EMBEDDED_FEATURE_DIMS = {
    "day_of_week": 4,
    "time_bucket": 16,
    "week_of_year": 8,
    "month": 4,
}
RAW_NUMERIC_FEATURES = [
    column
    for column in FEATURES
    if column != STATION_FEATURE and column not in EMBEDDED_FEATURES
]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES
TARGET = "net_flow"
MODEL_COLUMNS = FEATURES + [TARGET]
TRAIN_FILES = 95
VALIDATION_FILES = 5
HELD_OUT_FILES = 5
EXPECTED_FILES = TRAIN_FILES + HELD_OUT_FILES
CHECKPOINT_VERSION = 5
NN_ARTIFACTS_DIR = Path(__file__).resolve().parents[3] / "artifacts" / "nn"
MODEL_PATH = NN_ARTIFACTS_DIR / "nn-model.pt"
BEST_METRICS_PATH = NN_ARTIFACTS_DIR / "nn-model-best.json"
STATION_CATEGORIES_PATH = NN_ARTIFACTS_DIR / "station-categories.json"
NORMALIZATION_STATS_PATH = NN_ARTIFACTS_DIR / "normalization-stats.json"


class FlowcastNet(nn.Module):
    def __init__(self, numeric_dim, station_count):
        super().__init__()
        self.numeric_dim = numeric_dim
        self.station_count = station_count
        self.station_embedding_dim = 32
        self.station_embedding = nn.Embedding(station_count, self.station_embedding_dim)
        self.feature_embeddings = nn.ModuleDict(
            {
                column: nn.Embedding(spec[0], EMBEDDED_FEATURE_DIMS[column])
                for column, spec in EMBEDDED_FEATURE_SPECS.items()
            }
        )
        input_dim = (
            numeric_dim
            + self.station_embedding_dim
            + sum(EMBEDDED_FEATURE_DIMS.values())
        )
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, numeric_x, station_ids, embedded_x):
        if numeric_x.ndim != 2:
            raise ValueError("Expected numeric_x with shape [batch_size, num_features]")
        if numeric_x.shape[1] != self.numeric_dim:
            raise ValueError(
                f"Expected {self.numeric_dim} features, got {numeric_x.shape[1]}"
            )
        if station_ids.ndim != 1:
            raise ValueError("Expected station_ids with shape [batch_size]")
        if embedded_x.ndim != 2:
            raise ValueError("Expected embedded_x with shape [batch_size, num_features]")
        if embedded_x.shape[1] != len(EMBEDDED_FEATURES):
            raise ValueError(
                f"Expected {len(EMBEDDED_FEATURES)} embedded features, "
                f"got {embedded_x.shape[1]}"
            )

        parts = [numeric_x, self.station_embedding(station_ids)]
        for i, column in enumerate(EMBEDDED_FEATURES):
            parts.append(self.feature_embeddings[column](embedded_x[:, i]))
        x = torch.cat(parts, dim=1)
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
    for key, frame in iter_csv_files_for_keys(bucket, keys):
        validate_columns(frame, key)
        frame["station_id"] = normalize_station_ids(frame["station_id"])
        frame = drop_rows_with_integer_station_ids(frame)
        yield key, frame


def update_numeric_stats(stats, frame):
    work = convert_numeric_features(frame)
    numeric = work[NUMERIC_FEATURES].dropna()
    if numeric.empty:
        return
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    stats["count"] += values.shape[0]
    stats["sum"] += values.sum(axis=0)
    stats["square_sum"] += np.square(values).sum(axis=0)


def finalize_numeric_stats(stats):
    if stats["count"] == 0:
        raise ValueError("No rows found for numeric normalization")
    mean = stats["sum"] / stats["count"]
    variance = (stats["square_sum"] / stats["count"]) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return {
        "mean": dict(zip(NUMERIC_FEATURES, mean.tolist())),
        "std": dict(zip(NUMERIC_FEATURES, std.tolist())),
    }


def convert_numeric_features(df):
    work = df.copy()
    for column in RAW_NUMERIC_FEATURES + EMBEDDED_FEATURES:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    return work


def encode_embedded_features(df):
    work = df.copy()
    for column, spec in EMBEDDED_FEATURE_SPECS.items():
        size, offset = spec
        codes = work[column] - offset
        valid = codes.between(0, size - 1) & ((codes % 1) == 0)
        work[column] = codes.where(valid)
    return work


def collect_training_metadata(bucket, fit_keys, validation_keys):
    station_categories = set()
    total_rows = 0
    stats = {
        "count": 0,
        "sum": np.zeros(len(NUMERIC_FEATURES), dtype=np.float64),
        "square_sum": np.zeros(len(NUMERIC_FEATURES), dtype=np.float64),
    }
    for key, frame in iter_training_frames(bucket, fit_keys):
        total_rows += len(frame)
        station_categories.update(frame["station_id"].dropna().unique())
        update_numeric_stats(stats, frame)
        del frame
        gc.collect()
    for key, frame in iter_training_frames(bucket, validation_keys):
        total_rows += len(frame)
        station_categories.update(frame["station_id"].dropna().unique())
        del frame
        gc.collect()
    if total_rows == 0:
        raise ValueError("No rows found for this split")
    if not station_categories:
        raise ValueError("No station IDs found for this split")
    return sorted(station_categories), finalize_numeric_stats(stats)


def prepare(df, station_categories, normalization_stats):
    work = df[MODEL_COLUMNS].copy()
    categories = pd.CategoricalDtype(categories=station_categories)
    work["station_id"] = work["station_id"].astype(categories)
    work = convert_numeric_features(work)
    work = encode_embedded_features(work)
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    work = work.dropna(
        subset=[STATION_FEATURE, TARGET] + NUMERIC_FEATURES + EMBEDDED_FEATURES
    ).reset_index(drop=True)
    work["station_id"] = work["station_id"].cat.codes.astype(np.int64)
    for column in EMBEDDED_FEATURES:
        work[column] = work[column].astype(np.int64)
    for column in NUMERIC_FEATURES:
        mean = normalization_stats["mean"][column]
        std = normalization_stats["std"][column]
        work[column] = (work[column] - mean) / std
    return work


def make_arrays(df, station_categories, normalization_stats):
    prepared = prepare(df, station_categories, normalization_stats)
    if prepared.empty:
        return None
    numeric_features = np.ascontiguousarray(
        prepared[NUMERIC_FEATURES].to_numpy(dtype=np.float32, copy=True)
    )
    station_ids = np.ascontiguousarray(
        prepared[STATION_FEATURE].to_numpy(dtype=np.int64, copy=True)
    )
    embedded_features = np.ascontiguousarray(
        prepared[EMBEDDED_FEATURES].to_numpy(dtype=np.int64, copy=True)
    )
    target = np.ascontiguousarray(
        prepared[TARGET].to_numpy(dtype=np.float32, copy=True)
    )
    return numeric_features, station_ids, embedded_features, target


def iter_batches(
    numeric_features,
    station_ids,
    embedded_features,
    target,
    batch_size,
    shuffle,
    rng,
):
    row_count = len(target)
    if shuffle:
        order = rng.permutation(row_count)
    else:
        order = None
    for start in range(0, row_count, batch_size):
        stop = min(start + batch_size, row_count)
        if order is None:
            yield (
                numeric_features[start:stop],
                station_ids[start:stop],
                embedded_features[start:stop],
                target[start:stop],
            )
        else:
            batch_rows = order[start:stop]
            yield (
                numeric_features[batch_rows],
                station_ids[batch_rows],
                embedded_features[batch_rows],
                target[batch_rows],
            )


def iter_array_files(bucket, keys, station_categories, normalization_stats):
    for key, frame in iter_training_frames(bucket, keys):
        arrays = make_arrays(frame, station_categories, normalization_stats)
        del frame
        if arrays is not None:
            numeric_features, station_ids, embedded_features, target = arrays
            yield key, numeric_features, station_ids, embedded_features, target
            del numeric_features
            del station_ids
            del embedded_features
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
    station_categories, normalization_stats = collect_training_metadata(
        bucket,
        fit_keys,
        validation_keys,
    )

    return fit_keys, validation_keys, station_categories, normalization_stats


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def write_json(path, data):
    NN_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w") as file:
        json.dump(data, file, indent=2, sort_keys=True)
    temp_path.replace(path)


def read_best_val_mae():
    if not BEST_METRICS_PATH.exists():
        return float("inf")
    with BEST_METRICS_PATH.open() as file:
        data = json.load(file)
    if data.get("checkpoint_version") != CHECKPOINT_VERSION:
        return float("inf")
    return float(data["best_val_mae"])


def write_best_metrics(best_val_mae, epoch):
    write_json(
        BEST_METRICS_PATH,
        {
            "best_val_mae": best_val_mae,
            "checkpoint_version": CHECKPOINT_VERSION,
            "epoch": epoch,
        },
    )


def pick_batch_size(device):
    if device.type == "cuda":
        return 16384
    if device.type == "mps":
        return 4096
    return 2048


def train(bucket):
    learning_rate = 1e-3
    num_epochs = 200
    seed = 2024
    rng = np.random.default_rng(seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    batch_size = pick_batch_size(device)
    print(f"device={device} batch_size={batch_size:,} epochs={num_epochs}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    best_val_mae = read_best_val_mae()

    fit_keys, validation_keys, station_categories, normalization_stats = load_data(bucket)
    write_json(STATION_CATEGORIES_PATH, station_categories)
    write_json(NORMALIZATION_STATS_PATH, normalization_stats)

    # Build model
    model = FlowcastNet(
        numeric_dim=len(NUMERIC_FEATURES),
        station_count=len(station_categories),
    ).to(device)

    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-6,
    )
    loss_fn = nn.L1Loss()

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        total_train_rows = 0

        for (
            key,
            numeric_array,
            station_array,
            embedded_array,
            target_array,
        ) in iter_array_files(bucket, fit_keys, station_categories, normalization_stats):
            for (
                numeric_batch,
                station_batch,
                embedded_batch,
                target_batch,
            ) in iter_batches(
                numeric_array,
                station_array,
                embedded_array,
                target_array,
                batch_size,
                shuffle=True,
                rng=rng,
            ):
                numeric_features = torch.tensor(numeric_batch, device=device)
                station_ids = torch.tensor(station_batch, device=device)
                embedded_features = torch.tensor(embedded_batch, device=device)
                target = torch.tensor(target_batch, device=device)
                prediction = model(numeric_features, station_ids, embedded_features)
                loss = loss_fn(prediction, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_rows = target.size(0)
                total_train_loss += float(loss.item()) * batch_rows
                total_train_rows += batch_rows
            del numeric_array
            del station_array
            del embedded_array
            del target_array
            gc.collect()
        if total_train_rows == 0:
            raise ValueError("No training rows remain after dropping incomplete rows")

        model.eval()
        absolute_error_sum = 0.0
        total_val_rows = 0
        with torch.inference_mode():
            for (
                key,
                numeric_array,
                station_array,
                embedded_array,
                target_array,
            ) in iter_array_files(
                bucket, validation_keys, station_categories, normalization_stats
            ):
                for (
                    numeric_batch,
                    station_batch,
                    embedded_batch,
                    target_batch,
                ) in iter_batches(
                    numeric_array,
                    station_array,
                    embedded_array,
                    target_array,
                    batch_size,
                    shuffle=False,
                    rng=rng,
                ):
                    numeric_features = torch.tensor(numeric_batch, device=device)
                    station_ids = torch.tensor(station_batch, device=device)
                    embedded_features = torch.tensor(embedded_batch, device=device)
                    target = torch.tensor(target_batch, device=device)
                    prediction = model(numeric_features, station_ids, embedded_features)
                    absolute_error_sum += float(
                        torch.abs(prediction - target).sum().item()
                    )
                    total_val_rows += target.size(0)
                del numeric_array
                del station_array
                del embedded_array
                del target_array
                gc.collect()
        if total_val_rows == 0:
            raise ValueError("No validation rows remain after dropping incomplete rows")

        train_mae = total_train_loss / total_train_rows
        val_mae = absolute_error_sum / total_val_rows
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]
        saved = val_mae < best_val_mae
        if saved:
            save_model(model, MODEL_PATH)
            write_best_metrics(val_mae, epoch + 1)
            best_val_mae = val_mae
        print(
            f"Epoch {epoch + 1:2d}/{num_epochs}: "
            f"train_mae={train_mae:.4f} val_mae={val_mae:.4f} lr={current_lr:.2e}"
            + (" *" if saved else "")
        )


if __name__ == "__main__":
    train(MIXED_BUCKET)
