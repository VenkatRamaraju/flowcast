#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Evaluate the trained XGBoost model against the held-out eval file
"""

# Imports
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from train.model.data import get_eval_file
from train.model.model import FEATURES, TARGET, load_training_state, prepare, validate_columns


def eval():
    booster, station_map, state = load_training_state()
    if booster is None:
        print("No model found in artifacts")
        return

    key, raw = get_eval_file()
    print(f"Eval file: {key} | rows: {len(raw):,}")

    validate_columns(raw, key)
    df = prepare(raw, station_map)

    X = df[FEATURES].values
    y = df[TARGET].values

    preds = booster.predict(xgb.DMatrix(X))
    rounded = np.round(preds).astype(int)

    correct = int(np.sum(rounded == y))
    accuracy = correct / len(y)
    mae = mean_absolute_error(y, preds)

    print(df[TARGET].describe())
    print(df[TARGET].value_counts().head(20))

    print(f"Rows evaluated : {len(y):,}")
    print(f"Correct (exact): {correct:,} / {len(y):,}  ({accuracy:.2%})")
    print(f"Mean abs error : {mae:.4f}")


if __name__ == "__main__":
    eval()
