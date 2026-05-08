#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Load the trained XGBoost model and expose a net_flow inference function
"""

# Imports
import json
import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from train.model.model import FEATURES, MODEL_PATH, STATION_CATEGORIES_PATH

# Cached state
_booster = None
_station_categories = None
_station_dtype = None


def load():
    global _booster, _station_categories, _station_dtype
    if _booster is not None:
        return _booster, _station_categories, _station_dtype
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    if not STATION_CATEGORIES_PATH.exists():
        raise FileNotFoundError(f"Station categories not found at {STATION_CATEGORIES_PATH}")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    with STATION_CATEGORIES_PATH.open() as file:
        station_categories = json.load(file)
    _booster = booster
    _station_categories = station_categories
    _station_dtype = pd.CategoricalDtype(categories=station_categories)
    return _booster, _station_categories, _station_dtype


def predict_net_flow(
    day_of_week,
    time_bucket,
    is_weekend,
    week_of_year,
    month,
    is_us_federal_holiday,
    commute_hours,
    station_id,
    temperature,
    precipitation,
    wind,
):
    booster, station_categories, station_dtype = load()
    if station_id not in station_categories:
        raise ValueError(f"Unknown station_id: {station_id!r}")

    values = {
        "day_of_week": int(day_of_week),
        "time_bucket": int(time_bucket),
        "is_weekend": bool(is_weekend),
        "week_of_year": int(week_of_year),
        "month": int(month),
        "is_us_federal_holiday": bool(is_us_federal_holiday),
        "commute_hours": bool(commute_hours),
        "station_id": str(station_id),
        "temperature": float(temperature),
        "precipitation": float(precipitation),
        "wind": float(wind),
    }
    model_columns = booster.feature_names
    if not model_columns:
        model_columns = list(FEATURES)
    missing = [name for name in model_columns if name not in values]
    if missing:
        raise ValueError(
            f"Model expects features the API does not supply: {missing}. "
            "Retrain with train.model.model.FEATURES or upgrade the API schema."
        )
    row = {name: values[name] for name in model_columns}
    df = pd.DataFrame([row], columns=model_columns)
    df["station_id"] = df["station_id"].astype(station_dtype)

    dmatrix = xgb.DMatrix(df, enable_categorical=True)
    prediction = booster.predict(dmatrix)
    return round(float(prediction[0]), 2)
