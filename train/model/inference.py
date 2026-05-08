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

    row = {
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
    df = pd.DataFrame([row], columns=FEATURES)
    df["station_id"] = df["station_id"].astype(station_dtype)

    dmatrix = xgb.DMatrix(df, enable_categorical=True)
    prediction = booster.predict(dmatrix)
    return float(prediction[0])


# Test samples
def test():
    load()

    samples = [
        {
            "label": "Outside Lands festival exodus (Sat ~10:45pm, Aug)",
            "params": {
                "day_of_week": 5,
                "time_bucket": 91,
                "is_weekend": True,
                "week_of_year": 33,
                "month": 8,
                "is_us_federal_holiday": False,
                "commute_hours": False,
                "station_id": "SF-Outside Lands-Temp",
                "temperature": 56.9,
                "precipitation": 0.0,
                "wind": 8.6,
            },
        },
        {
            "label": "Hardly Strictly festival arrivals (Sat ~3pm, Oct)",
            "params": {
                "day_of_week": 5,
                "time_bucket": 61,
                "is_weekend": True,
                "week_of_year": 41,
                "month": 10,
                "is_us_federal_holiday": False,
                "commute_hours": False,
                "station_id": "HS-1",
                "temperature": 61.9,
                "precipitation": 0.0,
                "wind": 8.8,
            },
        },
        {
            "label": "Embarcadero AM commute inflow (Wed ~8:30am, Aug)",
            "params": {
                "day_of_week": 2,
                "time_bucket": 35,
                "is_weekend": False,
                "week_of_year": 32,
                "month": 8,
                "is_us_federal_holiday": False,
                "commute_hours": True,
                "station_id": "SF-F28-3",
                "temperature": 59.1,
                "precipitation": 0.0,
                "wind": 4.8,
            },
        },
        {
            "label": "Embarcadero PM commute outflow (Thu ~5pm, Jul)",
            "params": {
                "day_of_week": 3,
                "time_bucket": 69,
                "is_weekend": False,
                "week_of_year": 28,
                "month": 7,
                "is_us_federal_holiday": False,
                "commute_hours": True,
                "station_id": "SF-F28-3",
                "temperature": 64.2,
                "precipitation": 0.0,
                "wind": 13.0,
            },
        },
        {
            "label": "Quiet baseline (Tue 3am, Apr, mild + dry)",
            "params": {
                "day_of_week": 1,
                "time_bucket": 13,
                "is_weekend": False,
                "week_of_year": 14,
                "month": 4,
                "is_us_federal_holiday": False,
                "commute_hours": False,
                "station_id": "SF-F28-3",
                "temperature": 55.0,
                "precipitation": 0.0,
                "wind": 3.0,
            },
        },
    ]

    print(f"Running {len(samples)} sample inferences\n")
    for i, sample in enumerate(samples, start=1):
        prediction = predict_net_flow(**sample["params"])
        print(f"Sample {i}: {sample['label']}")
        print(f"  station_id    : {sample['params']['station_id']}")
        print(f"  day_of_week   : {sample['params']['day_of_week']}")
        print(f"  time_bucket   : {sample['params']['time_bucket']}")
        print(f"  week_of_year  : {sample['params']['week_of_year']}")
        print(f"  month         : {sample['params']['month']}")
        print(f"  weekend       : {sample['params']['is_weekend']}")
        print(f"  holiday       : {sample['params']['is_us_federal_holiday']}")
        print(f"  commute       : {sample['params']['commute_hours']}")
        print(
            f"  weather       : {sample['params']['temperature']}F, "
            f"{sample['params']['precipitation']}in, "
            f"{sample['params']['wind']}mph"
        )
        print(f"  predicted net_flow: {prediction:+.4f}\n")


if __name__ == "__main__":
    test()
