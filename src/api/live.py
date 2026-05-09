#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: HTTP handler for live net-flow prediction at a given station
"""

# Imports
import json
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, HTTPException
from src.data.transform import WEATHER_TIMEZONE, calendar_features_for_bucket
from src.model.inference import predict_net_flow

# Constants
REPO_ROOT = Path(__file__).resolve().parents[2]
STATION_MAPPING_PATH = REPO_ROOT / "artifacts" / "station-mapping.json"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 10
router = APIRouter()
cached_mapping = None


def load_station_mapping():
    global cached_mapping
    if cached_mapping is None:
        with STATION_MAPPING_PATH.open() as f:
            cached_mapping = json.load(f)
    return cached_mapping


def fetch_current_weather(lat, lon):
    query = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,precipitation,wind_speed_10m",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": WEATHER_TIMEZONE,
    })
    url = f"{WEATHER_URL}?{query}"
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    current = payload["current"]
    return {
        "temperature": round(float(current["temperature_2m"]), 2),
        "precipitation": round(float(current["precipitation"]), 2),
        "wind": round(float(current["wind_speed_10m"]), 2),
    }


@router.get("/stations/{station_id}/live")
def live(station_id: str):
    mapping = load_station_mapping()
    if station_id not in mapping:
        raise HTTPException(status_code=404, detail=f"Unknown station: {station_id!r}")
    lat, lon = mapping[station_id]
    try:
        weather = fetch_current_weather(lat, lon)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {exc}") from exc
    now = pd.Timestamp.now(tz=WEATHER_TIMEZONE).floor("15min")
    cal = calendar_features_for_bucket(now)
    try:
        prediction = predict_net_flow(**cal, station_id=station_id, **weather)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"station_id": station_id, "prediction": prediction, **weather}
