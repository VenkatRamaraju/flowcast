#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: HTTP handler for live net-flow prediction at a given station
"""

# Imports
import json
import time
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
GBFS_STATION_INFORMATION_URL = "https://gbfs.lyft.com/gbfs/2.3/bay/en/station_information.json"
GBFS_STATION_STATUS_URL = "https://gbfs.lyft.com/gbfs/2.3/bay/en/station_status.json"
GBFS_STATUS_TTL = 60
HTTP_TIMEOUT = 10
router = APIRouter()
cached_mapping = None
cached_bike_availability = None
cached_bike_availability_loaded_at = 0


def load_station_mapping():
    global cached_mapping
    if cached_mapping is None:
        with STATION_MAPPING_PATH.open() as f:
            cached_mapping = json.load(f)
    return cached_mapping


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_bike_availability(station_id):
    global cached_bike_availability, cached_bike_availability_loaded_at
    now = time.time()
    if cached_bike_availability is None or now - cached_bike_availability_loaded_at > GBFS_STATUS_TTL:
        station_payload = fetch_json(GBFS_STATION_INFORMATION_URL)
        status_payload = fetch_json(GBFS_STATION_STATUS_URL)
        station_names = {
            station["station_id"]: station["short_name"]
            for station in station_payload["data"]["stations"]
            if "short_name" in station
        }
        cached_bike_availability = {}
        for status in status_payload["data"]["stations"]:
            short_name = station_names.get(status["station_id"])
            if short_name is not None:
                cached_bike_availability[short_name] = {
                    "num_bikes_available": int(status["num_bikes_available"]),
                    "num_ebikes_available": int(status["num_ebikes_available"]),
                    "num_docks_available": int(status["num_docks_available"]),
                    "num_bikes_disabled": int(status["num_bikes_disabled"]),
                }
        cached_bike_availability_loaded_at = now
    if station_id not in cached_bike_availability:
        raise ValueError(f"Missing bike availability: {station_id!r}")
    return cached_bike_availability[station_id]


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
    payload = fetch_json(url)
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
    try:
        availability = fetch_bike_availability(station_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Bike availability fetch failed: {exc}") from exc
    now = pd.Timestamp.now(tz=WEATHER_TIMEZONE).floor("15min")
    cal = calendar_features_for_bucket(now)
    try:
        prediction = predict_net_flow(**cal, station_id=station_id, **weather)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"station_id": station_id, "prediction": prediction, **weather, **availability}
