#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Transform raw format to training format
"""

# Imports
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import pandas as pd
from tqdm import tqdm

# Constants
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_TIMEZONE = os.environ.get("WEATHER_TIMEZONE", "America/Los_Angeles")
WEATHER_COORD_DECIMALS = int(os.environ.get("WEATHER_COORD_DECIMALS", "4"))
STATION_COORDINATE_THRESHOLD = 0.001
HTTP_TIMEOUT = 10
HTTP_RETRIES = 3
HTTP_RETRY_CODES = frozenset({429, 500, 502, 503, 504})
COLUMN_ALIASES = {
    "started_at": "start_time",
    "ended_at": "end_time",
    "start_station_latitude": "start_lat",
    "start_station_longitude": "start_lng",
    "end_station_latitude": "end_lat",
    "end_station_longitude": "end_lng",
}
REQUIRED_COLUMNS = [
    "start_time",
    "end_time",
    "start_station_id",
    "end_station_id",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
]
FEATURE_COLUMNS = [
    "day_of_week",
    "time_bucket",
    "station_id",
    "temperature",
    "precipitation",
    "wind",
]
TARGET_COLUMN = "net_flow"
MODEL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def normalize_columns(data):
    normalized = data.rename(columns=COLUMN_ALIASES)
    missing = [column for column in REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return normalized


def parse_station(raw_station, column):
    if pd.isna(raw_station):
        raise ValueError(f"Missing {column}")
    text = str(raw_station).strip()
    if not text:
        raise ValueError(f"Missing {column}")
    return text


def parse_coordinate(raw_coordinate, column):
    if pd.isna(raw_coordinate):
        raise ValueError(f"Missing {column}")
    try:
        return float(raw_coordinate)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {column}: {raw_coordinate!r}")


def floor_to_bucket(raw_time, column):
    event_time = pd.to_datetime(raw_time, errors="coerce")
    if pd.isna(event_time):
        raise ValueError(f"Invalid {column}: {raw_time!r}")
    if event_time.tzinfo is not None:
        event_time = event_time.tz_convert(None)
    return event_time.floor("15min")


def time_bucket_for(bucket_start):
    minutes_into_day = bucket_start.hour * 60 + bucket_start.minute
    return (minutes_into_day // 15) + 1


def fetch_weather_range(latitude, longitude, start_date, end_date):
    query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m,precipitation,wind_speed_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": WEATHER_TIMEZONE,
        }
    )
    url = f"{WEATHER_URL}?{query}"
    for attempt in range(HTTP_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            if e.code not in HTTP_RETRY_CODES or attempt == HTTP_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

    hourly = payload["hourly"]
    weather_by_hour = {}
    for index, raw_time in enumerate(hourly["time"]):
        weather_by_hour[pd.Timestamp(raw_time)] = {
            "temperature": round(hourly["temperature_2m"][index], 2),
            "precipitation": round(hourly["precipitation"][index], 2),
            "wind": round(hourly["wind_speed_10m"][index], 2),
        }
    return weather_by_hour


def prefetch_station_weather(bounds_by_coord):
    weather_by_coord = {}
    for (latitude, longitude), (start_date, end_date) in tqdm(
        bounds_by_coord.items(), desc="Weather", unit="station"
    ):
        try:
            weather_by_coord[(latitude, longitude)] = fetch_weather_range(
                latitude, longitude, start_date, end_date
            )
        except Exception as e:
            print(f"Weather fetch failed at {latitude}, {longitude}: {e}")
    return weather_by_coord


def coordinates_match(first_coordinate, next_coordinate):
    return (
        abs(first_coordinate[0] - next_coordinate[0]) <= STATION_COORDINATE_THRESHOLD
        and abs(first_coordinate[1] - next_coordinate[1]) <= STATION_COORDINATE_THRESHOLD
    )


def save_station_coordinate(station_coordinates, station_id, latitude, longitude):
    coordinate = (
        round(latitude, WEATHER_COORD_DECIMALS),
        round(longitude, WEATHER_COORD_DECIMALS),
    )
    first_coordinate = station_coordinates.setdefault(station_id, coordinate)
    if not coordinates_match(first_coordinate, coordinate):
        raise ValueError(
            f"Station {station_id} moved from {first_coordinate} to {coordinate}"
        )


def transform(data: pd.DataFrame) -> pd.DataFrame:
    # Validate columns
    data = normalize_columns(data)
    
    # Build counts, skip bad rows
    bucket_counts = {}
    station_coordinates = {}
    trip_iter = tqdm(
        data.itertuples(index=False),
        total=len(data),
        desc="Trip rows",
        unit="trip",
    )
    for row in trip_iter:
        try:
            start_bucket = floor_to_bucket(row.start_time, "start_time")
            start_station = parse_station(row.start_station_id, "start_station_id")
            start_lat = parse_coordinate(row.start_lat, "start_lat")
            start_lng = parse_coordinate(row.start_lng, "start_lng")
            end_bucket = floor_to_bucket(row.end_time, "end_time")
            end_station = parse_station(row.end_station_id, "end_station_id")
            end_lat = parse_coordinate(row.end_lat, "end_lat")
            end_lng = parse_coordinate(row.end_lng, "end_lng")

            key = (start_bucket, start_station)
            bucket_counts[key] = bucket_counts.get(key, 0) - 1
            save_station_coordinate(station_coordinates, start_station, start_lat, start_lng)

            key = (end_bucket, end_station)
            bucket_counts[key] = bucket_counts.get(key, 0) + 1
            save_station_coordinate(station_coordinates, end_station, end_lat, end_lng)
        except ValueError as e:
            print(e)
            continue

    if not bucket_counts:
        return pd.DataFrame([], columns=MODEL_COLUMNS, index=None)

    # One archive request per station (full date span), not per day
    bounds_by_coord = {}
    for (bucket, station_id), _ in bucket_counts.items():
        day = bucket.strftime("%Y-%m-%d")
        coord = station_coordinates[station_id]
        if coord not in bounds_by_coord:
            bounds_by_coord[coord] = [day, day]
        else:
            lo, hi = bounds_by_coord[coord]
            if day < lo:
                bounds_by_coord[coord][0] = day
            if day > hi:
                bounds_by_coord[coord][1] = day

    station_hourly_weather = prefetch_station_weather(bounds_by_coord)

    # Flatten rows
    rows = []
    sorted_buckets = sorted(bucket_counts.items())
    for key, net_flow in tqdm(
        sorted_buckets, desc="Buckets & weather", unit="bucket"
    ):
        bucket, station_id = key
        weather_hour = bucket.floor("h")
        latitude, longitude = station_coordinates[station_id]
        weather_by_hour = station_hourly_weather.get((latitude, longitude), {})
        weather = weather_by_hour.get(weather_hour)
        if weather is None:
            print(f"Missing weather for {weather_hour} at {latitude}, {longitude}")
            continue

        rows.append(
            {
                "day_of_week": bucket.dayofweek,
                "time_bucket": time_bucket_for(bucket),
                "station_id": station_id,
                "temperature": weather["temperature"],
                "precipitation": weather["precipitation"],
                "wind": weather["wind"],
                "net_flow": net_flow,
            }
        )

    return pd.DataFrame(rows, columns=MODEL_COLUMNS, index=None)
