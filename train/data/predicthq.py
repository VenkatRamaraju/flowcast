#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Page PredictHQ events and write one JSON array under artifacts/.
"""

# Imports
from datetime import datetime, timedelta, timezone
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from dotenv import load_dotenv
REPO_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables
load_dotenv(REPO_ROOT / ".env")

# Constants
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUT_PATH = ARTIFACTS_DIR / "predicthq_events.json"
START_GTE = "2022-01-01"
START_LTE = datetime.now(timezone.utc).date().isoformat()
WITHIN = "10mi@37.7749,-122.4194"
CATEGORIES = "conferences,expos,concerts,festivals,performing-arts,sports,community"
MIN_RANK = 40
MIN_LOCAL_RANK = 40
LIMIT = 100
API_KEY = os.environ["PREDICTHQ_API_KEY"]   
EVENTS_URL = "https://api.predicthq.com/v1/events/"


def month_windows(start_date, end_date):
    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        window_end = min(end_date, next_month - timedelta(days=1))
        yield current, window_end
        current = next_month


def fetch_window_events(headers, window_start, window_end):
    params = urllib.parse.urlencode(
        {
            "start.gte": window_start.isoformat(),
            "start.lte": window_end.isoformat(),
            "start.tz": "America/Los_Angeles",
            "category": CATEGORIES,
            "rank.gte": str(MIN_RANK),
            "local_rank.gte": str(MIN_LOCAL_RANK),
            "parent.include": "true",
            "private.include": "false",
            "sort": "start",
            "within": WITHIN,
            "limit": str(LIMIT),
        }
    )
    url = f"{EVENTS_URL}?{params}"
    events = []
    page = 1
    while url:
        print(
            f"{window_start.isoformat()}..{window_end.isoformat()} page {page}"
        )
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as resp:
            payload = json.loads(resp.read().decode())
        events.extend(payload["results"])
        url = payload.get("next")
        page += 1
    return events


def main():
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json",
    }
    start_date = datetime.fromisoformat(START_GTE).date()
    end_date = datetime.fromisoformat(START_LTE).date()
    all_events = []
    events_by_id = {}
    for window_start, window_end in month_windows(start_date, end_date):
        window_events = fetch_window_events(headers, window_start, window_end)
        print(
            f"window events: {len(window_events)}"
        )
        all_events.extend(window_events)
        for event in window_events:
            events_by_id[event["id"]] = event
    deduped_events = list(events_by_id.values())
    print(f"raw events: {len(all_events)}")
    print(f"unique events: {len(deduped_events)}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fp:
        json.dump(deduped_events, fp)


if __name__ == "__main__":
    main()
