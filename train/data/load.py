#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Load Bay Wheels trip data URLs
"""

# Imports
import io
import os
import urllib.request
import zipfile
from pathlib import Path
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Constants
BASE_URL = "https://s3.amazonaws.com/baywheels-data"
SKIP_MONTH_CODES = frozenset({"202004"})
PLAIN_TRIPDATA_ZIP_MONTH_CODES = frozenset({"202510", "202601"})
LAST_FORDGO_MONTH_CODE = "201904"
LAST_MONTH_CODE = "202603"


def load():
    def append_row(rows, file_name):
        csv_name = file_name.removesuffix(".zip")
        if not csv_name.endswith(".csv"):
            csv_name = f"{csv_name}.csv"
        rows.append((file_name, csv_name))

    rows = []
    append_row(rows, "2017-fordgobike-tripdata.csv.zip")

    year, month = 2018, 1
    while True:
        yyyymm = f"{year}{month:02d}"
        if yyyymm > LAST_MONTH_CODE:
            break

        if yyyymm not in SKIP_MONTH_CODES:
            brand = "fordgobike" if yyyymm <= LAST_FORDGO_MONTH_CODE else "baywheels"
            trip_suffix = (
                "tripdata.zip"
                if yyyymm in PLAIN_TRIPDATA_ZIP_MONTH_CODES
                else "tripdata.csv.zip"
            )
            append_row(rows, f"{yyyymm}-{brand}-{trip_suffix}")

        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1

    return rows


def read_from_s3(entry):
    file_name, csv_name = entry
    url = f"{BASE_URL}/{file_name}"

    with urllib.request.urlopen(url) as response:
        blob = response.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        hits = [
            name
            for name in archive.namelist()
            if name.rstrip("/").rsplit("/", 1)[-1] == csv_name
        ]
        if len(hits) != 1:
            raise ValueError(
                f"Expected one member {csv_name!r} for {file_name!r}, got {hits or 'none'} in {url!r}"
            )
        raw = archive.read(hits[0])

    return pd.read_csv(io.BytesIO(raw), index_col=0)

def upload_to_s3(data: pd.DataFrame, entry):
    csv_name = entry[1]
    bucket = os.environ["S3_BUCKET"]
    key = f"training/{csv_name}"

    buf = io.BytesIO(data.to_csv(index=False).encode("utf-8"))
    buf.seek(0)
    boto3.client("s3").upload_fileobj(buf, bucket, key)