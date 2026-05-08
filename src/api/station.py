#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: HTTP handlers for station mapping data
"""

# Imports
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

# Constants
REPO_ROOT = Path(__file__).resolve().parents[2]
STATION_MAPPING_PATH = REPO_ROOT / "artifacts" / "station-mapping.json"
router = APIRouter()


@router.get("/stations")
def stations():
    if not STATION_MAPPING_PATH.is_file():
        raise HTTPException(status_code=404, detail="station-mapping.json not found")
    return FileResponse(STATION_MAPPING_PATH, media_type="application/json")
