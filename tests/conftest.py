#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Shared pytest fixtures for API tests
"""

# Imports
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from main import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model-categorical.ubj"
STATION_CATEGORIES_PATH = ARTIFACTS_DIR / "station-categories.json"


@pytest.fixture
def client():
    if not MODEL_PATH.exists() or not STATION_CATEGORIES_PATH.exists():
        pytest.skip(
            "Trained model artifacts required (artifacts/model-categorical.ubj and station-categories.json)"
        )
    try:
        import xgboost  # noqa: F401
    except Exception as exc:
        pytest.skip(f"XGBoost not loadable (needed for inference): {exc}")
    return TestClient(create_app())
