#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: HTTP tests for POST /predict (net_flow regression)
"""


def assert_prediction_payload(response, low, high):
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"prediction"}
    value = float(body["prediction"])
    assert low <= value <= high, f"prediction {value} not in [{low}, {high}]"


def test_predict_outside_lands_exodus(client):
    response = client.post(
        "/predict",
        json={
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
    )
    assert_prediction_payload(response, -95.0, 40.0)


def test_predict_hardly_strictly_arrivals(client):
    response = client.post(
        "/predict",
        json={
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
    )
    assert_prediction_payload(response, -40.0, 70.0)


def test_predict_embarcadero_am_commute(client):
    response = client.post(
        "/predict",
        json={
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
    )
    assert_prediction_payload(response, -25.0, 55.0)


def test_predict_embarcadero_pm_commute(client):
    response = client.post(
        "/predict",
        json={
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
    )
    assert_prediction_payload(response, -55.0, 30.0)


def test_predict_quiet_baseline(client):
    response = client.post(
        "/predict",
        json={
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
    )
    assert_prediction_payload(response, -20.0, 20.0)

def test_predict_commute_soma(client):
    response = client.post(
        "/predict",
        json={
            "day_of_week": 2,
            "time_bucket": 35,
            "is_weekend": False,
            "week_of_year": 11,
            "month": 3,
            "is_us_federal_holiday": False,
            "commute_hours": True,
            "station_id": "SF-F28-3",
            "temperature": 54.8,
            "precipitation": 0.12,
            "wind": 8.3,
        },
    )
    assert_prediction_payload(response, -40.0, 60.0)
