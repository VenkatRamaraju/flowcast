#!/usr/bin/env python3
"""
Author: Venkat Ramaraju
Description: Entrypoint for ETL, training, evaluation, and prediction API
"""

# Imports
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    day_of_week: int = Field(..., ge=0, le=6)
    time_bucket: int = Field(..., ge=0)
    is_weekend: bool
    week_of_year: int = Field(..., ge=1, le=53)
    month: int = Field(..., ge=1, le=12)
    is_us_federal_holiday: bool
    commute_hours: bool
    station_id: str
    temperature: float
    precipitation: float
    wind: float


def create_app():
    from src.model.inference import predict_net_flow

    app = FastAPI(title="Flowcast", version="1.0")

    @app.post("/predict")
    def predict_endpoint(body: PredictRequest):
        try:
            prediction = predict_net_flow(**body.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"prediction": prediction}

    return app


def main():
    parser = argparse.ArgumentParser(description="Flowcast ETL and training entrypoint")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--train",
        action="store_true",
        help="Run incremental XGBoost training (S3 mixed CSVs must match src.model.model.MODEL_COLUMNS)",
    )
    mode.add_argument("--eval", action="store_true", help="Evaluate model on held-out eval file")
    mode.add_argument(
        "--data",
        nargs=2,
        type=int,
        metavar=("START", "STOP"),
        help="Run ETL on catalogue slice entries[START:STOP]",
    )
    mode.add_argument(
        "--server",
        action="store_true",
        help="Run FastAPI prediction server (uvicorn on 0.0.0.0:8000)",
    )
    args = parser.parse_args()

    if args.train:
        from src.model.data import MIXED_BUCKET
        from src.model.model import train

        train(MIXED_BUCKET)
        return

    if args.eval:
        from src.model.data import MIXED_BUCKET
        from src.model.eval import eval

        eval(MIXED_BUCKET)
        return

    if args.server:
        import uvicorn

        uvicorn.run(create_app(), host="0.0.0.0", port=8000)
        return

    from src.data.load import load, read_from_s3, upload_to_s3
    from src.data.transform import transform
    from src.model.data import MIXED_BUCKET

    start, stop = args.data
    entries = load()[start:stop]
    for entry in entries:
        print(f"Downloading {entry[0]}")
        data = read_from_s3(entry)
        transformed_data = transform(data)
        upload_to_s3(transformed_data, entry)


if __name__ == "__main__":
    main()
