import logging
import os
import pickle
import time
import warnings
from typing import List  # noqa: UP035

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

# Instantiate Envs
APP_MODEL_PATH = os.environ["APP_MODEL_PATH"]
DEMOGRAPHICS_PATH = os.environ["APP_DEMOGRAPHICS_PATH"]
FEATURES = os.environ["FEATURES"]

# Configure logging
logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()


# Bonus: Define the expected payload structure
class SalesFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: int


class SalesBatchInput(BaseModel):
    houses: List[SalesFeatures]  # noqa: UP006


class UnseenFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


# For a list of these items:
class UnseenBatchInput(BaseModel):
    houses: List[UnseenFeatures]  # noqa: UP006


# Load the model
with open(APP_MODEL_PATH, "rb") as model_file:  # noqa: PTH123
    MODEL = pickle.load(model_file)  # noqa: S301


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):  # noqa: D417
    """
    Middleware that measures the processing time of each HTTP request and adds it to the response headers.

    Parameters
        request (Request): The incoming HTTP request object.
        call_next (function): A function that receives the request and returns a response.

    Returns
        Response: The HTTP response with an added "Response-time" header indicating
        the time (in seconds) taken to process the request.

    """  # noqa: D407
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = time.perf_counter() - start_time

    response.headers["Response-time"] = str(process_time)
    return response


# Endpoints
@app.post("/sales_prediction/")
async def filtered_prediction(payload: SalesBatchInput):
    """Handles the inference pipeline."""
    try:
        logging.info("Request received!!!")  # noqa: LOG015

        # Incoming data
        message_data = pd.DataFrame()

        for i, _ in enumerate(payload.houses):
            temp_data = pd.DataFrame([payload.houses[i].model_dump()])
            message_data = pd.concat([message_data, temp_data], ignore_index=True)

        # Demographics data
        demo_data = pd.read_csv(DEMOGRAPHICS_PATH)

        # Merging
        merged_data = message_data.merge(demo_data, how="left", on="zipcode").drop(
            columns="zipcode",
        )

        # Model features
        features = pd.read_json(FEATURES)
        features = features[0].to_list()

        # Make predictions on payload
        pred = MODEL.predict(merged_data[features])

        logger.info(pred)

        # prettifying the predictions
        response = {}
        for i in range(len(pred)):
            response[f"ZipCode-{message_data.iloc[i]['zipcode']}"] = f"${pred[i]}"

        return response  # noqa: TRY300

    except Exception as e:
        logger.exception(f"Processing failed: {e!s}")  # noqa: G004, TRY401
        raise HTTPException(
            status_code=500,
            detail="An error occured during processing",
        ) from e


@app.post("/unseen_predict/")
async def unfiltered_prediction(payload: UnseenBatchInput):
    """Handles the inference pipeline."""
    try:
        logger.info("Request received!!!")

        # Incoming data
        message_data = pd.DataFrame()

        for i, _ in enumerate(payload.houses):
            logger.info(i)

            temp_data = pd.DataFrame([payload.houses[i].model_dump()])
            message_data = pd.concat([message_data, temp_data], ignore_index=True)

        # Demographics data
        demo_data = pd.read_csv(DEMOGRAPHICS_PATH)

        # Merging
        merged_data = message_data.merge(demo_data, how="left", on="zipcode").drop(
            columns="zipcode",
        )

        # Model features
        features = pd.read_json(FEATURES)
        features = features[0].to_list()

        # Make predictions on payload
        pred = MODEL.predict(merged_data[features])

        logger.info(pred)

        # prettifying the predictions
        response = {}
        for i in range(len(pred)):
            response[f"ZipCode-{message_data.iloc[i]['zipcode']}"] = f"${pred[i]}"

        return response  # noqa: TRY300

    except Exception as e:
        logger.exception(f"Processing failed: {e!s}")  # noqa: G004, TRY401
        raise HTTPException(
            status_code=500,
            detail="An error occured during processing",
        ) from e


if __name__ == "__main__":
    # Supress warnings
    warnings.filterwarnings("ignore")

    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8080)  # noqa: S104
