import json
import logging
import os
import warnings

import pandas as pd
import requests
from dotenv import load_dotenv

# Set the env variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def test_sales_prediction(url: str, n):
    """
    Sends a POST request to a prediction API with a batch of sales data and prints the response.

    Args:
        url (str, optional): URL of the prediction endpoint. Defaults to the environment variable "URL_FILTERED_PREDICT".
        n (int, optional): Number of samples from the dataset to include in the request. Defaults to 3.

    Returns:
        None. Prints the response status and body text from the API request.

    """
    # Header
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

    # Load unseen data
    features = eval(os.getenv("SALES_COLUMN_SELECTION"))  # noqa: S307
    features.remove("price")
    unseen_data_path = os.getenv("UNSEEN_DATA")
    unseen_data = pd.read_csv(unseen_data_path, dtype={"zipcode": str})
    sales_data = unseen_data[features]

    # Json
    data_string = sales_data[:n].to_json(orient="records", force_ascii=True)
    data_load = json.loads(data_string)

    batch = {"houses": data_load}

    r = requests.post(url, json=batch, headers=headers)  # noqa: S113

    return print(
        f"Status Code:{r.status_code}, Predictions:{r.text}, Response-time: {r.headers['response-time']}",
    )


def test_unseen_prediction(url: str, n):
    """
    Sends a POST request to a prediction API using raw unseen data and prints the response.

    Args:
        url (str): URL of the prediction endpoint.
        n (int): Number of samples from the unseen dataset to send in the request.

    Returns:
        None. Prints the HTTP response status and body text returned by the API.

    """
    # Header
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

    unseen_data_path = os.getenv("UNSEEN_DATA")
    unseen_data = pd.read_csv(unseen_data_path, dtype={"zipcode": str})

    # Json
    data_string = unseen_data[:n].to_json(orient="records", force_ascii=True)
    data_load = json.loads(data_string)

    batch = {"houses": data_load}

    # Request
    r = requests.post(url, json=batch, headers=headers)  # noqa: S113
    return print(
        f"Status Code:{r.status_code}, Predictions:{r.text}, Response-time: {r.headers['response-time']}",
    )


if __name__ == "__main__":
    # Supress warnings
    warnings.filterwarnings("ignore")

    # Requests
    logger.info("Sales Request")
    test_sales_prediction(os.getenv("URL_SALES_PREDICT"), 3)

    logger.info("Unseen Request")
    test_unseen_prediction(os.getenv("URL_UNSEEN_PREDICT"), 3)
