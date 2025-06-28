import os
import json
import warnings
import logging
import pandas as pd
import requests
from dotenv import load_dotenv

# Set the env variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)


def test_filtered_prediction(url: str=os.getenv("URL_FILTERED_PREDICT"), n=3):  #os.getenv("URL_HARD_PREDICT")):

    """
    
    """
 
    # Header
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

    # Load unseen data
    features = eval(os.getenv("SALES_COLUMN_SELECTION"))
    features.remove('price')
    unseen_data_path = os.getenv("UNSEEN_DATA")
    unseen_data = pd.read_csv(unseen_data_path)
    unseen_data = unseen_data[features]


    #Json    
    data_string = unseen_data[:n].to_json(orient='records',force_ascii=True)
    data_load = json.loads(data_string)

    batch = {"houses":data_load}

    r = requests.post(url, json=batch, headers=headers)
    return print(r, r.text)
    
    
    #for sample in data_load:
        # Request
    #    r = requests.post(url, json=sample, headers=headers)
    #    return print(r, r.text)

def test_unfiltered_prediction(url: str=os.getenv("URL_UNFILTERED_PREDICT"), n=3):

    """
    
    """
 
    # Header
    headers = {"content-type": "application/json", "Accept-Charset": "UTF-8"}

    
    #data_string = '{"bedrooms":4,"bathrooms":1.0,"sqft_living":1680,"sqft_lot":5043,"floors":1.5,"waterfront":0,"view":0,"condition":4,"grade":6,"sqft_above":1680,"sqft_basement":0,"yr_built":1911,"yr_renovated":0,"zipcode":98118,"lat":47.5354,"long":-122.273,"sqft_living15":1560,"sqft_lot15":5765}'
    unseen_data_path = os.getenv("UNSEEN_DATA")
    unseen_data = pd.read_csv(unseen_data_path)

    #Json    
    data_string = unseen_data[:n].to_json(orient='records',force_ascii=True)
    data_load = json.loads(data_string)

    batch = {"houses":data_load}

    # Request
    r = requests.post(url, json=batch, headers=headers)
    return print(r, r.text)

if __name__ == "__main__":
    #Supress warnings
    warnings.filterwarnings("ignore")

    #Requests
    logging.info("Hard Request")
    test_filtered_prediction()

    logging.info("Soft Request")
    test_unfiltered_prediction()