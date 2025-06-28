import os
import json
import pickle
import uvicorn
import logging
import warnings
import pandas as pd
from typing import Dict, List, Union
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Instantiate Envs
APP_MODEL_PATH = os.environ['APP_MODEL_PATH']             #os.environ['APP_MODEL_PATH']
DEMOGRAPHICS_PATH = os.environ['APP_DEMOGRAPHICS_PATH']   #os.environ['APP_DEMOGRAPHICS_PATH']
FEATURES = os.environ['FEATURES']

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Initialize FastAPI app
app = FastAPI()

# Bonus: Define the expected payload structure
class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: int

class HouseBatchInput(BaseModel):
    houses: List[HouseFeatures]

class InputData(BaseModel):
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

# For a list of these items (as in your JSON string):
class InputDataBatch(BaseModel):
    houses: List[InputData]


# Load the model
model_file = open(APP_MODEL_PATH, 'rb')
MODEL = pickle.load(model_file)

# Endpoints
@app.post("/filtered_prediction/")
def filtered_prediction(payload: HouseBatchInput):
    """Handles the inference pipeline"""
    
    try:

        logging.info("Request received!!!")
        
        # Incoming data
        message_data=pd.DataFrame()
        
        for i,_ in enumerate(payload.houses):
            
            temp_data = pd.DataFrame([payload.houses[i].model_dump()])
            message_data = pd.concat([message_data,temp_data], ignore_index=True)

        # Demographics data
        demo_data = pd.read_csv(DEMOGRAPHICS_PATH)
        
        # Merging
        merged_data = message_data.merge(demo_data, how="left",
                                         on="zipcode").drop(columns="zipcode")
        
        #Model features
        features = pd.read_json(FEATURES)
        features = features[0].to_list()

        # Make predictions on payload
        pred = MODEL.predict(merged_data[features])

        logging.info(pred)

        #prettifying the predictions
        response = dict()
        for i in range(len(pred)):
            response[f"ZipCode-{message_data.iloc[i]['zipcode']}"] = f" ${pred[i]}"
        
        return response

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occured during processing") from e
    
@app.post("/unfiltered_predict/")
def unfiltered_prediction(payload: InputDataBatch):
    """Handles the inference pipeline"""
    try:

        logging.info("Request received!!!")

        # Incoming data
        message_data=pd.DataFrame()
        
        for i,_ in enumerate(payload.houses):
            logging.info(i)
            
            temp_data = pd.DataFrame([payload.houses[i].model_dump()])
            message_data = pd.concat([message_data,temp_data], ignore_index=True)
        
        # Demographics data
        demo_data = pd.read_csv(DEMOGRAPHICS_PATH)

        # Merging
        merged_data = message_data.merge(demo_data, how="left",
                                on="zipcode").drop(columns="zipcode")
        
        #Model features
        features = pd.read_json(FEATURES)
        features = features[0].to_list()

        # Make predictions on payload
        pred = MODEL.predict(merged_data[features])

        logging.info(pred)

        #prettifying the predictions
        response = dict()
        for i in range(len(pred)):
            response[f"ZipCode-{message_data.iloc[i]['zipcode']}"] = f" ${pred[i]}"
        
        return response

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occured during processing") from e


if __name__ == "__main__":
    #Supress warnings
    warnings.filterwarnings("ignore")

    #Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8080)