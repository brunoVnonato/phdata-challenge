import os
import sys
import pickle
import uvicorn
import logging
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

# Load enviroment variables from .env files
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Initialize FastAPI app
app = FastAPI()

# Define the expected payload structure
class PayloadValidation(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int

# Load the model
model_file = open('../artifacts/model.pkl', 'rb')
MODEL = pickle.load(model_file)

# Endpoints
@app.post("/")
async def get_prediction(payload: PayloadValidation):
    """Handles the inference pipeline"""
    try:

        logging.info("Request received!!!")

        payload = payload.model_dump()

        # Make predictions on payload
        pred = MODEL.predict(payload)

        #prettifying the predictions
        response = dict()
        for i in range(len(pred)):
            response[f"Sample-{i}"] = pred[i]
        
        return response

    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occured during processing") from e


if __name__ == "__main__":
    #Supress warnings
    warnings.filterwarnings("ignore")

    #Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8080)