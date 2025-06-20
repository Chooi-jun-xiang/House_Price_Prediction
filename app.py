from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, validator
import pandas as pd
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
import os

# Load model and metadata
trained_model = joblib.load('trained_model.joblib')
model_info = joblib.load('model_info.joblib')

# Load the scaler for car parks and size num
scaler_selected = joblib.load('scaler_selected.joblib')
data_min = scaler_selected.data_min_
data_max = scaler_selected.data_max_

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

class HousePriceInput(BaseModel):
    car_parks: int
    size_num: int
    location: str
    furnishing: str

    @validator('car_parks', 'size_num')
    def validate_positive(cls, value):
        if value <= 0:
            raise ValueError("Value must be positive")
        return value

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

def preprocess_raw_inputs(car_parks, size_num):

    """Normalize raw inputs to match training normalization using Min-Max scaling."""
    normalized_car_parks = (car_parks - data_min[0]) / (data_max[0] - data_min[0])
    normalized_size_num = (size_num - data_min[1]) / (data_max[1] - data_min[1])
    return normalized_car_parks, normalized_size_num

@app.post("/predict/")
async def predict_price(input_data: HousePriceInput):
    try:
        # Preprocess raw inputs
        normalized_car_parks, normalized_size_num = preprocess_raw_inputs(
            input_data.car_parks, input_data.size_num
        )
        print(f"Normalized inputs: car_parks={normalized_car_parks}, size_num={normalized_size_num}")

        # Prepare feature vector for the model
        columns_order = model_info['feature_columns']
        data = {col: [0] for col in columns_order}
        data['Car Parks'] = [normalized_car_parks]
        data['Size Num'] = [normalized_size_num]

        # Handle one-hot encoding for categorical features
        location_col = f"Location_{input_data.location.lower()}"
        furnishing_col = f"Furnishing_{input_data.furnishing}"
        if location_col in data:
            data[location_col] = [1]
        if furnishing_col in data:
            data[furnishing_col] = [1]

        # Create DataFrame for prediction
        df = pd.DataFrame(data)[columns_order]
        print(f"Prepared DataFrame:\n{df}")

        # Predict using the trained model
        prediction = trained_model.best_estimator_.predict(df)
        predicted_price = np.expm1(prediction[0])  # Reverse log transformation
        return {"predicted_price": round(predicted_price, 2)}

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

