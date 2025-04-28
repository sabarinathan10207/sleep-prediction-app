from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the pre-trained model
model = pickle.load(open('sleep_model.pkl', 'rb'))

# Create a FastAPI instance
app = FastAPI()

# Define the input data model
class SleepPredictionInput(BaseModel):
    hours_worked: int
    stress_level: int
    caffeine_intake_mg: int
    physical_activity_min: int

# Create a prediction endpoint
@app.post("/predict/")
def predict_sleep(input_data: SleepPredictionInput):
    data = [[input_data.hours_worked, input_data.stress_level, input_data.caffeine_intake_mg, input_data.physical_activity_min]]
    prediction = model.predict(data)
    return {"predicted_sleep_hours": prediction[0]}
