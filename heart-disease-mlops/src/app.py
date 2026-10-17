import joblib
import pandas as pd
import logging

from fastapi import FastAPI
from src.schemas import PatientData

# Logging Configuration
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

# Monitoring Variables
request_count = 0
positive_predictions = 0
negative_predictions = 0

# Load Champion Model
model = joblib.load(
    "models/champion_model.pkl"
)

@app.get("/")
def home():
    return {
        "message": "Heart Disease Predictor API"
    }

@app.post("/predict")
def predict(data: PatientData):

    global request_count
    global positive_predictions
    global negative_predictions

    request_count += 1

    df = pd.DataFrame([data.dict()])

    prediction = int(
        model.predict(df)[0]
    )

    probability = float(
        model.predict_proba(df)[0][1]
    )

    if prediction == 1:
        positive_predictions += 1
    else:
        negative_predictions += 1

    logging.info(
        f"Request={request_count}, Prediction={prediction}, Probability={round(probability,4)}"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4)
    }

@app.get("/metrics")
def metrics():

    return {
        "total_requests": request_count,
        "positive_predictions": positive_predictions,
        "negative_predictions": negative_predictions
    }