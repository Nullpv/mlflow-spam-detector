from pathlib import Path
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "logistic_regression.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Spam Detection API",
    version="1.0",
    description="Machine Learning Spam Detection using FastAPI"
)


class SMS(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "Spam Detection API is running."
    }


@app.post("/predict")
def predict(sms: SMS):

    prediction = model.predict([sms.message])[0]

    result = "SPAM" if prediction == 1 else "HAM"

    return {
        "message": sms.message,
        "prediction": result
    }

