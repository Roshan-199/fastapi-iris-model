import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="My first FastAPI")

model = joblib.load("model.pkl")

@app.get("/")
def home():
    return {"message":"API is running"}

@app.get("/health")
def health():
    return {"status":"up"}

class InputData(BaseModel):
    features: list[float]

@app.post("/get_prediction")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction":int(prediction[0])}