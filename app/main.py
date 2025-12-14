from fastapi import FastAPI, HTTPException
from app.schema import PredictRequest, PredictResponse
from app.logger import logging
from app.model import MLModel

app = FastAPI(title="Production ready model")

model = MLModel()

@app.get("/health")
def health():
    return {"status":"up"}

@app.post("/get_prediction", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        logging.info(f"Received data {request.features}")
        prediction = model.predict(request.features)
        return {"prediction":prediction}
    except Exception as e:
        logging.error(str(e))
        return HTTPException(status_code=500, detail="Prediction Failed")