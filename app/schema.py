from pydantic import BaseModel, Field
from typing import List

class PredictRequest(BaseModel):
    features: List[float] = Field(..., min_items=4, max_items=4)

class PredictResponse(BaseModel):
    prediction: int