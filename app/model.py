import joblib
import numpy as np

model_path = "model/model.pkl"

class MLModel:
    def __init__(self):
        self.model = joblib.load(model_path)
    
    def predict(self, features):
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]
        return prediction