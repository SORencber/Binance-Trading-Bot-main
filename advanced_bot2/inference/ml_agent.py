# strategy/ml_agent.py
"""
Basit scikit-learn / keras model => predict => up/down
"""
import joblib
import numpy as np

class MLAgent:
    def __init__(self, model_path="model_data/rf_model.pkl"):
        self.model_path= model_path
        self.model= None
    def load_model(self):
        self.model= joblib.load(self.model_path)

    def predict(self, X: np.ndarray):
        if self.model is None:
            return [0]
        return self.model.predict(X)
