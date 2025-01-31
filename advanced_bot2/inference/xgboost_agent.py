# inference/xgboost_agent.py

import numpy as np
import joblib

class XGBoostAgent:
    def __init__(self, model_path="xgboost_model.pkl"):
        self.model_path= model_path
        self.model= None
        self.loaded= False

    def load_model(self):
        try:
            self.model= joblib.load(self.model_path)
            self.loaded= True
        except Exception as e:
            print(f"[XGBoostAgent] load_model => {e}")
            self.loaded= False

    def predict_signal(self, features: np.ndarray)-> int:
        """
        features => shape=(1,n_features)
        return => 0=SELL, 1=BUY
        """
        if not self.loaded or self.model is None:
            return 0
        pred= self.model.predict(features)
        return int(pred[0])
