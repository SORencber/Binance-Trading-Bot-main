# ml_model.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
# Log örneği
try:
    from core.logging_setup import log
except ImportError:
    def log(msg, level="info"):
        print(f"[{level.upper()}] {msg}")
##############################################################################
# 4.1) ML MODEL
##############################################################################
class PatternEnsembleModel:
    """
    Örnek ML modeli (RandomForest) pipeline (v2).
    """
    def __init__(self, model_path:str=None):
        self.model_path = model_path
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier())
        ])
        self.is_fitted = False

    def fit(self, X, y):
        self.pipe.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.pipe.predict(X)

    def extract_features(self, wave):
        """
        Wave yapısından basit feature çıkarma. 
        (Pivot sayısı, son pivot tip, son pivot fiyatı, amplitude, vs.)
        """
        n= len(wave)
        if n<2:
            return np.zeros((1,5))
        last= wave[-1]
        second= wave[-2]
        maxi= max([w[1] for w in wave])
        mini= min([w[1] for w in wave])
        amp= maxi- mini
        arr= [n, last[2], last[1], second[2], amp]
        return np.array([arr])

    def save(self):
        if self.model_path:
            joblib.dump(self.pipe, self.model_path)
            log(f"Model saved to {self.model_path}","info")

    def load(self):
        if self.model_path and os.path.exists(self.model_path):
            self.pipe= joblib.load(self.model_path)
            self.is_fitted= True
            log(f"Model loaded from {self.model_path}","info")
