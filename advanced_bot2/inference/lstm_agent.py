# inference/lstm_agent.py

import numpy as np
import tensorflow as tf

class LSTMAgent:
    def __init__(self, model_path="lstm_model.h5"):
        self.model_path= model_path
        self.model= None
        self.loaded= False

    def load_model(self):
        try:
            self.model= tf.keras.models.load_model(self.model_path)
            self.loaded= True
        except Exception as e:
            print(f"[LSTMAgent] => {e}")
            self.loaded= False

    def predict_signal(self, features: np.ndarray)-> int:
        """
        features => shape=(1,1, n_features) => ex: (1,1,2)
        return => 0=SELL,1=BUY
        """
        if not self.loaded or self.model is None:
            return 0
        out= self.model.predict(features)
        # out => shape=(1,1)
        return 1 if out[0][0]>0.5 else 0
