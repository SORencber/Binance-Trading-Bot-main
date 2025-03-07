# advanced_btc_predictor.py
import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta
import numpy as np      # Import numpy


from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from flask import Flask, jsonify, render_template
from datetime import datetime, timedelta
import joblib
import pytz
import ccxt
import logging
import requests
import os
import hashlib
import schedule
import time
import threading
import plotly.graph_objs as go
from dotenv import load_dotenv
from keras_tuner import BayesianOptimization
from keras.losses import Huber
from keras.layers import Input,LSTM, Dense, Dropout, Bidirectional, Attention,GlobalMaxPooling1D
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from fred import Fred

# FRED API Anahtarınız
FRED_API_KEY = 'feb4158a5ee846436df1884bd52f88d0'
fred = Fred(api_key=FRED_API_KEY)
# Ortam değişkenleri
load_dotenv()
logging.basicConfig(level=logging.INFO)

class QuantumPredictor:
    def __init__(self):
        self.model_path = 'quantum_btc_v1.h5'
        self.scaler_path = 'quantum_scaler_v1.pkl'
        self.data_cache = 'btc_quantum_data.parquet'
        self.window_size = 90
        self.future_steps = 7
        self.exchange = ccxt.binance({
            'apiKey': '4scum9hMsb1CCNINJofhjb2SaXOLbWe0sXWzLynvIst3FPTPtW0ROBQLdBj1KTL1',
            'secret': 'vGVFP4O75gYGJcI5LTrvGuovS29BLq9Ib3HcPneWfqM2GH1hNWm4orA2uQvI9edt',
            'enableRateLimit': True
        })

        # Model initialization
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            self.model = None
            self.scaler = RobustScaler()
   
    def _get_hybrid_data(self,symbol):
        try:
            # Doğrudan kullanıcının verdiği data.csv'yi oku
            data = pd.read_csv(f"{symbol}_data.csv", parse_dates=["timestamp"])
            data.sort_values("timestamp", inplace=True)
            data = data.ffill().bfill()  # Eksik verileri doldur
            data = data[data['close'] > 0]  # Geçersiz değerleri temizle
            # Geçmiş bir tarih aralığı için tahmin yap
            # Eğitimde kullanılmayacak sütunları çıkar (örneğin 'date')
            if 'date' in data.columns:
                data.drop('date', axis=1, inplace=True)
            
            return data
        except Exception as e:
            logging.error(f"Error in _get_hybrid_data: {e}")
            return pd.DataFrame()
    def _create_sequences(self, data):
            """Advanced feature engineering"""
            X, y = [], []
            for i in range(len(data)-self.window_size-self.future_steps):
                seq = data.iloc[i:i+self.window_size].values
                target = data.iloc[i+self.window_size:i+self.window_size+self.future_steps]['close'].values

                X.append(seq)
                y.append(target)
            return np.array(X), np.array(y)

    def _build_quantum_model(self, hp):
            """Self-optimizing neural architecture"""
            inputs = Input(shape=(self.window_size, 52))
            x = Bidirectional(LSTM(hp.Int('units', 64, 256, step=32), return_sequences=True))(inputs)
            x = Attention()([x, x])
            x = GlobalMaxPooling1D()(x)  # Critical fix: Collapse time dimension
            x = Dropout(hp.Float('dropout', 0.2, 0.5))(x)
            x = Dense(hp.Int('dense_units', 32, 128), activation='selu')(x)
            outputs = Dense(self.future_steps)(x)

            model = Model(inputs, outputs)
            model.compile(
                optimizer=Adam(hp.Float('lr', 1e-4, 1e-2)),
                loss=Huber(),
                metrics=['mape']
            )
            return model

    def automated_training(self,symbol):
        try:
            data = self._get_hybrid_data(symbol)
            
            for column in data.columns:
                try:
                    data[column] = data[column].astype(float)  # Sayıya çevirmeyi dene
                except Exception as e:
                    print(f"Hata veren sütun: {column}, Hata: {e}")

            # 'timestamp' sütununu datetime formatına dönüştür
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

            # Timestamp'i sayısal verilere çevirelim (UNIX zaman damgası)
            data['timestamp'] = data['timestamp'].astype('int64') // 10**9  # Bu işlem saniye cinsinden UNIX zaman damgası oluşturur

            if data.empty:
                logging.error("No data available for training.")
                return
            csv_path = f"{symbol}_data.csv"
            data.to_csv(csv_path, index=False)
            # Veriyi ölçeklendir
            scaled_data = self.scaler.fit_transform(data)

            # Sequence'ler oluştur
            X, y = self._create_sequences(pd.DataFrame(scaled_data, columns=data.columns))
            if X.size == 0 or y.size == 0:
                logging.error("No sequences created for training.")
                return

            # Modeli eğit
            tuner = BayesianOptimization(
                self._build_quantum_model,
                objective='val_loss',
                max_trials=20,
                executions_per_trial=2,
                directory='tuner_logs',
                max_consecutive_failed_trials=10
            )

            tuner.search(X, y, epochs=100, validation_split=0.2, callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ModelCheckpoint(self.model_path, save_best_only=True)
            ])

            self.model = tuner.get_best_models(num_models=1)[0]
            joblib.dump(self.scaler, self.scaler_path)
            logging.info("Model upgrade completed with quantum optimization")
        except Exception as e:
            logging.error(f"Error in automated_training: {e}")
    def live_prediction(self,symbol):
        """Real-time prediction engine"""
        raw_data = self._get_hybrid_data(symbol).iloc[-self.window_size:]
        scaled_data = self.scaler.transform(raw_data)

        prediction = self.model.predict(scaled_data[np.newaxis, ...])
        prediction = prediction.flatten()  # (7,)
              # Sadece 'close' fiyatı için scaler kullan
        scaled_close = self.scaler.fit_transform(raw_data[['close']])
        inv_prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()

        return inv_prediction
    def backtest_historical(self,symbol, start_date, end_date):
            """Geçmiş bir tarih aralığı için tahmin yapar ve gerçek değerlerle karşılaştırır."""
            try:
                data = self._get_hybrid_data(symbol).iloc[-20:]
                #print(data)
                
                historical_data = data
                
                  # Sadece sayısal sütunları seç
                numeric_cols = historical_data.select_dtypes(include=np.number).columns.tolist()
                scaled_data = self.scaler.transform(historical_data[numeric_cols])
        
                # Tahminleri yap
                predictions = self.model.predict(scaled_data)
                
                # Tahminleri ters ölçeklendir
                dummy = np.zeros((predictions.shape[0], 8))
                dummy[:, 0] = predictions.flatten()
                inv_predictions = self.scaler.inverse_transform(dummy)[:, 0]
                
                # Gerçek değerleri al
                actual_values = historical_data['close'].values[-len(inv_predictions):]
                
                return inv_predictions, actual_values
            except Exception as e:
                logging.error(f"Backtest hatası: {e}")
                return None, None
    def risk_management(self, prediction):
            """Trading risk assessment"""
            volatility = np.std(prediction)
            trend_strength = (prediction[-1] - prediction[0]) / prediction[0]
            print ("selam",volatility,trend_strength)
            confidence=min(100, max(0, 100 - (volatility * 100)))
            risk_level='High' if volatility > 0.15 else 'Medium' if volatility > 0.1 else 'Low'
            recommended_action= 'Long' if trend_strength > 0.05 else 'Short' if trend_strength < -0.05 else 'Bekle'
             
            return confidence,risk_level,recommended_action
             # return {
            #     'confidence': min(100, max(0, 100 - (volatility * 100))),
            #     'risk_level': 'High' if volatility > 0.15 else 'Medium' if volatility > 0.1 else 'Low',
            #     'recommended_action': 'Long' if trend_strength > 0.05 else 'Short' if trend_strength < -0.05 else 'Hold'
            # }

# Flask API Engine
app = Flask(__name__)
predictor = QuantumPredictor()

@app.route('/quantum/api/v1/forecast')
def get_forecast():
    prediction = predictor.live_prediction()
    risk = predictor.risk_management(prediction)
    timestamps = [datetime.now(pytz.utc) + timedelta(days=i) for i in range(1, 8)]

    return jsonify({
        'prediction': prediction.tolist(),
        'timestamps': [ts.isoformat() for ts in timestamps],
        'risk_analysis': risk,
        'model_version': hashlib.md5(open(predictor.model_path, 'rb').read()).hexdigest(),
        'last_trained': datetime.fromtimestamp(os.path.getmtime(predictor.model_path)).isoformat()
    })

@app.route('/dashboard')
def trading_dashboard():
    return render_template('quantum_dashboard.html')
import schedule
import time
import threading

# Automated Training Scheduler
def training_job():
    logging.info("Initiating automated training cycle")
    try:
        predictor.automated_training()
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")

schedule.every().day.at("04:00").do(training_job)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == '__main__':
    # Model dosyası yoksa ilk eğitimi 
    symbol="BTCUSDT"
    if not os.path.exists(predictor.model_path):
        predictor.automated_training(symbol)

    # Flask başlamadan önce bugünkü tahmini al ve logla
    try:
       # Geçmiş tahminleri çalıştır
        predictions, actual = predictor.backtest_historical(symbol,'2023-01-01', '2023-12-31')
        if predictions is not None:
            error = np.mean(np.abs(predictions - actual) / actual) * 100
            print(f"Geçmiş Tahmin Hata Oranı (MAPE): {error:.2f}%")
            
    #    a = predictor._get_hybrid_data()

    #    real_values = a[-52:]  # Son 7 günün gerçek verisi
    #    predicted_values = predictor.live_prediction()
    #    error = np.mean(np.abs(real_values - predicted_values) / real_values) * 100  # MAPE
    #    print(f"Model hata oranı: {error:.2f}%")
        
       #todays_prediction = predictor.live_prediction()
        prediction = predictor.live_prediction(symbol)
        confidence,risk_level,recommended_action = predictor.risk_management(prediction)
        timestamps = [datetime.now(pytz.utc) + timedelta(days=i) for i in range(1, 8)]

        
        logging.info("7 Günlük Tahmin: %s", prediction)
        logging.info("Güven Testi: %s", confidence)
        logging.info("Risk Seviyesi: %s", risk_level)
        logging.info("Önerilen Pozisyon: %s", recommended_action)


       #logging.info("Son Güncelleme Tarih: %", datetime.fromtimestamp(os.path.getmtime(predictor.model_path)).isoformat())


    except Exception as e:
        logging.error("Bugünkü tahmin alınırken hata oluştu: %s", e)

    # Arka planda eğitim zamanlayıcısını başlat
    threading.Thread(target=run_scheduler).start()

    # Flask sunucusunu başlat
   # app.run(host='0.0.0.0', port=8000, threaded=True)
