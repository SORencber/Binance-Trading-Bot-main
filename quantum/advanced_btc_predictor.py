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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: Tüm loglar, 1: Bilgi loglarını gizle, 2: Uyarıları gizle, 3: Hataları gizle
class QuantumPredictor:
    def __init__(self):
        self.model_path = 'vquantum_btc_v1.h5'
        self.scaler_path = 'vquantum_scaler_v1.pkl'
        self.data_cache = 'bbtc_quantum_data.parquet'
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
            
            # Modeli derle
            self.model.compile(optimizer='adam', loss='mse', metrics=['mape'])
        else:
            self.model = None
            self.scaler = RobustScaler()
   
    # FRED'den veri çekme fonksiyonu
    def fetch_fred_data(self,series_id,start_date,end_date):
        url = f"https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors='coerce')
            df = df.drop('realtime_end', axis=1)
            df = df.drop('realtime_start', axis=1)
            #p(series_id,df)
            return df
        else:
            print(f"Hata oluştu: {data}")
            return None
    def _get_macroeconomic_data(self):
        """Enflasyon, altın, dolar kuru ve diğer makro verileri alır"""
        try:
            # Fear & Greed Index verisini al ve indeksi düzelt
            url = "https://api.alternative.me/fng/?limit=0"
            response = requests.get(url)
            data = response.json()
            
            fear_greed = pd.DataFrame(data["data"])
            fear_greed["timestamp"] = pd.to_datetime(pd.to_numeric(fear_greed["timestamp"]), unit="s")
            fear_greed = fear_greed.rename(columns={"value": "Fear_Greed"})
            fear_greed = fear_greed.set_index("timestamp")[["Fear_Greed"]]
            fear_greed.index = fear_greed.index.normalize()  # Saat bilgisini kaldır

            # FRED verilerini çek ve işle
            SERIES_IDS = {
                "GOLD": "GVZCLS",         # Altın fiyatı
                "GS10": "GS10",          # 10 Yıllık Faiz
                "OIL": "DCOILWTICO",     # Petrol fiyatı (WTI)
                "CPI": "CPIAUCSL",       # Tüketici Fiyat Endeksi (CPI)
                "M2": "M2SL",            # Para arzı
                "UNRATE": "UNRATE",      # İşsizlik oranı
                "SP500": "SP500",        # S&P 500 Endeksi
                "DXY": "DTWEXBGS",       # Dolar Endeksi (DXY)
                "VIX": "VIXCLS",         # VIX Endeksi (Volatilite Endeksi)
                "RETAIL": "RSXFS",       # Perakende Satışlar
                "GDP": "GDP",            # GSYİH
                "T10Y2Y": "T10Y2Y",      # 10Y-2Y Faiz Farkı (inverted yield curve)
                "BALANCE_OF_TRADE": "NETEXP", # Dış Ticaret Dengesi
                "PPI": "PPIACO",         # Üretici Fiyat Endeksi
                "M1": "M1SL",            # M1 Para Arzı
                "IP": "INDPRO",          # Sanayi Üretimi
               # "STLFSI": "STLFSI",      # Finansal Durum Endeksi (Financial Stress Index)
            }

            # Tarih aralığını ayarlayalım
            end_date = datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.today() - timedelta(days=5*365)).strftime('%Y-%m-%d')

            dfs = []
            for name, series_id in SERIES_IDS.items():
                df = self.fetch_fred_data(series_id, start_date, end_date)
                if df is not None:
                    df.index = pd.to_datetime(df['date'])
                    df = df.rename(columns={"value": name})
                    
                    # Aylık verileri günlük frekansa dönüştür
                    if name in ["GS10", "CPI", "M2", "UNRATE", "PPI", "RETAIL", "GDP", "T10Y2Y"]:
                        df = df.resample('D').ffill()  # Ayın ilk değerini tüm ay boyunca yay

                    dfs.append(df[[name]])  # Sadece ilgili sütunu al

            # Tüm verileri birleştir
            macro_df = fear_greed.copy()
            for df in dfs:
                macro_df = macro_df.join(df, how='outer')  # İndeksler otomatik hizalanır

            # Eksik verileri doldur ve son 5 yılı filtrele
            macro_df = macro_df.ffill().bfill()
            macro_df = macro_df.loc[macro_df.index >= (datetime.today() - timedelta(days=5*365))]

            #print(macro_df)
            return macro_df

        except Exception as e:
            logging.error(f"Error in _get_macroeconomic_data: {e}", exc_info=True)
            return pd.DataFrame()

    def _get_hybrid_data(self,symbol):
        """Bitcoin fiyatlarını ve makro verileri birleştirir"""
        try:
            csv_path = f"../advanced_bot2/data_storage/{symbol}_1d.csv"
            if not os.path.exists(csv_path):
               print("dosya yok")

            # CSV oku
            btc = pd.read_csv(csv_path, parse_dates=["timestamp"])
            btc.sort_values("timestamp", inplace=True)
            btc.reset_index(drop=True, inplace=True)
            btc.columns = [col.lower() for col in btc.columns]
            #print(btc.columns)
            # 📌 **2. Makroekonomik verileri al ve tarihleri eşleştir**
           
            # 📌 **4. Eksik verileri doldur**
            #btc = btc.ffill().bfill()

            # 📌 **5. Teknik göstergeleri hesapla**
            high = btc["high"]
            low  = btc["low"]
            close= btc["close"]
            vol  = btc["volume"]
            open_= btc["open"]
            btc["RSI"] = ta.RSI(close, 14)
            rsi_=btc["RSI"]
            btc["StochRSI"] = (rsi_ - rsi_.rolling(14).min()) / (rsi_.rolling(14).max() - rsi_.rolling(14).min() + 1e-9)

            btc['OBV'] = ta.OBV(close, vol)
            btc['ATR'] = ta.ATR(high, low, close)
            btc['ADX'] = ta.ADX(high, low, close)
            macd, signal, hist = ta.MACD(close)  # Üçüncü değeri de al
            btc['MACD'] = macd
            btc['MACD_Signal'] = signal  # Yeni özellik eklendi
            btc['MACD_Hist'] = hist  # Yeni özellik eklendi
            btc['SMA_50'] = ta.SMA(btc['close'], timeperiod=50)
            btc['SMA_20'] = ta.SMA(btc['close'], timeperiod=20)
            btc['DI_plus'] = ta.PLUS_DI(btc['high'], btc['low'], btc['close'], timeperiod=14)
            btc['DI_minus'] = ta.MINUS_DI(btc['high'], btc['low'], btc['close'], timeperiod=14)
             # 13) Momentum, ROC
            btc["MOM_"] = ta.MOM(close, timeperiod=10)
            btc["ROC_"] = ta.ROC(close, timeperiod=10)
            cmf_val = ((2*close - high - low)/(high - low + 1e-9)) * vol
            btc["CMF_"] = cmf_val.rolling(window=20).mean()
            btc["Candle_Body_"] = (close - open_).abs()
            btc["Upper_Wick_"]  = btc[["close","open"]].max(axis=1).combine(btc["high"], lambda x,y: y - x)
            btc["Lower_Wick_"]  = btc[["close","open"]].min(axis=1) - btc["low"]
            btc['target'] = btc['close'].pct_change().rolling(5).std() * 100  # Yüzde volatilite

            btc['Bollinger_Upper'], btc['Bollinger_Middle'], btc['Bollinger_Lower'] = ta.BBANDS(btc['close'])
            # btc["MarketCap_"]   = close*vol
            # btc["RealizedCap_"] = close*(vol.cumsum()/max(1,len(btc)))
            # rolling_std = close.rolling(365).std()
            # btc["MVRV_Z_"] = (btc["MarketCap_"] - btc["RealizedCap_"]) / (rolling_std+1e-9)
            # btc["NVT_"]    = btc["MarketCap_"]/(vol+1e-9)
                
           
            df_macro = self._get_macroeconomic_data()

            # Makro verilerini yükle
            df_macro['date'] = pd.to_datetime(df_macro.index)  # Index'i datetime'a çevir
            btc['timestamp'] = pd.to_datetime(btc['timestamp'])  # Timestamp'i datetime'a çevir

            # İki veri setini birleştir
            df_merged = pd.merge(btc, df_macro, left_on='timestamp', right_on='date', how='left')

            # Eksik makro verileri önceki günün verisiyle doldur
            df_merged.ffill().bfill()      
            #print(btc)     
            return df_merged 
            # btc[['close', 'RSI', 'MACD', 'OBV', 'ATR', 'ADX','MACD','MACD_Signal','MACD_Hist','Bollinger_Upper','Bollinger_Lower','SMA_50','DXY', 'Gold', 'Oil', 'value', 'value_infl']]
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
            #print(f"Initial data shape: {data.shape}")
            #print(data.columns)
            #print(data)
            # 1A. Strategic imputation before dropping rows
            data = data.ffill().bfill()
            
            # 1B. Column-specific imputation
            for col in data.columns:
                if data[col].isna().any():
                    if col in ['Fear_Greed', 'GOLD', 'OIL']:
                        data[col].interpolate(method='time', inplace=True)
                    else:
                        data[col].fillna(data[col].median(), inplace=True)

            # Remove only rows with invalid closes
            data = data[data['close'] > 0]
            print(f"Post-cleaning shape: {data.shape}")
            data = data.drop('date', axis=1)
            #data = data.dropna(subset=['timestamp'])
               # 'timestamp' sütununu datetime formatına dönüştür
            data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
            # Timestamp'i sayısal verilere çevirelim (UNIX zaman damgası)
            data['timestamp'] = data['timestamp'].astype('int64') // 10**9  # Bu işlem saniye cinsinden UNIX zaman damgası oluşturur
            # 'timestamp' sütununu datetime formatına dönüştür
            data['close time'] = pd.to_datetime(data['close time'], errors='coerce')
            # Timestamp'i sayısal verilere çevirelim (UNIX zaman damgası)
            data['close time'] = data['close time'].astype('int64') // 10**9  # Bu işlem saniye cinsinden UNIX zaman damgası oluşturur

            for column in data.columns:
                try:
                    data[column] = data[column].astype(float)  # Sayıya çevirmeyi dene
                except Exception as e:
                    print(f"Hata veren sütun: {column}, Hata: {e}")

            # 'timestamp' sütununu datetime formatına dönüştür
            #data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')

            # Timestamp'i sayısal verilere çevirelim (UNIX zaman damgası)
            #data['timestamp'] = data['timestamp'].astype('int64') // 10**9  # Bu işlem saniye cinsinden UNIX zaman damgası oluşturur

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
    def live_prediction(self):
            """Real-time prediction engine"""
        
            raw_data = self._get_hybrid_data().iloc[-self.window_size:]
            scaled_data = self.scaler.transform(raw_data)
            prediction = self.model.predict(scaled_data[np.newaxis, ...])
            self.model.compile(optimizer="adam", loss="mse", metrics=["mape"])

            # Modelin çıktısı prediction, shape (1, 7)
            dummy = np.zeros((prediction.shape[0], 8))
            print(f"Prediction shape: {prediction.shape}")  # Beklenen çıktı: (1,7)
            print(f"Dummy shape: {dummy.shape}")  # Beklenen çıktı: (1, 52)

            dummy[:, 0] = prediction.reshape(-1)  # close sütunu modelin çıktısı
            inv_prediction = self.scaler.inverse_transform(dummy)[:, 0]
            return inv_prediction
            #return self.scaler.inverse_transform(prediction.reshape(-1, 8))[:,0]

    def risk_management(self, prediction):
            """Trading risk assessment"""
            volatility = np.std(prediction)
            trend_strength = (prediction[-1] - prediction[0]) / prediction[0]
            return {
                'confidence': min(100, max(0, 100 - (volatility * 100))),
                'risk_level': 'High' if volatility > 0.15 else 'Medium' if volatility > 0.1 else 'Low',
                'recommended_action': 'Long' if trend_strength > 0.05 else 'Short' if trend_strength < -0.05 else 'Hold'
            }

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
    # Model dosyası yoksa ilk eğitimi yap
    symbol="BTCUSDT"
    if not os.path.exists(predictor.model_path):
        predictor.automated_training(symbol)

    # Flask başlamadan önce bugünkü tahmini al ve logla
    #try:
       # todays_prediction = predictor.live_prediction()
       # logging.info("Bugünkü Tahmin: %s", todays_prediction)
    #except Exception as e:
        #logging.error("Bugünkü tahmin alınırken hata oluştu: %s", e)

    # Arka planda eğitim zamanlayıcısını başlat
    #threading.Thread(target=run_scheduler).start()

    # Flask sunucusunu başlat
   # app.run(host='0.0.0.0', port=5000, threaded=True)
