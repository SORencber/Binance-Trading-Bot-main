"""
auto_seq_train.py

Son Optimizasyonlar:
1) Hedef değişken için MinMaxScaler kullanımı
2) Data leakage düzeltmeleri
3) Gelişmiş teknik indikatörler
4) Model mimarisi iyileştirmeleri
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import talib

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Flatten, 
                                     Conv1D, MaxPooling1D, LSTM, Permute, Multiply)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import Huber

from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, TerminateOnNaN)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

try:
    from kerastuner.tuners import RandomSearch
except ImportError:
    import keras_tuner as kt
    RandomSearch = kt.RandomSearch

# 1. VERİ İŞLEME FONKSİYONLARI -------------------------------------------------
def log_return(df, col='close'):
    df['log_return'] = np.log(df[col] / df[col].shift(1))
    return df

def add_technical_indicators(df):
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Temel indikatörler
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'])
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Volatilite
    df['volatility'] = df['close'].rolling(window=5).std()
    
    # Hacim tabanlı
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Log dönüşüm
    df = log_return(df, 'close')
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def remove_outliers(df, z_thresh=2.0):  # Daha agresif filtreleme
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        mean_val = series.mean()
        std_val = series.std() + 1e-9
        z_score = (series - mean_val) / std_val
        df = df[np.abs(z_score) < z_thresh]
    return df.reset_index(drop=True)

def time_based_split(df, train_ratio=0.9, time_col='close time'):  # Daha fazla train verisi
    if time_col in df.columns:
        df.sort_values(by=time_col, inplace=True)
    else:
        df.sort_index(inplace=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def create_sequence_data(X, y, seq_length=24):  # Sequence uzunluğu artırıldı
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# 2. MODEL MİMARİSİ -----------------------------------------------------------
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.time_steps = input_shape[1]
        self.dense = Dense(self.time_steps, activation='softmax')  # Softmax'a geri dön
        
    def call(self, inputs):
        attention = self.dense(inputs)
        return Multiply()([inputs, attention])
    
    def get_config(self):
        return super().get_config()

def build_model(hp, input_shape):
    model = Sequential()
    
    # Ortak hiperparametreler
    l2_reg = regularizers.l2(hp.Choice('l2_reg', [1e-4, 1e-5]))
    dropout_rate = hp.Float('dropout', 0.3, 0.5)
    
    # CNN-LSTM-Attention Hybrid
    model.add(Conv1D(
        filters=hp.Int('filters', 64, 256, step=64),
        kernel_size=hp.Choice('kernel_size', [3,5]),
        activation='linear',
        padding='same',
        input_shape=input_shape
    ))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(dropout_rate))
    
    # Attention
    model.add(AttentionBlock())
    
    # LSTM
    model.add(LSTM(
        hp.Int('lstm_units', 128, 512, step=64),
        return_sequences=False,
        kernel_regularizer=l2_reg
    ))
    model.add(Dropout(dropout_rate))
    
    # Yoğun Katmanlar
    model.add(Dense(hp.Int('dense_units', 128, 512, step=64), activation='linear'))
    model.add(LeakyReLU(alpha=0.01))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Çıkış Katmanı
    model.add(Dense(1, activation='linear'))  # Linear aktivasyon
    
    # Optimizer
    optimizer = AdamW(
        learning_rate=hp.Choice('lr', [1e-4, 5e-5, 1e-5]),
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss=Huber(),
        metrics=['mae']
    )
    return model

# 3. ANA İŞLEM FONKSİYONU ----------------------------------------------------
def auto_run(df, target_col='close', time_col='close time', 
             sequence_length=24, train_ratio=0.9, max_trials=100, 
             executions_per_trial=3, epochs=300, batch_size=32, 
             save_model_name="optimized_model.h5"):
    
    # Veri Hazırlığı
    df = add_technical_indicators(df)
    df = remove_outliers(df, z_thresh=2.0)
    
    train_df, test_df = time_based_split(df, train_ratio, time_col)
    
    # Ölçeklendirme (Eğitim verisi ile fit)
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Hedef için MinMax
    
    X_train = scaler_X.fit_transform(train_df.drop(columns=[target_col]))
    y_train = scaler_y.fit_transform(train_df[[target_col]])
    
    X_test = scaler_X.transform(test_df.drop(columns=[target_col]))
    y_test = scaler_y.transform(test_df[[target_col]])
    
    # Sequence Oluşturma
    X_train_seq, y_train_seq = create_sequence_data(X_train, y_train, sequence_length)
    X_test_seq, y_test_seq = create_sequence_data(X_test, y_test, sequence_length)
    
    # Hiperparametre Optimizasyonu
    tuner = RandomSearch(
        lambda hp: build_model(hp, (sequence_length, X_train.shape[1])),
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='tuner_logs',
        project_name='financial_forecast'
    )
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint('best_temp.h5', save_best_only=True)
    ]
    
    tuner.search(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # En iyi model
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(save_model_name)
    
    # Değerlendirme
    preds = scaler_y.inverse_transform(best_model.predict(X_test_seq))
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, preds)),
        'MAE': mean_absolute_error(y_true, preds),
        'MAPE': mean_absolute_percentage_error(y_true, preds)*100,
        'R2': r2_score(y_true, preds)
    }
    
    return {
        'model': best_model,
        'metrics': metrics,
        'next_prediction': scaler_y.inverse_transform(
            best_model.predict(X_test_seq[-1:].reshape(1, sequence_length, -1))
        )[0][0]
    }

if __name__ == "__main__":
    df = pd.read_csv("BTCUSDT_data.csv")
    results = auto_run(df)
    
    print("\n--- Final Test Sonuçları ---")
    for k, v in results['metrics'].items():
        print(f"{k}: {v:.4f}")
    print(f"Sonraki Tahmin: {results['next_prediction']:.2f}")