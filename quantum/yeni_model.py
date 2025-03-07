"""
Geliştirilmiş Otonom Zaman Serisi Modeli
- Daha fazla teknik gösterge
- Gelişmiş veri ön işleme
- Attention katmanı desteği
- Görselleştirme ve detaylı metrikler
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import os
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Flatten, 
                                     Conv1D, MaxPooling1D, LSTM, Permute, Multiply)
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, TerminateOnNaN)
from tensorflow.keras.initializers import HeNormal

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
import keras_tuner as kt

##############################################################################
# 1) Gelişmiş Yardımcı Fonksiyonlar
##############################################################################

def log_return(df, col='close'):
    df['log_return'] = np.log(df[col] / df[col].shift(1))
    return df

def add_technical_indicators(df):
    """15+ teknik gösterge ekler"""
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    
   
    df['AD'] = talib.AD(high, low, close, volume)
    
    # Diğer
    df['CCI'] = talib.CCI(high, low, close, timeperiod=14)
    df['MOM'] = talib.MOM(close, timeperiod=10)
    df['WILLR'] = talib.WILLR(high, low, close, timeperiod=14)
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def remove_outliers(df, method='iqr', threshold=1.5):
    """IQR veya Z-Score ile aykırı değer temizleme"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - threshold*IQR) & (df[col] <= Q3 + threshold*IQR)]
        else: # Z-Score
            z = np.abs((df[col] - df[col].mean())/df[col].std())
            df = df[z < threshold]
            
    return df.reset_index(drop=True)

def time_based_split(df, train_ratio=0.8):
    df = df.sort_index().reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

def create_sequence_data_from_arrays(X_scaled, y_scaled, seq_length=12):
    data = np.hstack((X_scaled, y_scaled.reshape(-1,1)))
    X_seq, y_seq = [], []
    
    for i in range(len(data)-seq_length):
        X_seq.append(data[i:i+seq_length, :-1])
        y_seq.append(data[i+seq_length, -1])
        
    return np.array(X_seq), np.array(y_seq)

##############################################################################
# 2) Gelişmiş Model Mimarisi (Attention Destekli)
##############################################################################

# Düzeltilmiş Attention Bloğu
def attention_block(inputs):
    time_steps = inputs.shape[1]  # Dinamik boyut hesabı
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

# Düzeltilmiş Model Yapısı
def build_model(hp, input_shape):
    model_type = hp.Choice('model_type', ['mlp', 'cnn', 'lstm', 'attention'])
    
    model = Sequential()
    
    if model_type == 'attention':
        # Conv1D with padding
        model.add(Conv1D(filters=hp.Int('filters', 32, 128, step=32),
                         kernel_size=hp.Choice('kernel_size', [3,5]),
                         activation='linear',
                         padding='same',  # <--- CRITICAL FIX
                         input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling1D(2))
        model.add(Dropout(hp.Float('dropout1', 0.2, 0.5)))
        
        # Modified attention block
        model.add(attention_block(model.layers[-1].output))  # <--- Dinamik boyut
        
        # LSTM with updated units
        model.add(LSTM(hp.Int('lstm_units', 64, 256, step=64), return_sequences=False))
        model.add(Dense(hp.Int('dense_units', 64, 256, step=64), activation='linear'))
        model.add(LeakyReLU())
        model.add(Dropout(hp.Float('dropout2', 0.2, 0.5)))
    
    # Diğer model tipleri aynı kalabilir...
##############################################################################
# 3) Gelişmiş Ana Fonksiyon
##############################################################################

def auto_run(df, target_col='close', sequence_length=24, max_trials=10, 
             save_model_name="best_model_seq.h5"):
    # Veri işleme
    df = add_technical_indicators(df)
    df = remove_outliers(df, method='iqr')
    
    train_df, test_df = time_based_split(df)
    
    # Feature engineering
    feature_cols = [col for col in df.columns if col != target_col]
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values.reshape(-1,1)
    
    # Scaling
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    
    X_train_s = scaler_X.transform(X_train)
    y_train_s = scaler_y.transform(y_train)
    
    # Sequence
    X_seq, y_seq = create_sequence_data_from_arrays(X_train_s, y_train_s, sequence_length)
    
    # Model tuning
    tuner = kt.RandomSearch(
        lambda hp: build_model(hp, input_shape=(sequence_length, len(feature_cols))),
        objective='val_loss',
        max_trials=max_trials,
        directory='tuning',
        project_name='advanced_ts'
    )
    
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True),
        ModelCheckpoint(save_model_name, save_best_only=True)
    ]
    
    tuner.search(X_seq, y_seq, epochs=100, validation_split=0.2,
                batch_size=64, callbacks=callbacks, verbose=1)
    
    # En iyi model
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Test
    X_test_s = scaler_X.transform(test_df[feature_cols].values)
    y_test_s = scaler_y.transform(test_df[target_col].values.reshape(-1,1))
    X_test_seq, y_test_seq = create_sequence_data_from_arrays(X_test_s, y_test_s, sequence_length)
    
    # Tahmin ve metrikler
    preds = scaler_y.inverse_transform(best_model.predict(X_test_seq))
    y_true = scaler_y.inverse_transform(y_test_seq.reshape(-1,1))
    
    # Görselleştirme
    plt.figure(figsize=(15,6))
    plt.plot(y_true[-200:], label='Gerçek Değerler')
    plt.plot(preds[-200:], label='Tahminler', alpha=0.7)
    plt.title('Model Performansı: Gerçek vs Tahmin')
    plt.legend()
    plt.savefig('performance_plot.png')
    
    # Sonraki Tahmin
    last_sequence = X_test_seq[-1:].reshape(1, sequence_length, len(feature_cols))
    next_pred = scaler_y.inverse_transform(best_model.predict(last_sequence))[0][0]
    
    results = {
        'model': best_model,
        'next_prediction': next_pred,
        'test_metrics': {
            'RMSE': np.sqrt(mean_squared_error(y_true, preds)),
            'MAE': mean_absolute_error(y_true, preds),
            'R2': r2_score(y_true, preds),
            'MAPE': mean_absolute_percentage_error(y_true, preds)
        },
        'plot': 'performance_plot.png'
    }
    
    return results

##############################################################################
# 4) Çalıştırma ve Test
##############################################################################

if __name__ == "__main__":
    # Veri yükleme
    df = pd.read_csv("BTCUSDT_data.csv", parse_dates=['close time'])
    
    # Modeli çalıştır
    results = auto_run(
        df,
        target_col='close',
        sequence_length=24,
        max_trials=15,
        save_model_name="advanced_model.h5"
    )
    
    print(f"\nBir sonraki tahmini close fiyatı: {results['next_prediction']:.2f}")
    print("Test Metrikleri:")
    for k,v in results['test_metrics'].items():
        print(f"{k}: {v:.4f}")
    
    print("\nPerformans grafiği 'performance_plot.png' olarak kaydedildi.")