"""
auto_seq_train.py

Düzeltilmiş versiyon:

1) LeakyReLU -> Ayrı layer yaklaşımı
2) tuner.build(...) -> tuner.hypermodel.build(...)
3) Bazı ufak iyileştirmeler
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
# LeakyReLU'yu "activation" yerine ayrı layer olarak kullanacağız
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.losses import Huber

from tensorflow.keras.utils import custom_object_scope,get_custom_objects

from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam,AdamW, Nadam,RMSprop
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                        ModelCheckpoint, TerminateOnNaN)
from tensorflow.keras.initializers import HeNormal
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Keras Tuner (kerastuner veya keras-tuner olarak kurulabilir)
try:
    from kerastuner.tuners import RandomSearch
except ImportError:
    import keras_tuner as kt
    RandomSearch = kt.RandomSearch

##############################################################################
# 1) Yardımcı Fonksiyonlar
##############################################################################

def log_return(df, col='close'):
    """Close fiyatı üzerinden log_return hesaplar."""
    df['log_return'] = np.log(df[col] / df[col].shift(1))
    return df

def add_technical_indicators(df):
    """
    RSI, MACD, Bollinger Bands gibi temel indikatörleri ekler.
    Eksik değerleri doldurur. 'Close' kolonu olmalı.
    """
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    if 'close' not in df.columns:
        raise ValueError("DataFrame'te 'Close' kolonu yok.")

    close_np = df['close'].values

    
    # add_technical_indicators() içine ekleyin:
           # Log Return
    df = log_return(df, 'close')

    # Eksik doldurma
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def remove_outliers(df, z_thresh=1.0):
    """
    Aykırı değerleri z-score'a göre filtreleme.
    Büyük z-score'lu satırlar atılır. 
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        mean_val = series.mean()
        std_val = series.std() + 1e-9
        z_score = (series - mean_val) / std_val
        df = df[np.abs(z_score) < z_thresh]
    df.reset_index(drop=True, inplace=True)
    return df

def time_based_split(df, train_ratio=0.9, time_col='close time'):
    """
    Zaman serisi verisini kronolojik şekilde ayırır: 
    ilk %80 train, son %20 test (varsayılan).
    """
    if time_col in df.columns:
        df.sort_values(by=time_col, inplace=True)
    else:
        df.sort_index(inplace=True)
    df.reset_index(drop=True, inplace=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df

def create_sequence_data_from_arrays(X_scaled, y_scaled, seq_length=12):
    """
    X_scaled shape (N, feature_count)
    y_scaled shape (N, 1)
    => birleştir => sequence
    => X_seq shape (M, seq_length, feature_count), y_seq shape(M,)
    """
    data = np.concatenate([X_scaled, y_scaled], axis=1)
    rows, cols = data.shape
    X_list, y_list = [], []

    for i in range(rows - seq_length):
        # features kısım: 0:cols-1
        X_seq = data[i : i+seq_length, :-1] 
        # target: son sütun
        y_val = data[i+seq_length, -1]
        X_list.append(X_seq)
        y_list.append(y_val)

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    return X_arr, y_arr

##############################################################################
# 2) Model Kurma (LeakyReLU -> Ayrı Katman)
##############################################################################
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):  # kwargs ekleyerek hata önlenir
        super(AttentionBlock, self).__init__(**kwargs)

    def build(self, input_shape):
        self.time_steps = input_shape[1]
        self.Permute1 = Permute((2, 1))
        self.Dense = Dense(self.time_steps, activation='softmax')
        self.Permute2 = Permute((2, 1))
        self.Multiply = Multiply()
        
    def call(self, inputs):
        a = self.Permute1(inputs)
        a = self.Dense(a)
        a_probs = self.Permute2(a)
        return self.Multiply([inputs, a_probs])

    def get_config(self):
        """Serileştirme için gerekli konfigürasyon metodu."""
        config = super(AttentionBlock, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        """Modeli yüklerken hata almamak için gerekli metod."""
        return cls(**config)


def build_model(hp, input_shape):
    """
    Keras Tuner icin 'build_model' fonksiyonu.
    Basit bir secim: ['mlp','cnn','lstm','cnn_lstm']
    Dikkat: LeakyReLU 'activation' parametresi yerine 
    => Dense(activation='linear') + model.add(LeakyReLU(alpha=...))
    """

    #model_type = hp.Choice('model_type', ['mlp', 'cnn', 'lstm', 'cnn_lstm'])
    model_type = hp.Choice('model_type', ['mlp', 'cnn', 'lstm', 'cnn_lstm','attention'])

    model = Sequential()
   
    if model_type == 'mlp':
        model.add(Flatten(input_shape=input_shape))
        mlp_layers = hp.Int('mlp_layers', 1, 3)
        for i in range(mlp_layers):
            units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
            l2_val = hp.Choice('l2_reg', [1e-4, 1e-5, 1e-6])
            dropout_val = hp.Float('dropout_mlp', 0.1, 0.5, step=0.1)

            # 1) Dense
            model.add(Dense(units, activation='linear',
                            kernel_regularizer=regularizers.l2(l2_val)))
            # 2) LeakyReLU layer
            model.add(LeakyReLU(alpha=0.01))
            # 3) BatchNorm
            model.add(BatchNormalization())
            # 4) Dropout
            model.add(Dropout(dropout_val))

    elif model_type == 'cnn':
        filters = hp.Int('filters_cnn', 64, 256, step=64)
        kernel_size = hp.Choice('kernel_size', [ 3, 5])
        l2_val = hp.Choice('l2_reg', [1e-4, 1e-5])
        dropout_val = hp.Float('dropout_cnn', 0.1, 0.5, step=0.1)
        dense_units = hp.Int('dense_units_cnn', min_value=128, max_value=256, step=64)
        dropout2_val = hp.Float('dropout2_cnn', 0.1, 0.5, step=0.1)

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                         activation='linear',
                         kernel_regularizer=regularizers.l2(l2_val),
                         input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_val))
        model.add(Flatten())

        model.add(Dense(dense_units, activation='linear'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout2_val))

    elif model_type == 'lstm':
        lstm_units = hp.Int('lstm_units', 64, 256, step=64)
        l2_val     = hp.Choice('l2_reg', [1e-4, 1e-5])
        dropout_val= hp.Float('dropout_lstm', 0.1, 0.5, step=0.1)
        dense_units= hp.Int('dense_units_lstm', min_value=64, max_value=256, step=64)
        dropout2_val= hp.Float('dropout2_lstm', 0.1, 0.5, step=0.1)

        model.add(LSTM(lstm_units,
                       return_sequences=False,
                       kernel_regularizer=regularizers.l2(l2_val),
                       activation='linear',
                       input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_val))

        model.add(Dense(dense_units, activation='linear'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout2_val))

    elif model_type == 'cnn_lstm':
        # CNN + LSTM
        filters = hp.Int('filters_cnnlstm', 64, 256, step=64)
        kernel_size= hp.Choice('kernel_size_cnnlstm', [2, 3, 5])
        l2_val     = hp.Choice('l2_reg', [1e-4, 1e-5])
        dropout_val= hp.Float('dropout_cnnlstm', 0.1, 0.5, step=0.1)

        lstm_units = hp.Int('lstm_units_cnnlstm', 64, 256, step=64)
        dropout2_val= hp.Float('dropout2_cnnlstm', 0.1, 0.5, step=0.1)

        dense_units= hp.Int('dense_units_cnnlstm', min_value=64, max_value=256, step=64)
        dropout3_val= hp.Float('dropout3_cnnlstm', 0.1, 0.5, step=0.1)

        # CNN
        model.add(Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         activation='linear',
                         kernel_regularizer=regularizers.l2(l2_val),
                         input_shape=input_shape))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(dropout_val))

        # LSTM
        model.add(LSTM(lstm_units, return_sequences=False, activation='linear'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(BatchNormalization())
        model.add(Dropout(dropout2_val))

        # Dense
        model.add(Dense(dense_units, activation='linear'))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(dropout3_val))
    else :
# Conv1D with padding
        model.add(Conv1D(
            filters=hp.Int('filters', 64, 256, step=64),
            kernel_size=hp.Choice('kernel_size', [3,5]),
            activation='linear',
            padding='same',
            input_shape=input_shape
        ))
        model.add(LeakyReLU(alpha=0.01))
        model.add(MaxPooling1D(2))
        model.add(Dropout(hp.Float('dropout1', 0.1, 0.5)))
        
        # Attention Block as Layer
        model.add(AttentionBlock())  # <--- Özel Katman
        
        # LSTM
        model.add(LSTM(hp.Int('lstm_units', 64, 256, step=64), return_sequences=False))
        model.add(Dense(hp.Int('dense_units', 64, 256, step=64), activation='linear'))
        model.add(LeakyReLU())
        model.add(Dropout(hp.Float('dropout2', 0.1, 0.3)))
    
    
    model.add(Dense(1, activation='linear'))
    
    #optimizer = hp.Choice('optimizer', ['adam', 'nadam'])
    optimizer = hp.Choice('optimizer', ['adam', 'nadam', 'adamw', 'rmsprop'])

    lr = hp.Choice('lr', [1e-3, 5e-4, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=lr) if optimizer == 'adam' else AdamW(learning_rate=lr),
        loss=Huber(),
        metrics=['mae']
    )
    
    
    return model

##############################################################################
# 3) Ana Fonksiyon: Otomatik Seq
##############################################################################
import os
def auto_run(df,
             target_col='close',
             time_col='close time',
             sequence_length=12,
             train_ratio=0.8,
             max_trials=5,
             executions_per_trial=1,
             epochs=50,
             batch_size=32,
             use_outlier_removal=True,
             save_model_name="best_autonomous_model_seq.h5",
             use_pretrained=True
             ):
    
    # Önceden eğitilmiş modeli yükleme denemesi
    if use_pretrained and os.path.exists(save_model_name):
        try:
            from tensorflow.keras.models import load_model
            with custom_object_scope({'AttentionBlock': AttentionBlock}):
                best_model = load_model(save_model_name)
            print(f"\nÖnceden eğitilmiş model başarıyla yüklendi: {save_model_name}")
            
            # Test verisiyle tahmin ve metrikleri hesapla
            X_test_seq, y_test_seq, scaler_y = prepare_test_data(df, target_col, time_col, sequence_length, train_ratio)
            preds_s = best_model.predict(X_test_seq)
            preds = scaler_y.inverse_transform(preds_s).flatten()
            y_test_act = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

            rmse = np.sqrt(mean_squared_error(y_test_act, preds))
            mae = np.mean(np.abs(y_test_act - preds))
            mape = np.mean(np.abs((y_test_act - preds) / (y_test_act + 1e-9))) * 100
            r2 = r2_score(y_test_act, preds)

            print(f"\n--- Test Sonuçları (Önceden Eğitilmiş Model) ---\nRMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")

            next_pred = predict_future(best_model, X_test_seq[-1:], scaler_y)
            
            return {
                "model": best_model,
                "next_prediction": next_pred,
                "test_metrics": {
                    "RMSE": rmse,
                    "MAE": mae,
                    "MAPE": mape,
                    "R2": r2
                }
            }
        except Exception as e:
            print(f"Model yüklenirken hata oluştu: {e}\nYeni model eğitilecek.")

    # Eğer önceden eğitilmiş model yoksa, yeni model eğit
    print("\nÖnceden eğitilmiş model bulunamadı veya yüklenemedi. Yeni model eğitilecek.")
    
    X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_y = prepare_data(df, target_col, time_col, sequence_length, train_ratio, use_outlier_removal)

    from kerastuner.tuners import RandomSearch

    def tuner_build_fn(hp):
        return build_model(hp, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

    tuner = RandomSearch(
        tuner_build_fn,
        objective='val_loss',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
                directory='financial_forecast_logs',

        project_name="financial_forecast",
        overwrite=True
    )

    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    nan_cb = TerminateOnNaN()
    mc = ModelCheckpoint('temp_best_model.h5', monitor='val_loss', save_best_only=True,
                         save_weights_only=False, verbose=1)

    tuner.search(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlp, nan_cb, mc],
        verbose=1
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n--- En iyi hiperparametreler ---")
    for param in best_hp.values.keys():
        print(f"{param} = {best_hp.values[param]}")

    # Modeli oluştur ve eğit
    best_model = tuner.hypermodel.build(best_hp)

    history = best_model.fit(
        X_train_seq, y_train_seq,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlp, nan_cb, mc],
        verbose=2
    )

    # En iyi modeli kaydet
    best_model.save(save_model_name)
    print(f"\nModel başarıyla kaydedildi: {save_model_name}")

    # Test sonuçlarını hesapla
    preds_s = best_model.predict(X_test_seq)
    preds = scaler_y.inverse_transform(preds_s).flatten()
    y_test_act = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    rmse = np.sqrt(mean_squared_error(y_test_act, preds))
    mae = np.mean(np.abs(y_test_act - preds))
    mape = np.mean(np.abs((y_test_act - preds) / (y_test_act + 1e-9))) * 100
    r2 = r2_score(y_test_act, preds)

    print(f"\n--- Test Sonuçları ---\nRMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.2f}%, R2={r2:.4f}")

    next_pred = predict_future(best_model, X_test_seq[-1:], scaler_y)

    return {
        "model": best_model,
        "next_prediction": next_pred,
        "test_metrics": {
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "R2": r2
        }
    }


def prepare_data(df, target_col, time_col, sequence_length, train_ratio, use_outlier_removal):
    """ Eğitim ve test verilerini hazırlar. """

    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

    df = add_technical_indicators(df)
    
    if use_outlier_removal:
        df = remove_outliers(df, z_thresh=1.5)

    train_df, test_df = time_based_split(df, train_ratio=train_ratio, time_col=time_col)
    
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != target_col]
    print(f"Train feature count: {len(feature_cols)} -> {feature_cols}")

    X_train_df = train_df[feature_cols].copy()
    y_train_df = train_df[target_col].copy()

    X_test_df = test_df[feature_cols].copy()
    y_test_df = test_df[target_col].copy()
    #print(f"X_train_seq shape: {X_train_seq.shape}")

    for dff in [X_train_df, X_test_df]:
        dff.replace([np.inf, -np.inf], np.nan, inplace=True)
        dff.ffill(inplace=True)
        dff.bfill(inplace=True)

    # Ölçekleme
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Hedef için MinMax

    X_train_np = X_train_df.values
    y_train_np = y_train_df.values.reshape(-1,1)

    X_test_np  = X_test_df.values
    y_test_np  = y_test_df.values.reshape(-1,1)

    scaler_X.fit(X_train_np)
    scaler_y.fit(y_train_np)

    X_train_s = scaler_X.transform(X_train_np)
    y_train_s = scaler_y.transform(y_train_np)

    X_test_s  = scaler_X.transform(X_test_np)
    y_test_s  = scaler_y.transform(y_test_np)

    # Sequence oluşturma
    X_train_seq, y_train_seq = create_sequence_data_from_arrays(X_train_s, y_train_s, seq_length=sequence_length)
    X_test_seq,  y_test_seq  = create_sequence_data_from_arrays(X_test_s,  y_test_s,  seq_length=sequence_length)
    print(f"X_train_seq shape: {X_train_seq.shape}")

    return X_train_seq, y_train_seq, X_test_seq, y_test_seq, scaler_y


def prepare_test_data(df, target_col, time_col, sequence_length, train_ratio):
    """ Sadece test verisini hazırlar. """
    
    train_df, test_df = time_based_split(df, train_ratio=train_ratio, time_col=time_col)

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != target_col]
    print(f"Test feature count: {len(feature_cols)} -> {feature_cols}")

    X_test_df = test_df[feature_cols].copy()
   # Eksik sütunları tamamla (train'de olup testte olmayan sütunlar)
    full_feature_cols = ['log_return']  # Train setindeki ek sütunlar
    for col in full_feature_cols:
        if col not in X_test_df.columns:
            X_test_df[col] = 0  # Veya df[col].mean() ile ortalama değer atanabilir

    print(f"Updated Test feature count: {len(X_test_df.columns)} -> {list(X_test_df.columns)}")

    y_test_df = test_df[target_col].copy()

    X_test_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test_df.ffill(inplace=True)
    X_test_df.bfill(inplace=True)

    scaler_X = StandardScaler()
    #scaler_y = StandardScaler()
    scaler_y = MinMaxScaler(feature_range=(-1, 1))  # Hedef için MinMax

    X_test_np  = X_test_df.values
    y_test_np  = y_test_df.values.reshape(-1,1)

    scaler_X.fit(X_test_np)
    scaler_y.fit(y_test_np)

    X_test_s  = scaler_X.transform(X_test_np)
    y_test_s  = scaler_y.transform(y_test_np)

    X_test_seq, y_test_seq = create_sequence_data_from_arrays(X_test_s, y_test_s, seq_length=sequence_length)
    print(f"X_train_seq shape: {X_test_seq.shape}")

    if X_test_seq.size == 0:
        raise ValueError("HATA: Test verisi boş! Model tahmin yapamaz.")
    #preds_s = best_model.predict(X_test_seq)
    return X_test_seq, y_test_seq, scaler_y


# Opsiyonel "tek adım ileri tahmin"
def predict_future(model, last_data, scaler_y):
    """
    last_data shape => (1, seq_len, feature_count)
    Tek adım ileri tahmin döndürür.
    """
    pred_s = model.predict(last_data)
    pred   = scaler_y.inverse_transform(pred_s).flatten()[0]
    return pred


if __name__ == "__main__":
    print("Bu kod, sekans tabanlı tam otonom training'i LeakyReLU fix ve tuner.hypermodel.build fix ile içerir.")

    symbol="BTCUSDT"
  
    csv_path = f"{symbol}_data.csv"
    #df_main = pd.read_csv(csv_path)
    if not os.path.exists(csv_path):
        print("veri.csv yok. Test edemiyorum.")
        exit()

    df = pd.read_csv(csv_path)
    # df['Open time'] = pd.to_datetime(df['Open time'])

    results = auto_run(
        df,
        target_col='close',
        time_col='close time',
        sequence_length=36,
        train_ratio=0.75,
        max_trials=100,
        executions_per_trial=5,
        epochs=200,
        batch_size=64,
        use_outlier_removal=True,
        save_model_name="best_autonomous_model_seq.h5",use_pretrained=True
    )
    print(f"\nBir sonraki tahmini close fiyatı: {results['next_prediction']:.2f}")
    print("Test Metrikleri:")
    for k,v in results['test_metrics'].items():
        print(f"{k}: {v:.4f}")
