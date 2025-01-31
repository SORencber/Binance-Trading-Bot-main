# models/train_lstm.py
"""
Basit Keras LSTM => 0=SELL,1=BUY sınıflandırması.
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import tensorflow as tf

def main():
    # 1) Veri
    df= pd.read_csv("../data/price_data.csv")
    #df["rsi"]= ta.rsi(df["close"], length=14)
    #df["adx"]= ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["future"]= df["Close"].shift(-1)
    df.dropna(inplace=True)
    df["label"]= (df["future"]> df["Close"]).astype(int)

    features= df.values
    labels= df["label"].values

    # 2) Basit LSTM => Aslında 2D veriyi 3D => (samples,timesteps,features)
    # placeholder => timesteps=1
    X= features.reshape((features.shape[0],1, features.shape[1]))
    y= labels

    split= int(len(X)*0.8)
    X_train, X_test= X[:split], X[split:]
    y_train, y_test= y[:split], y[split:]

    # 3) Model
    model= tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(16, input_shape=(1,X.shape[2])))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=32)
    loss, acc= model.evaluate(X_test, y_test)
    print(f"LSTM test acc={acc:.2f}")

    model.save("lstm_model.h5")
    print("Saved => lstm_model.h5")

if __name__=="__main__":
    main()
