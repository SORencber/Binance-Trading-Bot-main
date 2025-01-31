#models/train_xgboost.py
"""
Örnek XGBoost eğitim scripti:
  - Basit CSV veri
  - RSI/ADX feature
  - Label => sonraki bar up/down
  - Hyperparam rastgele
  - Kaydet => xgboost_model.pkl
"""
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split

def main():
    # 1) Veri
    df= pd.read_csv("data/price_data.csv")  # placeholder
    df["rsi"]= ta.rsi(df["close"], length=14)
    df["adx"]= ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    df["future"]= df["close"].shift(-1)
    df.dropna(inplace=True)

    df["label"]= (df["future"]> df["close"]).astype(int)
    features= df[["rsi","adx"]].values
    labels= df["label"].values

    X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.2, shuffle=False)

    # 2) XGBoost model
    model= xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # 3) Basit skor
    acc_train= model.score(X_train, y_train)
    acc_test= model.score(X_test, y_test)
    print(f"Train acc={acc_train:.2f}, Test acc={acc_test:.2f}")

    # 4) Kaydet
    joblib.dump(model,"xgboost_model.pkl")
    print("Saved => xgboost_model.pkl")

if __name__=="__main__":
    main()
