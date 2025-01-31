# backtests/trainer_behavior.py
"""
Offline => Davranışsal veriyi (sentiment+onchain+fiyat) toplayıp ML model train + save
"""

import asyncio
import pandas as pd
import pandas_ta as ta
from binance import AsyncClient
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier

from data.sentiment_data import combine_sentiment_scores
from data.onchain_data import fetch_fear_greed_index
# placeholder => gerçekte tam veri toplanır

async def train_behavior_model(symbol="BTCUSDT", days=30, out_path="model_data/rf_behavior.pkl"):
    from binance.enums import KLINE_INTERVAL_15MINUTE
    end= int(datetime.utcnow().timestamp()*1000)
    start= end- days*24*3600*1000
    client= await AsyncClient.create("YOUR_KEY","YOUR_SECRET")
    raw= await client.get_historical_klines(symbol, KLINE_INTERVAL_15MINUTE, str(start), str(end))
    await client.close_connection()

    df= pd.DataFrame(raw, dtype=float,
        columns=["open_time","open","high","low","close","vol","close_time","qav","n","tb","tq","ig"])
    df["open_time"]= pd.to_datetime(df["open_time"], unit="ms")
    df.sort_values("open_time", inplace=True)
    df["rsi"]= ta.rsi(df["close"], length=14)
    df["adx"]= ta.adx(df["high"], df["low"], df["close"], length=14)["ADX_14"]
    # Sentiment & Onchain => placeholder sabit vs.
    # Gerçekte news/twitter => df["sentiment"] = ...
    # fear&greed => ...
    # pseudo
    df["sentiment"]= 0.0
    df["onchain"]= 0.0

    df["future"]= df["close"].shift(-1)
    df.dropna(inplace=True)
    df["label"]= (df["future"]> df["close"]).astype(int)

    X= df[["rsi","adx","sentiment","onchain"]]
    y= df["label"]
    rf= RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X,y)
    joblib.dump(rf, out_path)
    print(f"[BehaviorTrainer] Model => {out_path}, score={rf.score(X,y):.2f}")

def main():
    asyncio.run(train_behavior_model())

if __name__=="__main__":
    main()
