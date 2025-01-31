# data/data_collector.py
import pandas as pd
import pandas_ta as ta
from binance import AsyncClient
from core.logging_setup import log
from core.context import SharedContext
import asyncio
import numpy as np
import traceback
from .sentiment_data  import( fetch_news_headlines)
from .onchain_data import (fetch_fear_greed_index)
from .sp import (sp500)
from .tf_indicators import (
    calculate_indicators_1m,
    calculate_indicators_5m,
    calculate_indicators_15m,
    calculate_indicators_30m,
    calculate_indicators_1h,
    calculate_indicators_4h,
    calculate_indicators_1d,
    add_oi_indicators,
    analyze_trends_and_signals_v6
)
from .data_fetching import(
    fetch_additional_data

)
# 1m verisi => 3000 bar ~ 2 gün
INTERVAL_1M = "1m"
LIMIT_1M    = 38100
# 5m verisi => 
INTERVAL_5M = "5m"
LIMIT_5M    = 7620
# 15m verisi => 
INTERVAL_15M = "15m"
LIMIT_15M    = 2560
# 30m verisi => 
INTERVAL_30M = "30m"
LIMIT_30M   = 1280

# 1h verisi => 
INTERVAL_1H= "1h"
LIMIT_1H    = 640

# 4h verisi => 200 bar ~ 33 gün
INTERVAL_4H = "4h"
LIMIT_4H    = 160

# 1d verisi => 200 bar ~ ~10 gün
INTERVAL_1D = "1d"
LIMIT_1D    = 30
###############################################################################
# 1) fetch_klines + load_and_clean_data
###############################################################################

async def fetch_klines(client_async: AsyncClient, symbol: str, timeframe, candle_count):

    print(f"[DEBUG] fetch_klines => symbol={symbol}, interval={timeframe}, limit={candle_count}")
   
            # Veri süresi hesaplama
    data_duration = (
                f"{candle_count * 1} minutes ago UTC" if timeframe == "1m" else
                f"{candle_count * 5} minutes ago UTC" if timeframe == "5m" else
                f"{candle_count * 10} minutes ago UTC" if timeframe == "10m" else
                f"{candle_count * 15} minutes ago UTC" if timeframe == "15m" else
                f"{candle_count * 30} minutes ago UTC" if timeframe == "30m" else             
                f"{candle_count} hours ago UTC" if timeframe == "1h" else
                f"{candle_count * 4} hours ago UTC" if timeframe == "4h" else
                f"{candle_count} days ago UTC"
            ) 
    klines =await  client_async.get_historical_klines(symbol, timeframe, data_duration)  
    print(f"[DEBUG] fetch_klines => raw klines len={len(klines)}")

    df = pd.DataFrame(klines, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ])

    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit='ms')

    numeric_cols = ["Open", "High", "Low", "Close", "Volume",
                    "Quote Asset Volume", "Taker Buy Base Volume", "Taker Buy Quote Volume"]
    for col in numeric_cols:
        df[col] = df[col].astype(float)

    print(f"[DEBUG] fetch_klines => df.shape={df.shape}")
    return df

def load_and_clean_data(df):
    print(f"[DEBUG] load_and_clean_data => input df.shape={df.shape}")
    df = df.copy()

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)

    if 'Open Time' in df.columns:
        df.rename(columns={'Open Time': 'timestamp'}, inplace=True)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    df.fillna(0, inplace=True)

    print(f"[DEBUG] load_and_clean_data => after cleaning, df.shape={df.shape}")
    print(f"[DEBUG] load_and_clean_data => NaN summary:\n{df.isna().sum()}")
    return df

# Daha önce tanımladığınızı varsayıyoruz:
# - fetch_klines(client, symbol, interval, limit) -> pd.DataFrame
# - calculate_indicators_1m, calculate_indicators_5m, ... calculate_indicators_1d
# - build_multi_timeframe_df -> asof merge
#   (veya "merge_asof" adımları)

# Burada, "calculate_indicators_*" fonksiyonları
# ve "merge_asof" mantığını bu fonksiyonun içinde göreceğiz.

async def fetch_and_merge_all_timeframes(
    client_async: AsyncClient,
    symbol: str,
    LIMIT_1M=LIMIT_1M,
    LIMIT_5M=LIMIT_5M,
    LIMIT_15M=LIMIT_15M,
    LIMIT_30M=LIMIT_30M,
    LIMIT_1H=LIMIT_1H,
    LIMIT_4H=LIMIT_4H,
    LIMIT_1D=LIMIT_1D
) -> pd.DataFrame:
    """
    1) Her TF için kline çek: 1m, 5m, 15m, 30m, 1h, 1d
    2) İndikatör hesaplama (1m/5m/15m/30m/1h/1d fonksiyonları)
    3) asof-merge => 1m "ana" tablo, 
       5m/15m/30m/1h/1d => suffix ile ekle
    4) Geriye pd.DataFrame döndür
    """
   
    # 1) Kline fetch
    df_1m  = await fetch_klines(client_async, symbol, INTERVAL_1M,  LIMIT_1M)
    df_5m  = await fetch_klines(client_async, symbol, INTERVAL_5M,  LIMIT_5M)
    df_15m = await fetch_klines(client_async, symbol, INTERVAL_15M, LIMIT_15M)
    df_30m = await fetch_klines(client_async, symbol, INTERVAL_30M, LIMIT_30M)
    df_1h  = await fetch_klines(client_async, symbol, INTERVAL_1H,  LIMIT_1H)
    df_4h  = await fetch_klines(client_async, symbol, INTERVAL_4H,  LIMIT_4H)

    df_1d  = await fetch_klines(client_async, symbol, INTERVAL_1D,  LIMIT_1D)

    # 2) Veri temizleme
    #    timestamp rename vs. 
    #    (Sizde fetch_klines belki 'timestamp' kolonu oluşturuyordur,
    #     yine de şöyle bir kontrol ekleyebilirsiniz.)
    for df_ in [df_1m, df_5m, df_15m, df_30m, df_1h,df_4h, df_1d]:
        if "Open Time" in df_.columns:
            df_.rename(columns={"Open Time": "timestamp"}, inplace=True)
        df_["timestamp"] = pd.to_datetime(df_["timestamp"], utc=True)
        df_.sort_values("timestamp", inplace=True)
        df_.reset_index(drop=True, inplace=True)

    # 3) İndikatör hesaplamaları
    df_1m_ind  = calculate_indicators_1m(df_1m)
    df_5m_ind  = calculate_indicators_5m(df_5m)
    df_15m_ind = calculate_indicators_15m(df_15m)
    df_30m_ind = calculate_indicators_30m(df_30m)
    df_1h_ind  = calculate_indicators_1h(df_1h)
    df_4h_ind  = calculate_indicators_4h(df_4h)

    df_1d_ind  = calculate_indicators_1d(df_1d)

    # 4) asof-merge => 1m ana tablo
    #    Aşağıdaki "merge_asof" adımlarını 
    #    "build_multi_timeframe_df" gibi bir fonksiyonla da yapabilirsiniz.
    df_1m_ind.sort_values("timestamp", inplace=True)
    df_5m_ind.sort_values("timestamp", inplace=True)
    df_15m_ind.sort_values("timestamp", inplace=True)
    df_30m_ind.sort_values("timestamp", inplace=True)
    df_1h_ind.sort_values("timestamp",  inplace=True)
    df_4h_ind.sort_values("timestamp",  inplace=True)
    df_1d_ind.sort_values("timestamp",  inplace=True)

    # 4.1 => 1m + 5m
    df_m5 = pd.merge_asof(
        df_1m_ind, 
        df_5m_ind, 
        on="timestamp",
        direction="backward",
        suffixes=("", "_5m")
    )
    # 4.2 => df_m5 + 15m
    df_m15 = pd.merge_asof(
        df_m5,
        df_15m_ind,
        on="timestamp",
        direction="backward",
        suffixes=("", "_15m")
    )
    # 4.3 => df_m15 + 30m
    df_m30 = pd.merge_asof(
        df_m15,
        df_30m_ind,
        on="timestamp",
        direction="backward",
        suffixes=("", "_30m")
    )
    # 4.4 => df_m30 + 1h
    df_m1h = pd.merge_asof(
        df_m30,
        df_1h_ind,
        on="timestamp",
        direction="backward",
        suffixes=("", "_4h")
    )
    df_m4h = pd.merge_asof(
        df_m1h,
        df_4h_ind,
        on="timestamp",
        direction="backward",
        suffixes=("", "_1h")
    )
    # 4.5 => df_m1h + 1d
    df_final = pd.merge_asof(
        df_m4h,
        df_1d_ind,
        on="timestamp",
        direction="backward",
        suffixes=("", "_1d")
    )

    # 5) NaN doldurma
    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    df_final.fillna(0, inplace=True)

    # 6) Tekrar sort + reset
    df_final.sort_values("timestamp", inplace=True)
    df_final.reset_index(drop=True, inplace=True)

    return df_final

async def  get_fetch_data(df_all,s,INTERVAL, LIMIT=1000):
                funding_rate, open_interest, order_book,open_interest_historical=await fetch_additional_data(s,INTERVAL,LIMIT,start_time=None, end_time=None)
                df_all['Funding_Rate'] = funding_rate
                df_all['Open_Interest'] = open_interest
                # df_oi = pd.DataFrame(open_interest_historical)

                # # 3) timestamp => datetime (milisaniye ise unit='ms', eğer saniye ise unit='s'!)
                # df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")

                # # 4) sumOpenInterest => float, sumOpenInterestValue => float
                # df_oi[f"{INTERVAL}_sumOpenInterest"] = df_oi["sumOpenInterest"].astype(float)
                # df_oi[f"{INTERVAL}_sumOpenInterestValue"] = df_oi["sumOpenInterestValue"].astype(float)
                # print("--- openintersts--",df_oi[f"{INTERVAL}_sumOpenInterestValue"])
                # # df_all: ana tablo (1m verisi)
                # df_all["timestamp"] = df_all["timestamp"].dt.tz_localize(None)

                # # df_oi: open interest DataFrame
                # df_oi["timestamp"] = df_oi["timestamp"].dt.tz_localize(None)
                # # 6) df_all da timestamp'e göre sıralanmış olmalı
                # df_all.sort_values("timestamp", inplace=True)
                # df_all.reset_index(drop=True, inplace=True)

                # # 7) asof merge => direction="backward"
                # df_all = pd.merge_asof(
                #     df_all,           # ana tablo
                #     df_oi,            # eklemek istediğimiz tablo
                #     on="timestamp",   # ortak kolon
                #     direction="backward",
                #     suffixes=(INTERVAL, "_oi")  # çakışan kolon isimlerini ayırmak isterseniz
                # )
                # df_all = add_oi_indicators(df_all, suffix=f"{INTERVAL}_")
                #print(df_all.columns)
                #print(df_all.tail())
              
                # Order Book analizi
                if "bids" in order_book and "asks" in order_book:
                    bids = order_book["bids"]
                    asks = order_book["asks"]
                    total_bids = sum(float(bid[1]) for bid in bids if bid and len(bid) > 1)
                    total_asks = sum(float(ask[1]) for ask in asks if ask and len(ask) > 1)
                    df_all['OrderBook_BidVol'] = total_bids
                    df_all['OrderBook_AskVol'] = total_asks

                    if total_bids > total_asks * 1.2:
                        order_book_num = 1
                    elif total_asks > total_bids * 1.2:
                        order_book_num = -1
                    else:
                        order_book_num = 0
                else:
                    order_book_num = 0
                
                df_all['Order_Book_Num'] = order_book_num
                
                return df_all

async def loop_data_collector(ctx: SharedContext, strategy):
    while True:
        try:
            for s in ctx.config["symbols"]:
                print(f"[DEBUG] loop_data_collector => symbol={s}")

                # 1) 1m veriyi çek
                #df = await fetch_klines(ctx.client_async, s, "1m", 1000)
                #print(f"[DEBUG] loop_data_collector => fetched df.shape={df.shape}")

                # 2) Data temizleme
                #df = load_and_clean_data(df)
                #print(f"[DEBUG] loop_data_collector => cleaned df.shape={df.shape}")

                # 3) MTF veriyi oluştur
                print("LIMITT.................",LIMIT_1M)
                df_all= await fetch_and_merge_all_timeframes(
                    ctx.client_async,
                    s,
                    LIMIT_1M=LIMIT_1M,
                    LIMIT_5M=LIMIT_5M,
                    LIMIT_15M=LIMIT_15M,
                    LIMIT_30M=LIMIT_30M,
                    LIMIT_1H=LIMIT_1H,
                    LIMIT_4H=LIMIT_4H,
                    LIMIT_1D=LIMIT_1D)  
                #print(f"[DEBUG] loop_data_collector => df_all.shape={df_all.shape}")

                # 4) Kaydet
                if s not in ctx.df_map:
                    ctx.df_map[s] = {}
                ctx.df_map[s]["1m"] = df_all

                # NaN kolonları
                #cols_with_nan = df_all.columns[df_all.isna().any()]
                #print(f"[DEBUG] loop_data_collector => NaN columns: {cols_with_nan.tolist()}")

                # CSV
                #df_all.to_csv("data/price_data.csv", index=False)

                #await get_fetch_data(df_all,s,LIMIT_15M,LIMIT_15M)

                #await get_fetch_data(df_all,s,INTERVAL_30M,LIMIT_30M)

                await get_fetch_data(df_all,s,INTERVAL_1H,LIMIT_1H)
                await get_fetch_data(df_all,s,INTERVAL_4H,LIMIT_4H)
                await get_fetch_data(df_all,s,INTERVAL_1D,LIMIT_1D)

                 # >70 iyi
                df_all['Fear_Greed_Index'] =fetch_fear_greed_index()
                
                # News Headline  -1 => çok negatif, +1 => çok pozitif.

                df_all['News_Headlines'] = fetch_news_headlines(s,"")


                df_all['SP500'], df_all['DXY'], df_all['VIX']=await sp500(INTERVAL_1D)
                # Son satır debug
                last_row = df_all.iloc[-1]
                
                #last_oi_val = last_row["sumOpenInterestValue_oi"]
                #print("------------------------<<<<>>>>>",last_oi_val)
                #rsi_1h = last_row.get("RSI_", None)
                #print(f"[DEBUG] => Last row:  close={df_all.columns}")
                #exit(1)
                result = analyze_trends_and_signals_v6(df_all)
                print("RESULT => ", result)

                print("\nDetail Scores => ", result["detail_scores"])
                print("\nDelayed Signals => ", result["delayed_signals"])

                df_all.to_csv("data/price_data.csv", index=False)

            await strategy.analyze_data()
            await asyncio.sleep(60)

        except Exception as e:
            log(f"[loop_data_collector] => {e}{e}\n{traceback.format_exc()}", "error")
            print(f"[DEBUG] Exception in loop_data_collector => {e}{e}\n{traceback.format_exc()}")
            await asyncio.sleep(30)
