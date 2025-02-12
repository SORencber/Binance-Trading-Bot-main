# data/data_collector.py
import pandas as pd
from binance import AsyncClient
from core.logging_setup import log
from core.context import SharedContext
import asyncio,os,aiohttp
from dotenv import load_dotenv
import traceback
import datetime
from datetime import timedelta 
from .sentiment_data  import( fetch_news_headlines_cached)
from .onchain_data import (fetch_fgi_and_onchain_15min)
from .sp import (fetch_sp500_dxy_vix_15min)
from .tf_indicators import (
    calculate_indicators_1m,
    calculate_indicators_5m,
    calculate_indicators_15m,
    calculate_indicators_30m,
    calculate_indicators_1h,
    calculate_indicators_4h,
    calculate_indicators_1d,calculate_indicators_1w,
    add_oi_indicators,holy_grail_all_timeframes,
    analyze_trends_and_signals_v6
)
from .data_fetching import(
    fetch_additional_data,fetch_oi_in_chunks

)
# .env dosyasını yükle
load_dotenv()
NEW_API_KEY    = os.getenv("NEW_API_KEY","")

async def fetch_klines(client_async: AsyncClient, symbol: str, timeframe, candle_count,    csv_path: str):

    print(f"[DEBUG] fetch_klines => symbol={symbol}, interval={timeframe}, limit={candle_count}")
   
            # Veri süresi hesaplama
    data_duration = (
        f"{candle_count} minutes ago UTC" if timeframe == "1m" else
        f"{candle_count * 5} minutes ago UTC" if timeframe == "5m" else
        f"{candle_count * 15} minutes ago UTC" if timeframe == "15m" else
        f"{candle_count * 30} minutes ago UTC" if timeframe == "30m" else
        f"{candle_count} hours ago UTC"        if timeframe == "1h" else
        f"{candle_count * 4} hours ago UTC"    if timeframe == "4h" else
        f"{candle_count} days ago UTC"         if timeframe == "1d" else
        f"{candle_count * 7} days ago UTC"     if timeframe == "1w" else
        ...
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
    numeric_cols = ["Open","High","Low","Close","Volume",
                    "Quote Asset Volume","Taker Buy Base Volume","Taker Buy Quote Volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)
   
        df.rename(columns={"Open Time": "timestamp"}, inplace=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        start_ts = df["timestamp"].iloc[0]
        end_ts   = df["timestamp"].iloc[-1]
        print(f"[DEBUG] Klines => start={start_ts}, end={end_ts}")

        # 2) OI => parça parça
        async with aiohttp.ClientSession() as session:
            df_oi = await fetch_oi_in_chunks(
                session=session,
                symbol=symbol,
                period=timeframe,
                start_ts=start_ts,
                end_ts=end_ts,
                max_bars=50000  # 50k bar
            )
        print(f"[DEBUG] OI => shape={df_oi.shape}")
        
        # 3) asof merge (direction="backward"), 
        #    kline timestamp'ına en yakın OI timestamp satırını eşleştir
        if not df_oi.empty:
           
            # milis => datetime
            df_oi["timestamp"] = pd.to_datetime(df_oi["timestamp"], unit="ms")  
            df_oi.sort_values("timestamp", inplace=True)
            df_oi.reset_index(drop=True, inplace=True)

            df_final = pd.merge_asof(
                df,
                df_oi[["timestamp","sumOpenInterest","sumOpenInterestValue"]],
                on="timestamp",
                direction="backward"
            )
        else:
            df_final = df.copy()
            df_final["sumOpenInterest"] = None
            df_final["sumOpenInterestValue"] = None

        # 4) CSV'yi güncelle (tek dosya)
        df_final.to_csv(csv_path, index=False)
        print(f"[DONE] => Klines+OI => shape={df_final.shape}, saved to {csv_path}")
        return df_final

async def fetch_initial_klines_to_csv(
    client: AsyncClient,
    symbol: str,
    timeframe: str,
    limit: int,
    csv_path: str
):
    """
    Tek seferde 'limit' kadar bar çekip csv_path'e kaydeder.
    """
    print(f"[INIT] => Fetching {limit} bars for {symbol} ({timeframe})...")
    klines = await client.get_historical_klines(
        symbol, 
        timeframe, 
        f"{limit} {timeframe} ago UTC"
    )
    df = pd.DataFrame(klines, columns=[
        "Open Time", "Open", "High", "Low", "Close", "Volume",
        "Close Time", "Quote Asset Volume", "Number of Trades",
        "Taker Buy Base Volume", "Taker Buy Quote Volume", "Ignore"
    ])
    df["Open Time"] = pd.to_datetime(df["Open Time"], unit='ms')
    df["Close Time"] = pd.to_datetime(df["Close Time"], unit='ms')

    numeric_cols = ["Open","High","Low","Close","Volume",
                    "Quote Asset Volume","Taker Buy Base Volume","Taker Buy Quote Volume"]
    for c in numeric_cols:
        df[c] = df[c].astype(float)
    
    df.rename(columns={"Open Time": "timestamp"}, inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[INIT] => {symbol}({timeframe}) initial fetch done, shape={df.shape}, saved to {csv_path}")



async def update_klines_csv(
    client: AsyncClient,
    symbol: str,
    timeframe: str,
    csv_path: str,
    max_rows=50000
) -> pd.DataFrame:
    """
    1) CSV'yi okuyup en son timestamp'ten sonraki KLINE verilerini ekler.
    2) Ardından en son OI timestamp'inden (last_oi_ts) sonrasını partial fetch ile alıp,
       var olan 'sumOpenInterest' kolonunu rename+fill yöntemiyle günceller.
    """
    # 1) Eğer CSV yoksa önce kline + OI initial fetch (tamamen size kalmış)
    if not os.path.exists(csv_path):
        await fetch_klines(client, symbol, timeframe, max_rows, csv_path)

    # CSV oku
    df_local = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df_local.sort_values("timestamp", inplace=True)
    df_local.reset_index(drop=True, inplace=True)

    if df_local.empty:
        # Boşsa bir kere daha fetch_klines
        await fetch_klines(client, symbol, timeframe, max_rows, csv_path)
        df_local = pd.read_csv(csv_path, parse_dates=["timestamp"])
        df_local.sort_values("timestamp", inplace=True)
        df_local.reset_index(drop=True, inplace=True)

    # 2) En son Kline timestamp => yeni klines'i çek
    last_ts = df_local["timestamp"].iloc[-1]
    fetch_start_time = last_ts + pd.Timedelta(seconds=1)
    fetch_start_str = fetch_start_time.strftime("%Y-%m-%d %H:%M:%S UTC")

    print(f"[UPDATE] => {symbol}({timeframe}), start={fetch_start_str}")
    new_klines = await client.get_historical_klines(symbol, timeframe, fetch_start_str)

    if not new_klines:
        print("[UPDATE] => No new klines.")
        df_merged = df_local
    else:
        df_new = pd.DataFrame(new_klines, columns=[
            "Open Time","Open","High","Low","Close","Volume",
            "Close Time","Quote Asset Volume","Number of Trades",
            "Taker Buy Base Volume","Taker Buy Quote Volume","Ignore"
        ])
        df_new["Open Time"] = pd.to_datetime(df_new["Open Time"], unit='ms')
        df_new["Close Time"] = pd.to_datetime(df_new["Close Time"], unit='ms')

        numeric_cols = ["Open","High","Low","Close","Volume",
                        "Quote Asset Volume","Taker Buy Base Volume","Taker Buy Quote Volume"]
        for c in numeric_cols:
            df_new[c] = df_new[c].astype(float)
    
        df_new.rename(columns={"Open Time": "timestamp"}, inplace=True)
        df_new.sort_values("timestamp", inplace=True)
        df_new.reset_index(drop=True, inplace=True)

        df_merged = pd.concat([df_local, df_new], ignore_index=True)
        df_merged.drop_duplicates(subset=["timestamp"], inplace=True)
        df_merged.sort_values("timestamp", inplace=True)
        df_merged.reset_index(drop=True, inplace=True)

    # max_rows sınırı
    if len(df_merged) > max_rows:
        df_merged = df_merged.iloc[-max_rows:]

    # Kline aralığı
    start_ts = df_merged["timestamp"].iloc[0]
    end_ts   = df_merged["timestamp"].iloc[-1]
    print(f"[DEBUG] Klines => start={start_ts}, end={end_ts}")

    # ==============================================================
    #  A) CSV'de OI kolonları yoksa ekle.
    # ==============================================================
    if "sumOpenInterest" not in df_merged.columns:
        df_merged["sumOpenInterest"] = None
    if "sumOpenInterestValue" not in df_merged.columns:
        df_merged["sumOpenInterestValue"] = None

    # ==============================================================
    #  B) En son NaN olmayan OI satırının timestamp'ini bul
    # ==============================================================
    df_oi_exist = df_merged.dropna(subset=["sumOpenInterest"])
    if not df_oi_exist.empty:
        last_oi_ts = df_oi_exist["timestamp"].iloc[-1]
    else:
        last_oi_ts = df_merged["timestamp"].iloc[0]

    print(f"[OI UPDATE] => last OI timestamp={last_oi_ts}")
    
    # ==============================================================
    #  C) partial fetch => [last_oi_ts, end_ts]
    # ==============================================================
    async with aiohttp.ClientSession() as session:
        df_oi_new = await fetch_oi_in_chunks(
            session=session,
            symbol=symbol,
            period=timeframe,  # örn '5m'
            start_ts=last_oi_ts,  # datetime
            end_ts=end_ts,        # datetime
            max_bars=50000
        )
    print(f"[DEBUG] OI => new partial shape={df_oi_new.shape}")

    if not df_oi_new.empty:
        # Dönüş => "timestamp"(int?), "sumOpenInterest", "sumOpenInterestValue"
        # 1) milis => datetime
        df_oi_new["timestamp"] = pd.to_datetime(df_oi_new["timestamp"], unit="ms")
        df_oi_new.sort_values("timestamp", inplace=True)
        df_oi_new.reset_index(drop=True, inplace=True)

        # 2) Kolonları rename => conflict olmadan asof merge
        df_oi_temp = df_oi_new.rename(columns={
            "sumOpenInterest": "sumOpenInterest_new",
            "sumOpenInterestValue": "sumOpenInterestValue_new"
        })

        # 3) asof merge
        df_temp = pd.merge_asof(
            df_merged,
            df_oi_temp[["timestamp","sumOpenInterest_new","sumOpenInterestValue_new"]],
            on="timestamp",
            direction="backward"
        )
        # 1) sumOpenInterest_new -> float
        df_temp["sumOpenInterest_new"] = df_temp["sumOpenInterest_new"].astype(float)
        df_temp["sumOpenInterest"] = df_temp["sumOpenInterest"].astype(float)
        df_temp["sumOpenInterestValue_new"] = df_temp["sumOpenInterestValue_new"].astype(float)
        df_temp["sumOpenInterestValue"] = df_temp["sumOpenInterestValue"].astype(float)

        df_temp["sumOpenInterest"] = (
        df_temp["sumOpenInterest"]
        .fillna(df_temp["sumOpenInterest_new"])
        .astype(float)
     )

        df_temp["sumOpenInterestValue"] = (
        df_temp["sumOpenInterestValue"]
        .fillna(df_temp["sumOpenInterestValue_new"])
        .astype(float)
        )

        # 5) geçici kolonları drop
        df_temp.drop(columns=["sumOpenInterest_new","sumOpenInterestValue_new"],
                     inplace=True, errors="ignore")

        df_final = df_temp
    else:
        df_final = df_merged.copy()

    df_final.to_csv(csv_path, index=False)
    print(f"[UPDATE] => Klines+OI => shape={df_final.shape}, new klines and partial OI update => {csv_path}")

    return df_final

def load_and_calc_1m(csv_path_1m: str) -> pd.DataFrame:
    df_1m = pd.read_csv(csv_path_1m, parse_dates=["timestamp"])
    df_1m.sort_values("timestamp", inplace=True)
    df_1m.reset_index(drop=True, inplace=True)
    
    df_1m = calculate_indicators_1m(df_1m)  # proje içindeki fonk

    # suffix => "_1m"
    old_cols = df_1m.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        elif c.endswith("_1m"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
        else:
            new_cols.append(c + "_1m")
    df_1m.columns = new_cols
    
    return df_1m

def load_and_calc_5m(csv_path_5m: str) -> pd.DataFrame:
    df_5m = pd.read_csv(csv_path_5m, parse_dates=["timestamp"])
    df_5m.sort_values("timestamp", inplace=True)
    df_5m.reset_index(drop=True, inplace=True)
    #print(df_5m.columns)
    
    df_5m = calculate_indicators_5m(df_5m)
    # suffix => "_5m"
    old_cols = df_5m.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        elif c.endswith("_5m"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
        else:
            new_cols.append(c + "_5m")
    df_5m.columns = new_cols
    
    return df_5m

def load_and_calc_15m(csv_path_15m: str) -> pd.DataFrame:
    df_15m = pd.read_csv(csv_path_15m, parse_dates=["timestamp"])
    df_15m.sort_values("timestamp", inplace=True)
    df_15m.reset_index(drop=True, inplace=True)
    
    df_15m = calculate_indicators_15m(df_15m)

    # suffix => "_15m"
    old_cols = df_15m.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        elif c.endswith("_15m"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
        else:
            new_cols.append(c + "_15m")
    df_15m.columns = new_cols
    
    return df_15m

def load_and_calc_30m(csv_path_30m: str) -> pd.DataFrame:
    df_30m = pd.read_csv(csv_path_30m, parse_dates=["timestamp"])
    df_30m.sort_values("timestamp", inplace=True)
    df_30m.reset_index(drop=True, inplace=True)
    
    df_30m = calculate_indicators_30m(df_30m)

    # suffix => "_30m"
    old_cols = df_30m.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        elif c.endswith("_30m"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
        else:
            new_cols.append(c + "_30m")
    df_30m.columns = new_cols
    
    return df_30m

def load_and_calc_1h(csv_path_1h: str) -> pd.DataFrame:
    df_1h = pd.read_csv(csv_path_1h, parse_dates=["timestamp"])
    df_1h.sort_values("timestamp", inplace=True)
    df_1h.reset_index(drop=True, inplace=True)
    
    df_1h = calculate_indicators_1h(df_1h)

    # suffix => "_1h"
    old_cols = df_1h.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        elif c.endswith("_1h"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
        else:
            new_cols.append(c + "_1h")
    df_1h.columns = new_cols
    
    return df_1h

def load_and_calc_4h(csv_path_4h: str) -> pd.DataFrame:
    df_4h = pd.read_csv(csv_path_4h, parse_dates=["timestamp"])
    df_4h.sort_values("timestamp", inplace=True)
    df_4h.reset_index(drop=True, inplace=True)
    
    df_4h = calculate_indicators_4h(df_4h)

    # suffix => "_4h"
    old_cols = df_4h.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        elif c.endswith("_4h"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
        else:
            new_cols.append(c + "_4h")
    df_4h.columns = new_cols
    
    return df_4h

def load_and_calc_1d(csv_path_1d: str) -> pd.DataFrame:
    df_1d = pd.read_csv(csv_path_1d, parse_dates=["timestamp"])
    df_1d.sort_values("timestamp", inplace=True)
    df_1d.reset_index(drop=True, inplace=True)
    
    df_1d = calculate_indicators_1d(df_1d)

    # suffix => "_1d"
    old_cols = df_1d.columns
    new_cols = []
    for c in old_cols:
      if c == "timestamp":
            new_cols.append(c)
      elif c.endswith("_1d"):
            # zaten _1d ile bitiyor => bir daha ekleme!
            new_cols.append(c)
      else:
            new_cols.append(c + "_1d")
    df_1d.columns = new_cols
    
    return df_1d

def load_and_calc_1w(csv_path_1w: str) -> pd.DataFrame:
    df_1w = pd.read_csv(csv_path_1w, parse_dates=["timestamp"])
    df_1w.sort_values("timestamp", inplace=True)
    df_1w.reset_index(drop=True, inplace=True)

    # Haftalık indikatör hesaplamalarınız (ör: RSI, EMA, vb.)
    df_1w = calculate_indicators_1w(df_1w)  # Kendi oluşturduğunuz fonksiyon

    # Sütun isimlerine "_1w" eki eklemek isterseniz:
    old_cols = df_1w.columns
    new_cols = []
    for c in old_cols:
        if c == "timestamp":
            new_cols.append(c)
        else:
            new_cols.append(c + "_1w")
    df_1w.columns = new_cols

    return df_1w

def merge_all_tfs(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,    df_1w: pd.DataFrame

) -> pd.DataFrame:
    # Sort by timestamp
    df_1m.sort_values("timestamp", inplace=True)
    df_5m.sort_values("timestamp", inplace=True)
    df_15m.sort_values("timestamp", inplace=True)
    df_30m.sort_values("timestamp", inplace=True)
    df_1h.sort_values("timestamp", inplace=True)
    df_4h.sort_values("timestamp", inplace=True)
    df_1d.sort_values("timestamp", inplace=True)
    df_1w.sort_values("timestamp", inplace=True)

    
    # 1m + 5m
    df_m5 = pd.merge_asof(
        df_1m,
        df_5m,
        on="timestamp",
        direction="backward",
        suffixes=("_1m", "_5m")   
          )
    # (df_m5) + 15m
    df_m15 = pd.merge_asof(
        df_m5,
        df_15m,
        on="timestamp",
        direction="backward",
        suffixes=("", "_15m")
    )
    # (df_m15) + 30m
    df_m30 = pd.merge_asof(
        df_m15,
        df_30m,
        on="timestamp",
        direction="backward",
        suffixes=("", "_30m")
    )
    # (df_m30) + 1h
    df_m1h = pd.merge_asof(
        df_m30,
        df_1h,
        on="timestamp",
        direction="backward",
        suffixes=("", "_1h")
    )
    # (df_m1h) + 4h
    df_m4h = pd.merge_asof(
        df_m1h,
        df_4h,
        on="timestamp",
        direction="backward",
        suffixes=("", "_4h")
    )
    # (df_m4h) + 1d
    df_m1d = pd.merge_asof(
        df_m4h,
        df_1d,
        on="timestamp",
        direction="backward",
        suffixes=("", "_1d")
    )
    df_final = pd.merge_asof(
        df_m1d,
        df_1w,
        on="timestamp",
        direction="backward",
        suffixes=("", "_1w")
    )

    df_final.reset_index(drop=True, inplace=True)
    return df_final

async def  get_fetch_data(s):
                funding_rate, open_interest, order_book=await fetch_additional_data(s)
             
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
                 # Ön tanımları yapıyoruz
             # Ön tanımları yapıyoruz
                order_book_num = 0
                total_asks = 0
                total_bids = 0

                if order_book:
                    if "bids" in order_book and "asks" in order_book:
                        bids = order_book["bids"]
                        asks = order_book["asks"]
                        total_bids = sum(float(bid[1]) for bid in bids if bid and len(bid) > 1)
                        total_asks = sum(float(ask[1]) for ask in asks if ask and len(ask) > 1)
                        
                        if total_bids > total_asks * 1.2:
                            order_book_num = 1
                        elif total_asks > total_bids * 1.2:
                            order_book_num = -1
                        else:
                            order_book_num = 0
                    else:
                        # Bids veya Asks yoksa
                        order_book_num = 0
                        # total_asks, total_bids burada 0 olarak kalır
                else:
                    # order_book False ya da None gelirse
                    order_book_num = 0
                    # total_asks, total_bids 0 olarak kalır

                return funding_rate, open_interest, order_book_num, total_asks, total_bids
             

async def loop_data_collector(ctx: SharedContext, strategy):
    while True:
      
        try:
            # now_utc = datetime.utcnow()
            # # Bir sonraki tam dakika (1m bar) kapanışının utc zamanı
            # next_minute = now_utc.replace(second=0, microsecond=0) + timedelta(minutes=1)
            
            # # Kaç saniye kaldı?
            # wait_sec = (next_minute - datetime.utcnow()).total_seconds()
            
            # # Geri sayım: her 1 saniyede bir ekrana yazalım
            # # int() alarak virgülleri yok ediyoruz.
            # for sec in range(int(wait_sec), 0, -1):
            #     print(f"Bar'ın kapanmasına {sec} saniye kaldı...")
            #     await asyncio.sleep(1)

            # Burada tam bar kapanış anına geldik
            print("Bar kapandı! (veya yeni bar açıldı). Şimdi veri toplanabilir/işlenebilir...\n")

            for s in ctx.config["symbols"]:
                print(f"[DEBUG] loop_data_collector => symbol={s}")

                # 1) CSV update => 1m
                csv_1m_path = f"data_storage/{s}_1m.csv"
                df_1m = await update_klines_csv(ctx.client_async, s, "1m", csv_1m_path, max_rows=50000)

                # 2) CSV update => 5m
                csv_5m_path = f"data_storage/{s}_5m.csv"
                df_5m = await update_klines_csv(ctx.client_async, s, "5m", csv_5m_path, max_rows=10000)

                # 3) CSV update => 15m
                csv_15m_path = f"data_storage/{s}_15m.csv"
                df_15m = await update_klines_csv(ctx.client_async, s, "15m", csv_15m_path, max_rows=6000)

                # 4) CSV update => 30m
                csv_30m_path = f"data_storage/{s}_30m.csv"
                df_30m = await update_klines_csv(ctx.client_async, s, "30m", csv_30m_path, max_rows=6000)

                # 5) CSV update => 1h
                csv_1h_path = f"data_storage/{s}_1h.csv"
                df_1h = await update_klines_csv(ctx.client_async, s, "1h", csv_1h_path, max_rows=6000)

                # 6) CSV update => 4h
                csv_4h_path = f"data_storage/{s}_4h.csv"
                df_4h = await update_klines_csv(ctx.client_async, s, "4h", csv_4h_path, max_rows=6000)

                # 7) CSV update => 1d
                csv_1d_path = f"data_storage/{s}_1d.csv"
                df_1d = await update_klines_csv(ctx.client_async, s, "1d", csv_1d_path, max_rows=1000)
               
                csv_1w_path = f"data_storage/{s}_1w.csv"
                await update_klines_csv(ctx.client_async, s, "1w", csv_1w_path, max_rows=1000)
               
                # 8) Şimdi CSV'lerden oku + indikatör hesapla + rename
                df_1m_ind  = load_and_calc_1m(csv_1m_path)
                df_5m_ind  = load_and_calc_5m(csv_5m_path)
                df_15m_ind = load_and_calc_15m(csv_15m_path)
                df_30m_ind = load_and_calc_30m(csv_30m_path)
                df_1h_ind  = load_and_calc_1h(csv_1h_path)
                df_4h_ind  = load_and_calc_4h(csv_4h_path)
                df_1d_ind  = load_and_calc_1d(csv_1d_path)
                df_1w_ind  = load_and_calc_1w(csv_1w_path)

                # 9) MTF merge => 1m bazlı final
                df_final = merge_all_tfs(
                    df_1m_ind,
                    df_5m_ind,
                    df_15m_ind,
                    df_30m_ind,
                    df_1h_ind,
                    df_4h_ind,
                    df_1d_ind,df_1w_ind
                )
                #print("df_final---->",df_final.shape)
                #print(df_final[['timestamp', 'Close_1m','Close_5m','Close_15m','Close_30m','Close_1h', 'Close_4h','Close_1d']].tail(240))
                #print(df_1h_ind[['timestamp', 'Close_1h']].tail(6))
                #print(df_1h[['timestamp', 'Close']].tail(10))

                # 10) Onchain, macro, sentiment
                fgi_val, onchain_val = await fetch_fgi_and_onchain_15min(s)
                sp500_val, sp500_chg, dxy_val, dxy_chg, vix_val, vix_chg = await fetch_sp500_dxy_vix_15min()
       
                last_idx = df_final.index[-1]
                
                df_final.loc[last_idx, "SP500"] = sp500_val
                df_final.loc[last_idx, "DXY"]   = dxy_val
                df_final.loc[last_idx, "VIX"]   = vix_val
                df_final.loc[last_idx, "SPX_Change"] = sp500_chg
                df_final.loc[last_idx, "DXY_Change"]   = dxy_chg
                df_final.loc[last_idx, "VIX_Change"]   = vix_chg

                df_final.loc[last_idx, "Fear_Greed_Index"] = fgi_val
                df_final.loc[last_idx, "Onchain_Score"]    = onchain_val

                df_final.loc[last_idx, "News_Headlines"]   = fetch_news_headlines_cached("BTCUSDT", NEW_API_KEY, interval_minutes=30)

                
                funding_rate,open_interest,order_book_num,total_asks,total_bids= await get_fetch_data(s)
                #print(funding_rate,open_interest,order_book_num,total_asks,total_bids)
                df_final.loc[last_idx,'Order_Book_Num'] = order_book_num
                df_final.loc[last_idx,'OrderBook_BidVol'] = total_bids
                df_final.loc[last_idx,'OrderBook_AskVol'] = total_asks
                df_final.loc[last_idx,'Funding_Rate'] = funding_rate
                df_final.loc[last_idx,'Open_Interest'] = open_interest
                # 11) synergy / analyze
                print("analiz basladi")
                #holy_grail_all_timeframes(df_final)
                #result = analyze_trends_and_signals_v6(df_final)
                #df_final.to_csv("data/price_data.csv", index=False)
                
                # print("RESULT =>", result)
                # print("Detail Scores =>", result["detail_scores"])
                # print("Delayed Signals =>", result["delayed_signals"])
                # if s not in ctx.df_map:
                ctx.df_map[s] = {}
                ctx.df_map[s]["merged"] = df_final
                ctx.df_map[s]["1m"] = df_1m_ind
                ctx.df_map[s]["5m"] = df_5m_ind
                ctx.df_map[s]["15m"] = df_15m_ind
                ctx.df_map[s]["30m"] = df_30m_ind
                ctx.df_map[s]["1h"] = df_1h_ind
                ctx.df_map[s]["4h"] = df_4h_ind
                ctx.df_map[s]["1d"] = df_1d_ind
                ctx.df_map[s]["1w"] = df_1w_ind




            # 12) strategy analyze
            await strategy.analyze_data()
            await asyncio.sleep(300)

        except Exception as e:
            log(f"[loop_data_collector] => {e}\n{traceback.format_exc()}", "error")
            print(f"[DEBUG] Exception in loop_data_collector => {e}\n{traceback.format_exc()}")
            await asyncio.sleep(30)

