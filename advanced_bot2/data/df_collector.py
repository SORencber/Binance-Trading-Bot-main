import pandas as pd
import pandas_ta as ta
from binance import AsyncClient
from core.logging_setup import log
from core.context import SharedContext
import asyncio
import talib
import numpy as np

# 1m verisi => 3000 bar ~ 2 gün
INTERVAL_1M = "1m"
LIMIT_1M    = 3000

# 4h verisi => 200 bar ~ 33 gün
INTERVAL_4H = "4h"
LIMIT_4H    = 200

# 1d verisi => 200 bar ~ ~200 gün
INTERVAL_1D = "1d"
LIMIT_1D    = 200
###############################################################################
# 1) fetch_klines + load_and_clean_data
###############################################################################

async def fetch_klines(client_async: AsyncClient, symbol: str, interval="1m", limit=1000):
    print(f"[DEBUG] fetch_klines => symbol={symbol}, interval={interval}, limit={limit}")
    raw = await client_async.get_klines(symbol=symbol, interval=interval, limit=limit)
    print(f"[DEBUG] fetch_klines => raw klines len={len(raw)}")

    df = pd.DataFrame(raw, columns=[
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

###############################################################################
# 2) Resample ve Partial Bar
###############################################################################

def resample_ohlc(df, rule='5min'):
    print(f"[DEBUG] resample_ohlc => rule={rule}, input df.shape={df.shape}")
    df_ = df.set_index('timestamp').sort_index()
    agg_dict = {
        'Open':   'first',
        'High':   'max',
        'Low':    'min',
        'Close':  'last',
        'Volume': 'sum'
    }
    df_res = df_.resample(rule).agg(agg_dict)
    df_res.dropna(subset=['Open','High','Low','Close'], inplace=True)
    df_res = df_res.reset_index()
    print(f"[DEBUG] resample_ohlc => output df_res.shape={df_res.shape}")
    return df_res

def build_partial_bar(df_1m, rule='5min'):
    print(f"[DEBUG] build_partial_bar => rule={rule}, input df_1m.shape={df_1m.shape}")
    df_1m = df_1m.copy()
    df_1m.sort_values('timestamp', inplace=True)
    df_1m.reset_index(drop=True, inplace=True)

    df_res = resample_ohlc(df_1m, rule=rule)
    if df_1m.empty:
        print("[DEBUG] build_partial_bar => df_1m is empty, returning df_res directly.")
        return df_res

    # Son 1m satır
    last_row = df_1m.iloc[-1]
    last_ts = last_row['timestamp']
    print(f"[DEBUG] build_partial_bar => last 1m timestamp={last_ts}, close={last_row['Close']:.2f}")

    # bar_start hesaplama
    freq_minutes = 5
    if rule.endswith("min"):
        freq_minutes = int(rule.replace("min",""))
    elif rule.endswith("h"):
        freq_minutes = int(rule.replace("h","")) * 60
    elif rule.endswith("d"):
        freq_minutes = int(rule.replace("d","")) * 1440

    m_of_day = last_ts.hour * 60 + last_ts.minute
    bar_start_minute = (m_of_day // freq_minutes) * freq_minutes
    start_hour = bar_start_minute // 60
    start_minute = bar_start_minute % 60
    bar_start = pd.Timestamp(
        year=last_ts.year, month=last_ts.month, day=last_ts.day,
        hour=start_hour, minute=start_minute, tz=last_ts.tz
    )
    bar_end = bar_start + pd.Timedelta(minutes=freq_minutes)

    partial_mask = (df_1m['timestamp'] >= bar_start) & (df_1m['timestamp'] < bar_end)
    df_partial = df_1m.loc[partial_mask].copy()
    print(f"[DEBUG] build_partial_bar => partial_mask sum={partial_mask.sum()} => {bar_start}–{bar_end}")

    if not df_partial.empty:
        p_open  = df_partial.iloc[0]['Open']
        p_high  = df_partial['High'].max()
        p_low   = df_partial['Low'].min()
        p_close = df_partial.iloc[-1]['Close']
        p_vol   = df_partial['Volume'].sum()

        print(f"[DEBUG] build_partial_bar => partial last O/H/L/C/V=({p_open:.2f}/{p_high:.2f}/{p_low:.2f}/{p_close:.2f}/{p_vol:.2f})")

        partial_row = {
            'timestamp': bar_start,
            'Open': p_open,
            'High': p_high,
            'Low': p_low,
            'Close': p_close,
            'Volume': p_vol
        }

        exist_idx = df_res.index[df_res['timestamp'] == bar_start]
        if len(exist_idx) == 0:
            print("[DEBUG] build_partial_bar => appended new partial_row to df_res")
            df_res = pd.concat([df_res, pd.DataFrame([partial_row])], ignore_index=True)
        else:
            print("[DEBUG] build_partial_bar => updated existing partial_row in df_res")
            idx = exist_idx[0]
            df_res.at[idx, 'Open']   = p_open
            df_res.at[idx, 'High']   = p_high
            df_res.at[idx, 'Low']    = p_low
            df_res.at[idx, 'Close']  = p_close
            df_res.at[idx, 'Volume'] = p_vol

    df_res.sort_values('timestamp', inplace=True)
    df_res.reset_index(drop=True, inplace=True)
    print(f"[DEBUG] build_partial_bar => final df_res.shape={df_res.shape}")
    return df_res

###############################################################################
# 3) Indikatörler
###############################################################################

def compute_indicators_4h(df, prefix="", price_col='Close'):
    print(f"[DEBUG] compute_indicators => prefix={prefix}, input df.shape={df.shape}")
    df = df.copy()
    # Bazı popüler MA'ler
    df[f'{prefix}SMA_10'] = df[price_col].rolling(10).mean()
    df[f'{prefix}SMA_20'] = df[price_col].rolling(20).mean()
    df[f'{prefix}EMA_9']  = df[price_col].ewm(span=9).mean()
    df[f'{prefix}EMA_21'] = df[price_col].ewm(span=21).mean()
    df[f'{prefix}EMA_50'] = df[price_col].ewm(span=50).mean()

    macd, macd_signal, macd_hist = talib.MACD(df[price_col], fastperiod=12, slowperiod=26, signalperiod=9)
    df[f'{prefix}MACD']        = macd
    df[f'{prefix}MACD_Signal'] = macd_signal
    df[f'{prefix}MACD_Hist']   = macd_hist

    df[f'{prefix}RSI_14'] = talib.RSI(df[price_col], timeperiod=14)

    # Bollinger Bands
    upb, midb, lowb = talib.BBANDS(df[price_col], timeperiod=20, nbdevup=2, nbdevdn=2)
    df[f'{prefix}BB_Upper'] = upb
    df[f'{prefix}BB_Mid']   = midb
    df[f'{prefix}BB_Lower'] = lowb

    # ATR 14
    df[f'{prefix}ATR_14'] = talib.ATR(df['High'], df['Low'], df[price_col], timeperiod=14)
    # ADX
    df[f'{prefix}ADX_14'] = talib.ADX(df['High'], df['Low'], df[price_col], timeperiod=14)
    # MFI
    df[f'{prefix}MFI_14'] = talib.MFI(df['High'], df['Low'], df[price_col], df['Volume'], timeperiod=14)
    # OBV
    df[f'{prefix}OBV'] = talib.OBV(df[price_col], df['Volume'])
    # Stoch (slow)
    slowk, slowd = talib.STOCH(df['High'], df['Low'], df[price_col],
                               fastk_period=5, slowk_period=3, slowk_matype=0,
                               slowd_period=3, slowd_matype=0)
    df[f'{prefix}STOCH_K'] = slowk
    df[f'{prefix}STOCH_D'] = slowd

    # CCI
    df[f'{prefix}CCI_14'] = talib.CCI(df['High'], df['Low'], df[price_col], timeperiod=14)
    # Williams %R
    df[f'{prefix}WILLR_14'] = talib.WILLR(df['High'], df['Low'], df[price_col], timeperiod=14)
    # Ultimate Oscillator
    df[f'{prefix}ULTOSC'] = talib.ULTOSC(df['High'], df['Low'], df[price_col],
                                         timeperiod1=7, timeperiod2=14, timeperiod3=28)
    # ROC
    df[f'{prefix}ROC_10'] = talib.ROC(df[price_col], timeperiod=10)
    # AD
    df[f'{prefix}AD'] = talib.AD(df['High'], df['Low'], df[price_col], df['Volume'])
    # TRIX
    df[f'{prefix}TRIX_14'] = talib.TRIX(df[price_col], timeperiod=14)
    # MOM
    df[f'{prefix}MOM_14'] = talib.MOM(df[price_col], timeperiod=14)
    # NATR
    df[f'{prefix}NATR_14'] = talib.NATR(df['High'], df['Low'], df[price_col], timeperiod=14)

    print(f"[DEBUG] compute_indicators => output df.shape={df.shape}")
    print(f"[DEBUG] compute_indicators => NaN summary:\n{df.isna().sum()}")

    # Ek olarak, 1h prefix ise son satırı gösterelim
    if prefix.startswith("1h"):
        last_idx = len(df) - 1
        last_row = df.iloc[last_idx]
        print(f"[DEBUG] => {prefix} last_row index={last_idx}")
        print(last_row)
        # Örnek: 1h_RSI_14
        rsi_val = last_row.get(f"{prefix}RSI_14", None)
        print(f"[DEBUG] => {prefix}RSI_14 last row => {rsi_val}")

    return df

def merge_asof_anyfreq(base_df, other_df, suffix=''):
    base_df_ = base_df.sort_values('timestamp')
    other_df_= other_df.sort_values('timestamp')
    merged = pd.merge_asof(
        base_df_, other_df_, on='timestamp', direction='backward', suffixes=('', suffix)
    )
    return merged

###############################################################################
# 4) MTF + partial => build_multi_timeframe_data
###############################################################################

def build_multi_timeframe_data(df_1m):
    print("[DEBUG] build_multi_timeframe_data => START")
    print(f"[DEBUG] build_multi_timeframe_data => input df_1m.shape={df_1m.shape}")

    df_1m_ = df_1m.copy()

    # 1) 1m'ye indikatör
    df_1m_ = compute_indicators(df_1m_, prefix='1m_')

    # 5m partial
    df_5m = build_partial_bar(df_1m_, '5min')
    df_5m = compute_indicators(df_5m, prefix='5m_')

    # 10m partial
    df_10m = build_partial_bar(df_1m_, '10min')
    df_10m = compute_indicators(df_10m, prefix='10m_')

    # 15m partial
    df_15m = build_partial_bar(df_1m_, '15min')
    df_15m = compute_indicators(df_15m, prefix='15m_')

    # 1h partial
    df_1h = build_partial_bar(df_1m_, '1h')
    df_1h = compute_indicators(df_1h, prefix='1h_')

    # Merge
    merged_5m  = merge_asof_anyfreq(df_1m_, df_5m,  suffix='_5m')
    merged_10m = merge_asof_anyfreq(merged_5m, df_10m,  suffix='_10m')
    merged_15m = merge_asof_anyfreq(merged_10m, df_15m, suffix='_15m')
    df_final   = merge_asof_anyfreq(merged_15m, df_1h,  suffix='_1h')

    print(f"[DEBUG] build_multi_timeframe_data => before fill, df_final.shape={df_final.shape}")
    # fill
    df_final.fillna(0, inplace=True)  # opsiyonel
    df_final.ffill(inplace=True)
    df_final.bfill(inplace=True)
    print(f"[DEBUG] build_multi_timeframe_data => after fill, df_final.shape={df_final.shape}")
    print("[DEBUG] build_multi_timeframe_data => DONE")

    return df_final



##############################################################################
# 4) FİYAT YAPISI => Pivot & HH-HL / LH-LL
##############################################################################

def detect_pivots(df: pd.DataFrame, window:int=3):
    """
    Basit pivot_high / pivot_low 
    """
    df = df.copy()
    pivot_high= [False]*len(df)
    pivot_low = [False]*len(df)
    highs= df["High"].values
    lows = df["Low"].values

    for i in range(window, len(df)-window):
        h_i= highs[i]
        l_i= lows[i]
        is_high= True
        is_low = True
        for j in range(i-window, i+window+1):
            if j<0 or j>=len(df):
                continue
            if highs[j]> h_i:
                is_high=False
            if lows[j]< l_i:
                is_low=False
            if not (is_high or is_low):
                break
        pivot_high[i]= is_high
        pivot_low[i] = is_low

    df["pivot_high"]= pivot_high
    df["pivot_low"] = pivot_low
    return df
def label_price_structure(df: pd.DataFrame, prefix:str="", window:int=3):
    """
    pivot_high => HH vs LH
    pivot_low  => HL vs LL
    dd. yoy
    """
    df= detect_pivots(df, window=window)
    df[f"{prefix}structure"]= [None]*len(df)

    last_type= None
    last_val = None

    for i in range(len(df)):
        if df.at[i,"pivot_high"]:
            # onceki pivot da high mı
            if last_type=="HIGH":
                # HH mi LH mi
                if df.at[i,"High"]> last_val:
                    df.at[i,f"{prefix}structure"]="HH"
                else:
                    df.at[i,f"{prefix}structure"]="LH"
            else:
                df.at[i,f"{prefix}structure"]="HIGH"
            last_type= "HIGH"
            last_val = df.at[i,"High"]
        elif df.at[i,"pivot_low"]:
            if last_type=="LOW":
                if df.at[i,"Low"]> last_val:
                    df.at[i,f"{prefix}structure"]="HL"
                else:
                    df.at[i,f"{prefix}structure"]="LL"
            else:
                df.at[i,f"{prefix}structure"]="LOW"
            last_type= "LOW"
            last_val = df.at[i,"Low"]
    return df
# ----------------------------------------------------------------------------
# C) INDIKATÖR HESAPLAMASI 4h and 1d 
# ----------------------------------------------------------------------------

def compute_indicators(df, prefix=""):
    """
    ADX(14), RSI(14), MACD(12,26,9), Boll(20,2), ATR(14) => Hepsi
    """
    df = df.copy()
    close = df["Close"]

    # ADX
    df[f"{prefix}ADX_14"] = talib.ADX(df["High"], df["Low"], close, timeperiod=14)

    # RSI
    df[f"{prefix}RSI_14"] = talib.RSI(close, timeperiod=14)

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df[f"{prefix}MACD"] = macd
    df[f"{prefix}MACD_Signal"] = macd_signal
    df[f"{prefix}MACD_Hist"] = macd_hist

    # Boll
    up, mid, low = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    df[f"{prefix}BB_Up"] = up
    df[f"{prefix}BB_Mid"] = mid
    df[f"{prefix}BB_Low"] = low

    # ATR
    df[f"{prefix}ATR_14"] = talib.ATR(df["High"], df["Low"], close, timeperiod=14)

    return df
###############################################################################
# 5) loop_data_collector
###############################################################################

async def loop_data_collector(ctx: SharedContext, strategy):
    while True:
        try:
            for s in ctx.config["symbols"]:
                print(f"[DEBUG] loop_data_collector => symbol={s}")

                # 1) 1m veriyi çek
                df = await fetch_klines(ctx.client_async, s, "1m", 1000)
                print(f"[DEBUG] loop_data_collector => fetched df.shape={df.shape}")

                # 2) Data temizleme
                df = load_and_clean_data(df)
                print(f"[DEBUG] loop_data_collector => cleaned df.shape={df.shape}")

                # 3) MTF veriyi oluştur
                df_all = build_multi_timeframe_data(df)
                print(f"[DEBUG] loop_data_collector => df_all.shape={df_all.shape}")

                # 4) Kaydet
                if s not in ctx.df_map:
                    ctx.df_map[s] = {}
                ctx.df_map[s]["1m"] = df_all

                # NaN kolonları
                cols_with_nan = df_all.columns[df_all.isna().any()]
                print(f"[DEBUG] loop_data_collector => NaN columns: {cols_with_nan.tolist()}")

                # CSV
                #df_all.to_excel("data/price_data.xlsx", index=False)

                # Son satır debug
                last_row = df_all.iloc[-1]
                
                rsi_1h = last_row.get("1h_RSI_14", None)
                print(f"[DEBUG] => Last row: {last_row['timestamp']} close={last_row['Close']}, 1h_RSI_14={rsi_1h}")
                #exit(1)
            await strategy.analyze_data()
            await asyncio.sleep(60)

        except Exception as e:
            log(f"[loop_data_collector] => {e}", "error")
            print(f"[DEBUG] Exception in loop_data_collector => {e}")
            await asyncio.sleep(30)
