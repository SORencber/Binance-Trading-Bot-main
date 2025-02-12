# data/tf_indicators.py

# Bu dosya: 
#   - 1) "indicator_config" => her TF parametreleri
#   - 2) Gelişmiş fonksiyonlar (Heikin, SuperTrend, T3, Vortex vs.)
#   - 3) 7 ayrı fonksiyon:
#       calculate_indicators_1m(df),
#       calculate_indicators_5m(df),
#       calculate_indicators_15m(df),
#       calculate_indicators_30m(df),
#       calculate_indicators_1h(df),
#       calculate_indicators_4h(df),
#       calculate_indicators_1d(df)
#     Her birinde: RSI, ADX, ATR, MACD, Boll, Stoch, Ichimoku, SuperTrend, 
#     Vortex, T3, Heikin, + fibo, pivot, cmo, stochRSI, MVRV, NVT, Candle Patterns...
###############################################################################

import pandas as pd
import numpy as np
import talib

########################################
# 1) İNDİKATÖR PARAMETRE SÖZLÜĞÜ
########################################
INDICATOR_CONFIG = {
    "1m": {
        "RSI": 7,
        "ADX": 7,
        "ATR": 10,
        "BOLL_period": 14,
        "BOLL_stddev": 2,
        "MACD_fast": 6,
        "MACD_slow": 13,
        "MACD_signal": 5,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },
    "5m": {
        "RSI": 9,
        "ADX": 9,
        "ATR": 14,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 8,
        "MACD_slow": 17,
        "MACD_signal": 6,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },
    "15m": {
        "RSI": 14,
        "ADX": 14,
        "ATR": 14,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 12,
        "MACD_slow": 26,
        "MACD_signal": 9,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },
    "30m": {
        "RSI": 14,
        "ADX": 14,
        "ATR": 14,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 12,
        "MACD_slow": 26,
        "MACD_signal": 9,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },
    "1h": {
        "RSI": 14,
        "ADX": 14,
        "ATR": 14,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 12,
        "MACD_slow": 26,
        "MACD_signal": 9,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },
    "4h": {
        "RSI": 14,
        "ADX": 14,
        "ATR": 21,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 12,
        "MACD_slow": 26,
        "MACD_signal": 9,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },
    "1w": {
        "RSI": 14,
        "ADX": 14,
        "ATR": 21,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 12,
        "MACD_slow": 26,
        "MACD_signal": 9,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    },  "1d": {
        "RSI": 14,
        "ADX": 14,
        "ATR": 21,
        "BOLL_period": 20,
        "BOLL_stddev": 2,
        "MACD_fast": 12,
        "MACD_slow": 26,
        "MACD_signal": 9,
        "STOCH_fastk": 5,
        "STOCH_slowk": 3,
        "STOCH_slowd": 3,
        "ICHIMOKU_tenkan": 9,
        "ICHIMOKU_kijun": 26,
        "ICHIMOKU_spanB": 52,
        "SUPER_period": 7,
        "SUPER_multiplier": 3,
        "VORTEX_period": 14,
        "T3_period": 9,
        "T3_vfactor": 0.7
    }
}


########################################
# 2) GELİŞMİŞ CUSTOM FONKSİYONLAR
########################################

def atr_simplified(df: pd.DataFrame, period: int=14) -> pd.Series:
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - df['Close'].shift(1)).abs()
    tr3 = (df['Low']  - df['Close'].shift(1)).abs()
    tr  = tr1.combine(tr2, max).combine(tr3, max)
    return tr.rolling(period).mean()

def supertrend(df: pd.DataFrame, period=7, multiplier=3):
    st_df = pd.DataFrame(index=df.index)
    st_df['STR_ATR'] = atr_simplified(df, period)
    
    mid = (df['High'] + df['Low']) / 2
    st_df['BASIC_UPPER'] = mid + multiplier * st_df['STR_ATR']
    st_df['BASIC_LOWER'] = mid - multiplier * st_df['STR_ATR']
    
    st_df['SuperTrend'] = (st_df['BASIC_UPPER'] + st_df['BASIC_LOWER'])/2
    st_df['SuperTrend_Direction'] = np.where(df['Close']> st_df['SuperTrend'], 1, -1)
    return st_df

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    df_ha = pd.DataFrame(index=df.index)
    df_ha['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    ha_open = [(df['Open'].iloc[0] + df['Close'].iloc[0]) / 2.0]
    for i in range(1, len(df)):
        ha_open.append( (ha_open[i-1] + df_ha['HA_Close'].iloc[i-1]) / 2.0 )
    df_ha['HA_Open'] = ha_open

    df_ha['HA_High'] = df[['High','Low','Close','Open']].max(axis=1)
    df_ha['HA_Low']  = df[['High','Low','Close','Open']].min(axis=1)
    df_ha['HA_Trend'] = (df_ha['HA_Close'] > df_ha['HA_Open']).astype(int)
    return df_ha

def vortex_indicator(df: pd.DataFrame, period=14):
    df_v = pd.DataFrame(index=df.index)
    trp1 = (df['High'].combine(df['Close'].shift(1), max) - df['Low'].combine(df['Close'].shift(1), min)).abs()
    df_v['TR'] = trp1.rolling(period).sum()
    
    vm_plus  = (df['High'] - df['Low'].shift(1)).abs().rolling(period).sum()
    vm_minus = (df['Low']  - df['High'].shift(1)).abs().rolling(period).sum()
    
    df_v['VI+'] = vm_plus  / df_v['TR']
    df_v['VI-'] = vm_minus / df_v['TR']
    return df_v

def T3_moving_average(series: pd.Series, period=9, v_factor=0.7):
    c1 = -v_factor**3
    c2 = 3*v_factor**2 + 3*v_factor**3
    c3 = -6*v_factor**2 - 3*v_factor - 3*v_factor**3
    c4 = 1 + 3*v_factor + v_factor**3 + 3*v_factor**2

    ema1 = series.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    ema4 = ema3.ewm(span=period, adjust=False).mean()
    ema5 = ema4.ewm(span=period, adjust=False).mean()
    ema6 = ema5.ewm(span=period, adjust=False).mean()

    T3 = c1*ema6 + c2*ema5 + c3*ema4 + c4*ema3
    return T3

def oi_bollinger(df: pd.DataFrame, column: str="x", 
                 period: int=20, nbdev: int=2, suffix: str="") -> pd.DataFrame:
    """
    Open Interest için Bollinger Band hesaplar. 
    df[suffix+"OI_BBMid"], df[suffix+"OI_BBUp"], df[suffix+"OI_BBLow"] sütunlarına yazar.
    """
    ma_oi = df[column].rolling(period).mean()
    std_oi = df[column].rolling(period).std()

    df["OI_BBMid"] = ma_oi
    df["OI_BBUp"]  = ma_oi + nbdev * std_oi
    df["OI_BBLow"] = ma_oi - nbdev * std_oi
    #print(suffix +"OI_BBLow------------------------>>>>>>>>>>>",df[suffix + "_OI_BBLow"].iloc[-1])
    #print(suffix +"OI_BBUp------------------------>>>>>>>>>>>",df[suffix + "_OI_BBUp"].iloc[-1])
    #print("OI_BBMid------------------------>>>>>>>>>>>",df[suffix + "_OI_BBMid"].iloc[-1])

    return df

def oi_roc(df: pd.DataFrame, column: str="x", 
           period: int=10, suffix: str="") -> pd.DataFrame:
    """
    OI ROC => ((OI - OI.shift(period)) / OI.shift(period)) * 100
    """
    shifted = df[column].shift(period)
    df["OI_ROC"] = (df[column] - shifted) / (shifted + 1e-9) * 100
    #print(suffix +"_OI_ROC------------------------>>>>>>>>>>>",df[suffix + "_OI_ROC"].iloc[-1])

    return df

def add_oi_indicators(
    df: pd.DataFrame,
    suffix: str = "",
    period_oi: int = 14,
    period_boll: int = 20,
    nbdev: int = 2,
    period_roc: int = 10
) -> pd.DataFrame:
    """
    df'ye şu sütunları ekler (suffix ekleyerek):
      - OI_MA, OI_STD
      - OI_Zscore
      - OI_RSI
      - OI_BBMid, OI_BBUp, OI_BBLow
      - OI_ROC
    """
    if "sumOpenInterestValue" not in df.columns:
        return df  # OI yoksa pas

    # 1) Rolling Mean & Std
    df["OI_MA"] = df["sumOpenInterestValue"].rolling(period_oi).mean()
    df["OI_STD"] = df["sumOpenInterestValue"].rolling(period_oi).std()

    # 2) Zscore => (OI - MA) / STD
    df["OI_Zscore"] = (
        df["sumOpenInterestValue"] - df["OI_MA"]
    ) / (df["OI_STD"] + 1e-9)

    # 3) OI RSI benzeri
    delta = df["sumOpenInterestValue"].diff(1)
    
    gain = delta.clip(lower=0).abs()
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.rolling(period_oi).mean()
    avg_loss = loss.rolling(period_oi).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    #print("yeni degerler---->",delta,gain,loss,avg_gain,avg_loss)
    df["OI_RSI"] = 100 - (100 / (1 + rs))
    df["OI_delta"] = df["sumOpenInterestValue"].diff(1)
    # 4) OI Bollinger
    df = oi_bollinger(df, column="sumOpenInterestValue", 
                      period=period_boll, nbdev=nbdev, suffix=suffix)

    # 5) OI ROC
    df = oi_roc(df, column="sumOpenInterestValue", 
                period=period_roc, suffix=suffix)
    
    return df



def detect_holy_grail_signals(df: pd.DataFrame, adx_threshold=30, prefix="_1h") -> pd.DataFrame:
    df = df.copy()
    # Sinyal kolonunu oluşturuyoruz
    sigcol = "HolyGrailSignal"
    df[sigcol] = 0  # Tüm satırlara 0 atandı

    # Son barı alalım
    last_idx = df.index[-1]
    last_bar = df.iloc[-1]

    # İlgili değerleri alalım
    adx_val   = last_bar.loc[f"ADX{prefix}"]
    plus_di   = last_bar.loc[f"DI_plus{prefix}"]
    minus_di  = last_bar.loc[f"DI_minus{prefix}"]
    ema_val   = last_bar.loc[f"EMA_20{prefix}"]
    c_close   = last_bar.loc["Close"]
    c_low     = last_bar.loc["Low"]
    c_high    = last_bar.loc["High"]

    #print(adx_val, adx_threshold, plus_di, minus_di, c_close, ema_val, c_high, c_low)

    # Eğer +DI > -DI ise up-trend
    if plus_di > minus_di:
        # Eğer c_low <= EMA20 ve c_close > EMA20 ise alış sinyali
        if (c_low <= ema_val) and (c_close > ema_val):
            df.at[last_idx, sigcol] = +1
    else:
        # Eğer -DI > +DI ise down-trend
        if (c_high >= ema_val) and (c_close < ema_val):
            df.at[last_idx, sigcol] = -1

    return df

def simple_holy_grail_stoploss(row, prefix="_1h", multiplier=1.5):
    """
    Stop = cClose +/- ATR * multiplier
    """
    sig = row[f"HolyGrailSignal"]
    c_close = row[f"Close"]
    atr_val = row[f"ATR{prefix}"]
    if sig==+1:
        return c_close - atr_val*multiplier
    elif sig==-1:
        return c_close + atr_val*multiplier
    return None

def holy_grail_all_timeframes(df: pd.DataFrame, prefix: str = "_1h") -> pd.DataFrame:
    """
    Verilen DataFrame üzerinde Holy Grail sinyallerini hesaplar.
    
    Adımlar:
      1) detect_holy_grail_signals fonksiyonu ile temel sinyal (HolyGrailSignal) hesaplanır.
      2) simple_holy_grail_stoploss fonksiyonu kullanılarak ATR tabanlı stop-loss sütunu eklenir.
      3) prefix değerine göre kullanılacak RSI sütunu belirlenir.
      4) Son bardaki (örneğin, df.iloc[-1]) HolyGrailSignal değeri ve RSI değeri kullanılarak
         "HolyGrail_FinalSignal" sütunu güncellenir.
           - Eğer sinyal +1 (bullish) ise ve ilgili RSI değeri (örneğin <35) teyit ediyorsa final sinyal +1,
             aksi halde 0 (hold).
           - Eğer sinyal -1 (bearish) ise ve ilgili RSI değeri (örneğin >60) teyit ediyorsa final sinyal -1,
             aksi halde 0.
      5) Fonksiyon güncellenmiş DataFrame’i döner.
    
    Parametreler:
      - df: İşlenecek fiyat ve indikatör verilerini içeren DataFrame.
      - prefix: İlgili zaman dilimini temsil eden ek (örn. "_1h", "_4h", "_1d" vs.).
    
    Dönüş:
      - df: "HolyGrail_StopLoss" ve "HolyGrail_FinalSignal" sütunlarını içeren DataFrame.
    """
    # 1) Önce temel Holy Grail sinyalini hesaplayalım
    df = detect_holy_grail_signals(df, adx_threshold=30, prefix=prefix)
    
    # 2) Stop-loss sütununu ekleyelim (örneğin, multiplier=1.5)
    holy_stop_col = "HolyGrail_StopLoss"
    df[holy_stop_col] = df.apply(
        lambda row: simple_holy_grail_stoploss(row, prefix=prefix, multiplier=1.5),
        axis=1
    )
    
    # 3) Hangi RSI sütununu kullanacağımızı belirleyelim
    # (Burada örnek olarak, 1h için 15m, 4h için 1h, 1d için 4h, 30m için 5m ve 15m için 5m RSI kullanılıyor.
    #  Gerekirse ayarlayabilirsiniz.)
    if prefix == "_1h":
        rsi_col = "RSI_15m"
    elif prefix == "_4h":
        rsi_col = "RSI_1h"
    elif prefix == "_1d":
        rsi_col = "RSI_4h"
    elif prefix == "_30m":
        rsi_col = "RSI_5m"
    elif prefix == "_15m":
        rsi_col = "RSI_5m"
    else:
        rsi_col = "RSI_1m"
    
    # 4) Final sinyal sütununu oluşturup, tüm satırlara önce 0 değeri atayalım
    df["HolyGrail_FinalSignal"] = 0

    # Son barın indeksini alalım
    last_idx = df.index[-1]
    # Temel sinyalin sütun adı (detect_holy_grail_signals fonksiyonunda oluşturulmuş)
    sig_col = "HolyGrailSignal"
    
    # Son bardaki temel sinyal değerini alalım (örn. +1 veya -1)
    sig_val = df.at[last_idx, sig_col]

    # Final sinyal için koşullar:
    if sig_val == 1:
        # Bullish senaryosu: Eğer ilgili RSI sütunu mevcutsa ve son satırdaki RSI değeri < 35 ise
        if rsi_col in df.columns:
            rsi_val = df.at[last_idx, rsi_col]
            if rsi_val < 35:
                df.at[last_idx, "HolyGrail_FinalSignal"] = 1
            else:
                df.at[last_idx, "HolyGrail_FinalSignal"] = 0
        else:
            df.at[last_idx, "HolyGrail_FinalSignal"] = 1
    elif sig_val == -1:
        # Bearish senaryosu: Eğer ilgili RSI sütunu mevcutsa ve son satırdaki RSI değeri > 60 ise
        if rsi_col in df.columns:
            rsi_val = df.at[last_idx, rsi_col]
            if rsi_val > 60:
                df.at[last_idx, "HolyGrail_FinalSignal"] = -1
            else:
                df.at[last_idx, "HolyGrail_FinalSignal"] = 0
        else:
            df.at[last_idx, "HolyGrail_FinalSignal"] = -1
    else:
        # Sinyal 0 veya başka bir durum ise final sinyal 0 (hold) olarak bırakılır.
        df.at[last_idx, "HolyGrail_FinalSignal"] = 0

    return df


########################################
# 3) HER TF İÇİN DEVASA FONKSİYONLAR
########################################

def calculate_indicators_1m(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 dakikalık veride, 'INDICATOR_CONFIG["1m"]' parametreleriyle
    TÜM kapsamlı (fibo, pivot, cmo, stochRSI, MVRV, Candle Patterns, Heikin, 
    SuperTrend, T3, Vortex, MACD, Boll vs.) hesaplar.
    """
    cfg = INDICATOR_CONFIG["1m"]
    df = df.copy()
    
    df['Volume'] = (
    df['Volume']
    .astype(str)
    # Remove everything that isn't digits or a decimal point:
    .str.replace(r'[^\d.]+', '', regex=True)
    .astype(float)
)
    open_= df["Open"]
    df['Close'] = (
    df['Close']
    .astype(str)
    .str.replace(r'[^\d.]+', '', regex=True)
    .astype(float)
)
    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    # df["Open_1m"] =df["Open"]
    # df["Close_1m"] =df["Close"]
    # df["High_1m"] =df["High"]
    # df["Low_1m"] =df["Low"]
    # df["Volume_1m"] =df["Volume"]
    



    # RSI, ADX, ATR
    df["RSI_1m"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_1m"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_1m"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # MACD
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=cfg["MACD_fast"],
        slowperiod=cfg["MACD_slow"],
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_1m"]       = macd
    df["MACDSig_1m"]    = macd_signal
    df["MACDHist_1m"]   = macd_hist

    # Bollinger
    up, mid, lw = talib.BBANDS(close, timeperiod=cfg["BOLL_period"],
                               nbdevup=cfg["BOLL_stddev"], nbdevdn=cfg["BOLL_stddev"])
    df["BBUp_1m"]  = up
    df["BBMid_1m"] = mid
    df["BBLow_1m"] = lw

    # Stoch
    slowk, slowd = talib.STOCH(high, low, close,
                               fastk_period=cfg["STOCH_fastk"],
                               slowk_period=cfg["STOCH_slowk"],
                               slowd_period=cfg["STOCH_slowd"])
    df["StochK_1m"] = slowk
    df["StochD_1m"] = slowd

    # Ichimoku
    tenkan_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    tenkan_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_1m"] = (tenkan_high + tenkan_low)/2

    kijun_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    kijun_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_1m"] = (kijun_high + kijun_low)/2

    span_a = (df["Ichi_Tenkan_1m"] + df["Ichi_Kijun_1m"])/2
    df["Ichi_SpanA_1m"] = span_a.shift(cfg["ICHIMOKU_kijun"])

    spb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    spb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_1m"] = ((spb_high + spb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # SuperTrend
    st_df = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_1m"]    = st_df["SuperTrend"]
    df["SuperTD_1m"]       = st_df["SuperTrend_Direction"]

    # Vortex
    vi_df = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_1m"] = vi_df["VI+"]
    df["VI-_1m"] = vi_df["VI-"]

    # T3
    df["T3_1m"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # Heikin-Ashi
    ha_df = heikin_ashi(df)
    df["HA_Open_1m"]  = ha_df["HA_Open"]
    df["HA_Close_1m"] = ha_df["HA_Close"]
    df["HA_High_1m"]  = ha_df["HA_High"]
    df["HA_Low_1m"]   = ha_df["HA_Low"]
    df["HA_Trend_1m"] = ha_df["HA_Trend"]

    # StochRSI
    rsi_ = talib.RSI(close, 14)
    stoch_rsi = (rsi_ - rsi_.rolling(14).min()) / (rsi_.rolling(14).max() - rsi_.rolling(14).min() +1e-9)
    df["StochRSI_1m"] = stoch_rsi

    # CMO
    df["CMO_1m"] = talib.CMO(close, timeperiod=14)

    # OBV, CMF
    df["OBV_1m"] = talib.OBV(df["Close"], df["Volume"])
    cmf_val = ((2*close - high - low)/(high - low +1e-9)) * vol
    df["CMF_1m"] = cmf_val.rolling(window=20).mean()

    # Momentum, ROC
    df["MOM_1m"] = talib.MOM(close, timeperiod=10)
    df["ROC_1m"] = talib.ROC(close, timeperiod=10)

    # Candle Patterns
    df["Candle_Body_1m"] = (close - open_).abs()
    df["Upper_Wick_1m"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_1m"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_1m"]   = ((df["Lower_Wick_1m"] > 2*df["Candle_Body_1m"]) & (df["Upper_Wick_1m"]< df["Candle_Body_1m"])).astype(int)
    df["CDL_ENGULFING_1m"] = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])

    # Pivot & Fibo
      # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_1m"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # On-chain Sim: MVRV, NVT
    df["MarketCap_1m"]   = close * vol
    df["RealizedCap_1m"] = close*(vol.cumsum()/max(1,len(df)))
    rolling_std = close.rolling(365).std()
    df["MVRV_Z_1m"]      = (df["MarketCap_1m"] - df["RealizedCap_1m"]) / (rolling_std+1e-9)
    df["NVT_1m"]         = df["MarketCap_1m"] / (vol+1e-9)

    # future_return + Label (ML)
    df["future_ret_1m"] = (close.shift(-1) - close)/(close+1e-9)
    df["Label_1m"]      = (df["future_ret_1m"]>0).astype(int)
    # ADX, DI+, DI-

    df['DI_plus_1m'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_1m'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_1m'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_1m'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_1m'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df


def calculate_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    """
    5 dakikalık veride, 'INDICATOR_CONFIG["5m"]' parametreleriyle
    tüm aynı büyük set (fibo, pivot, cmo, stochRSI, MVRV, Candle Patterns, 
    Heikin, SuperTrend, T3, Vortex, MACD, Boll vs.) 
    """
    cfg = INDICATOR_CONFIG["5m"]
    df = df.copy()
    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "5m",
        14,
        20,
        2,
         10)
    # RSI, ADX, ATR
    df["RSI_5m"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_5m"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_5m"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # MACD
    macd, macd_sig, macd_hist = talib.MACD(
        close, 
        fastperiod=cfg["MACD_fast"], 
        slowperiod=cfg["MACD_slow"], 
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_5m"]     = macd
    df["MACDSig_5m"]  = macd_sig
    df["MACDHist_5m"] = macd_hist

    # Bollinger
    up, mid, lw = talib.BBANDS(
        close, timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"], nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_5m"]  = up
    df["BBMid_5m"] = mid
    df["BBLow_5m"] = lw

    # Stoch
    sk, sd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_5m"] = sk
    df["StochD_5m"] = sd

    # Ichimoku
    t_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    t_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_5m"] = (t_high + t_low)/2
    kj_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    kj_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_5m"] = (kj_high + kj_low)/2
    sp_a = (df["Ichi_Tenkan_5m"] + df["Ichi_Kijun_5m"])/2
    df["Ichi_SpanA_5m"] = sp_a.shift(cfg["ICHIMOKU_kijun"])

    sb_h = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    sb_l = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_5m"] = ((sb_h + sb_l)/2).shift(cfg["ICHIMOKU_kijun"])

    # SuperTrend
    st5 = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_5m"] = st5["SuperTrend"]
    df["SuperTD_5m"]    = st5["SuperTrend_Direction"]

    # Vortex
    vi5 = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_5m"] = vi5["VI+"]
    df["VI-_5m"] = vi5["VI-"]

    # T3
    df["T3_5m"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # Heikin-Ashi
    ha5 = heikin_ashi(df)
    df["HA_Open_5m"]  = ha5["HA_Open"]
    df["HA_Close_5m"] = ha5["HA_Close"]
    df["HA_High_5m"]  = ha5["HA_High"]
    df["HA_Low_5m"]   = ha5["HA_Low"]
    df["HA_Trend_5m"] = ha5["HA_Trend"]

    # StochRSI
    rsi_ = talib.RSI(close, 14)
    sr5  = (rsi_ - rsi_.rolling(14).min())/(rsi_.rolling(14).max()-rsi_.rolling(14).min()+1e-9)
    df["StochRSI_5m"] = sr5

    # CMO
    df["CMO_5m"] = talib.CMO(close, timeperiod=14)

    # OBV, CMF
    df["OBV_5m"] = talib.OBV(close, vol)
    cmf_ = ((2*close - high - low)/(high - low+1e-9)) * vol
    df["CMF_5m"] = cmf_.rolling(window=20).mean()

    # Momentum, ROC
    df["MOM_5m"] = talib.MOM(close, timeperiod=10)
    df["ROC_5m"] = talib.ROC(close, timeperiod=10)

    # Candle Patterns
    df["Candle_Body_5m"] = (close - open_).abs()
    df["Upper_Wick_5m"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_5m"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_5m"]   = ((df["Lower_Wick_5m"]> 2*df["Candle_Body_5m"]) & (df["Upper_Wick_5m"]< df["Candle_Body_5m"])).astype(int)
    # Engulfing
    df["CDL_ENGULFING_5m"] = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])
        # Pivot & Fibo
        # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_5m"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # On-chain sim
    df["MarketCap_5m"]   = close*vol
    df["RealizedCap_5m"] = close*(vol.cumsum()/max(1,len(df)))
    rol_std = close.rolling(365).std()
    df["MVRV_Z_5m"] = (df["MarketCap_5m"]-df["RealizedCap_5m"])/(rol_std+1e-9)
    df["NVT_5m"]    = df["MarketCap_5m"]/(vol+1e-9)

    # future_return + Label
    df["future_ret_5m"] = (close.shift(-1) - close)/(close+1e-9)
    df["Label_5m"] = (df["future_ret_5m"]>0).astype(int)
   # DI+, DI-

    df['DI_plus_5m'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_5m'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_5m'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_5m'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_5m'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df


def calculate_indicators_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    15 dakikalık veri (en büyük set). 
    """
    cfg = INDICATOR_CONFIG["15m"]
    df = df.copy()
    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "15m",
        14,
        20,
        2,
         10)
    
    # RSI, ADX, ATR
    df["RSI_15m"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_15m"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_15m"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # MACD
    mc, ms, mh = talib.MACD(
        close, 
        fastperiod=cfg["MACD_fast"], 
        slowperiod=cfg["MACD_slow"], 
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_15m"]     = mc
    df["MACDSig_15m"]  = ms
    df["MACDHist_15m"] = mh

    # Bollinger
    up, mid, lw = talib.BBANDS(
        close, timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"], nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_15m"]  = up
    df["BBMid_15m"] = mid
    df["BBLow_15m"] = lw

    # Stoch
    sk, sd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_15m"] = sk
    df["StochD_15m"] = sd

    # Ichimoku
    t_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    t_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_15m"] = (t_high + t_low)/2

    k_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    k_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_15m"] = (k_high + k_low)/2

    sp_a = (df["Ichi_Tenkan_15m"] + df["Ichi_Kijun_15m"])/2
    df["Ichi_SpanA_15m"] = sp_a.shift(cfg["ICHIMOKU_kijun"])

    sb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    sb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_15m"] = ((sb_high + sb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # SuperTrend
    st15 = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_15m"] = st15["SuperTrend"]
    df["SuperTD_15m"]    = st15["SuperTrend_Direction"]

    # Vortex
    vi15 = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_15m"] = vi15["VI+"]
    df["VI-_15m"] = vi15["VI-"]

    # T3
    df["T3_15m"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # Heikin-Ashi
    ha15 = heikin_ashi(df)
    df["HA_Open_15m"]  = ha15["HA_Open"]
    df["HA_Close_15m"] = ha15["HA_Close"]
    df["HA_High_15m"]  = ha15["HA_High"]
    df["HA_Low_15m"]   = ha15["HA_Low"]
    df["HA_Trend_15m"] = ha15["HA_Trend"]

    # StochRSI, CMO
    rsi_ = talib.RSI(close, 14)
    srsi = (rsi_ - rsi_.rolling(14).min())/(rsi_.rolling(14).max()-rsi_.rolling(14).min()+1e-9)
    df["StochRSI_15m"] = srsi
    df["CMO_15m"] = talib.CMO(close, timeperiod=14)

    # OBV, CMF
    df["OBV_15m"] = talib.OBV(close, vol)
    cmf_val = ((2*close - high - low)/(high - low+1e-9)) * vol
    df["CMF_15m"] = cmf_val.rolling(window=20).mean()

    # Momentum, ROC
    df["MOM_15m"] = talib.MOM(close, timeperiod=10)
    df["ROC_15m"] = talib.ROC(close, timeperiod=10)

    # Candle Patterns
    df["Candle_Body_15m"] = (close - open_).abs()
    df["Upper_Wick_15m"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_15m"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_15m"]   = ((df["Lower_Wick_15m"]>2*df["Candle_Body_15m"]) & (df["Upper_Wick_15m"]< df["Candle_Body_15m"])).astype(int)
    df["CDL_ENGULFING_15m"] = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])

    # Pivot & Fibo
        # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_15m"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # On-chain sim
    df["MarketCap_15m"]   = close*vol
    df["RealizedCap_15m"] = close*(vol.cumsum()/max(1,len(df)))
    rolstd = close.rolling(365).std()
    df["MVRV_Z_15m"] = (df["MarketCap_15m"]-df["RealizedCap_15m"])/(rolstd+1e-9)
    df["NVT_15m"]    = df["MarketCap_15m"]/(vol+1e-9)

    # future_return + Label
    df["future_ret_15m"] = (close.shift(-1)-close)/(close+1e-9)
    df["Label_15m"]      = (df["future_ret_15m"]>0).astype(int)
    df['DI_plus_15m'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_15m'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_15m'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_15m'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_15m'] = df['Close'].ewm(span=20, adjust=False).mean()
    holy_grail_all_timeframes(df,"_15m")

    return df


def calculate_indicators_30m(df: pd.DataFrame) -> pd.DataFrame:
    """30 dakikalık veri, config => RSI(14), ADX(14), ATR(14) ... + full set."""
    # Aynı mantık => 15m fonksiyonunun kopyası => parametreler "30m" config'ten.
    cfg = INDICATOR_CONFIG["30m"]
    df = df.copy()
    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "30m",
        14,
        20,
        2,
         10)



 
    # RSI, ADX, ATR
    df["RSI_30m"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_30m"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_30m"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])
    
    # MACD
    mc, ms, mh = talib.MACD(
        close, 
        fastperiod=cfg["MACD_fast"], 
        slowperiod=cfg["MACD_slow"], 
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_30m"]     = mc
    df["MACDSig_30m"]  = ms
    df["MACDHist_30m"] = mh

    # Bollinger
    up, mid, lw = talib.BBANDS(
        close, timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"], nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_30m"]  = up
    df["BBMid_30m"] = mid
    df["BBLow_30m"] = lw

    # Stoch
    sk, sd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_30m"] = sk
    df["StochD_30m"] = sd

    # Ichimoku
    t_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    t_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_30m"] = (t_high + t_low)/2

    k_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    k_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_30m"] = (k_high + k_low)/2

    sp_a = (df["Ichi_Tenkan_30m"] + df["Ichi_Kijun_30m"])/2
    df["Ichi_SpanA_30m"] = sp_a.shift(cfg["ICHIMOKU_kijun"])

    sb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    sb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_30m"] = ((sb_high + sb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # SuperTrend
    st15 = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_30m"] = st15["SuperTrend"]
    df["SuperTD_30m"]    = st15["SuperTrend_Direction"]

    # Vortex
    vi15 = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_30m"] = vi15["VI+"]
    df["VI-_30m"] = vi15["VI-"]

    # T3
    df["T3_30m"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # Heikin-Ashi
    ha15 = heikin_ashi(df)
    df["HA_Open_30m"]  = ha15["HA_Open"]
    df["HA_Close_30m"] = ha15["HA_Close"]
    df["HA_High_30m"]  = ha15["HA_High"]
    df["HA_Low_30m"]   = ha15["HA_Low"]
    df["HA_Trend_30m"] = ha15["HA_Trend"]

    # StochRSI, CMO
    rsi_ = talib.RSI(close, 14)
    srsi = (rsi_ - rsi_.rolling(14).min())/(rsi_.rolling(14).max()-rsi_.rolling(14).min()+1e-9)
    df["StochRSI_30m"] = srsi
    df["CMO_30m"] = talib.CMO(close, timeperiod=14)

    # OBV, CMF
    df["OBV_30m"] = talib.OBV(close, vol)
    cmf_val = ((2*close - high - low)/(high - low+1e-9)) * vol
    df["CMF_30m"] = cmf_val.rolling(window=20).mean()

    # Momentum, ROC
    df["MOM_30m"] = talib.MOM(close, timeperiod=10)
    df["ROC_30m"] = talib.ROC(close, timeperiod=10)

    # Candle Patterns
    df["Candle_Body_30m"] = (close - open_).abs()
    df["Upper_Wick_30m"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_30m"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_30m"]   = ((df["Lower_Wick_30m"]>2*df["Candle_Body_30m"]) & (df["Upper_Wick_30m"]< df["Candle_Body_30m"])).astype(int)
    df["CDL_ENGULFING_30m"] = talib.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"])

    # Pivot & Fibo
       # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_30m"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # On-chain sim
    df["MarketCap_30m"]   = close*vol
    df["RealizedCap_30m"] = close*(vol.cumsum()/max(1,len(df)))
    rolstd = close.rolling(365).std()
    df["MVRV_Z_30m"] = (df["MarketCap_30m"]-df["RealizedCap_30m"])/(rolstd+1e-9)
    df["NVT_30m"]    = df["MarketCap_30m"]/(vol+1e-9)

    # future_return + Label
    df["future_ret_30m"] = (close.shift(-1)-close)/(close+1e-9)
    df["Label_30m"]      = (df["future_ret_30m"]>0).astype(int)
    df['DI_plus_30m'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_30m'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_30m'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_30m'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_30m'] = df['Close'].ewm(span=20, adjust=False).mean()
    holy_grail_all_timeframes(df,"_30m")

    return df

def calculate_indicators_1h(df: pd.DataFrame) -> pd.DataFrame:
    """1 saatlik veri, config => RSI(14), ADX(14), ATR(14) + full set."""
    cfg = INDICATOR_CONFIG["1h"]
    df = df.copy()
    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "1h",
        14,
        20,
        2,
         10)
    # RSI, ADX, ATR
    df["RSI_1h"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_1h"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_1h"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # MACD
    mc, ms, mh = talib.MACD(
        close, 
        fastperiod=cfg["MACD_fast"], 
        slowperiod=cfg["MACD_slow"], 
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_1h"]     = mc
    df["MACDSig_1h"]  = ms
    df["MACDHist_1h"] = mh

    # Bollinger
    up, mid, lw = talib.BBANDS(
        close, timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"], nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_1h"]  = up
    df["BBMid_1h"] = mid
    df["BBLow_1h"] = lw

    # Stoch
    sk, sd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_1h"] = sk
    df["StochD_1h"] = sd

    # Ichimoku
    t_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    t_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_1h"] = (t_high + t_low)/2

    k_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    k_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_1h"] = (k_high + k_low)/2

    sp_a = (df["Ichi_Tenkan_1h"] + df["Ichi_Kijun_1h"])/2
    df["Ichi_SpanA_1h"] = sp_a.shift(cfg["ICHIMOKU_kijun"])

    sb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    sb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_1h"] = ((sb_high + sb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # SuperTrend
    st15 = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_1h"] = st15["SuperTrend"]
    df["SuperTD_1h"]    = st15["SuperTrend_Direction"]

    # Vortex
    vi15 = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_1h"] = vi15["VI+"]
    df["VI-_1h"] = vi15["VI-"]

    # T3
    df["T3_1h"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # Heikin-Ashi
    ha15 = heikin_ashi(df)
    df["HA_Open_1h"]  = ha15["HA_Open"]
    df["HA_Close_1h"] = ha15["HA_Close"]
    df["HA_High_1h"]  = ha15["HA_High"]
    df["HA_Low_1h"]   = ha15["HA_Low"]
    df["HA_Trend_1h"] = ha15["HA_Trend"]

    # StochRSI, CMO
    rsi_ = talib.RSI(close, 14)
    srsi = (rsi_ - rsi_.rolling(14).min())/(rsi_.rolling(14).max()-rsi_.rolling(14).min()+1e-9)
    df["StochRSI_1h"] = srsi
    df["CMO_1h"] = talib.CMO(close, timeperiod=14)

    # OBV, CMF
    df["OBV_1h"] = talib.OBV(close, vol)
    cmf_val = ((2*close - high - low)/(high - low+1e-9)) * vol
    df["CMF_1h"] = cmf_val.rolling(window=20).mean()

    # Momentum, ROC
    df["MOM_1h"] = talib.MOM(close, timeperiod=10)
    df["ROC_1h"] = talib.ROC(close, timeperiod=10)

    # Candle Patterns
    df["Candle_Body_1h"] = (close - open_).abs()
    df["Upper_Wick_1h"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_1h"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_1h"]   = ((df["Lower_Wick_1h"]>2*df["Candle_Body_1h"]) & (df["Upper_Wick_1h"]< df["Candle_Body_1h"])).astype(int)

    # Pivot & Fibo
        # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_1h"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # On-chain sim
    df["MarketCap_1h"]   = close*vol
    df["RealizedCap_1h"] = close*(vol.cumsum()/max(1,len(df)))
    rolstd = close.rolling(365).std()
    df["MVRV_Z_1h"] = (df["MarketCap_1h"]-df["RealizedCap_1h"])/(rolstd+1e-9)
    df["NVT_1h"]    = df["MarketCap_1h"]/(vol+1e-9)

    # future_return + Label
    df["future_ret_1h"] = (close.shift(-1)-close)/(close+1e-9)
    df["Label_1h"]      = (df["future_ret_1h"]>0).astype(int)
    df['DI_plus_1h'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_1h'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_1h'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_1h'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_1h'] = df['Close'].ewm(span=20, adjust=False).mean()
    holy_grail_all_timeframes(df,"_1h")

    return df


def calculate_indicators_4h(df: pd.DataFrame) -> pd.DataFrame:
    """
    4 saatlik veri, config => RSI(14), ADX(14), ATR(21) + full set (fibo, pivot, cmo, 
    stochRSI, MVRV, Candle Patterns, Heikin, SuperTrend, T3, Vortex, MACD, Boll vs.).
    """
    cfg = INDICATOR_CONFIG["4h"]
    df = df.copy()

    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "4h",
        14,
        20,
        2,
         10)
    # 1) RSI, ADX, ATR
    df["RSI_4h"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_4h"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_4h"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # 2) MACD
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=cfg["MACD_fast"],
        slowperiod=cfg["MACD_slow"],
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_4h"]     = macd
    df["MACDSig_4h"]  = macd_signal
    df["MACDHist_4h"] = macd_hist

    # 3) Bollinger
    up, mid, lw = talib.BBANDS(
        close, 
        timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"], 
        nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_4h"]  = up
    df["BBMid_4h"] = mid
    df["BBLow_4h"] = lw

    # 4) Stoch
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_4h"] = slowk
    df["StochD_4h"] = slowd

    # 5) Ichimoku
    tenkan_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    tenkan_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_4h"] = (tenkan_high + tenkan_low)/2

    kijun_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    kijun_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_4h"] = (kijun_high + kijun_low)/2

    span_a = (df["Ichi_Tenkan_4h"] + df["Ichi_Kijun_4h"]) / 2
    df["Ichi_SpanA_4h"] = span_a.shift(cfg["ICHIMOKU_kijun"])

    spb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    spb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_4h"] = ((spb_high + spb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # 6) SuperTrend
    st4 = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_4h"] = st4["SuperTrend"]
    df["SuperTD_4h"]    = st4["SuperTrend_Direction"]

    # 7) Vortex
    vi4 = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_4h"] = vi4["VI+"]
    df["VI-_4h"] = vi4["VI-"]

    # 8) T3
    df["T3_4h"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # 9) Heikin-Ashi
    ha4 = heikin_ashi(df)
    df["HA_Open_4h"]  = ha4["HA_Open"]
    df["HA_Close_4h"] = ha4["HA_Close"]
    df["HA_High_4h"]  = ha4["HA_High"]
    df["HA_Low_4h"]   = ha4["HA_Low"]
    df["HA_Trend_4h"] = ha4["HA_Trend"]

    # 10) StochRSI
    rsi_ = talib.RSI(close, 14)
    stoch_rsi = (rsi_ - rsi_.rolling(14).min()) / (rsi_.rolling(14).max() - rsi_.rolling(14).min() + 1e-9)
    df["StochRSI_4h"] = stoch_rsi

    # 11) CMO
    df["CMO_4h"] = talib.CMO(close, timeperiod=14)

    # 12) OBV, CMF
    df["OBV_4h"] = talib.OBV(close, vol)
    cmf_val = ((2*close - high - low)/(high - low + 1e-9)) * vol
    df["CMF_4h"] = cmf_val.rolling(window=20).mean()

    # 13) Momentum, ROC
    df["MOM_4h"] = talib.MOM(close, timeperiod=10)
    df["ROC_4h"] = talib.ROC(close, timeperiod=10)

    # 14) Candle Patterns
    df["Candle_Body_4h"] = (close - open_).abs()
    df["Upper_Wick_4h"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_4h"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_4h"]   = ((df["Lower_Wick_4h"] > 2*df["Candle_Body_4h"]) & (df["Upper_Wick_4h"]< df["Candle_Body_4h"])).astype(int)

    # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_4h"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])

    # 16) On-chain sim (MVRV, NVT)
    df["MarketCap_4h"]   = close*vol
    df["RealizedCap_4h"] = close*(vol.cumsum()/max(1,len(df)))
    rolling_std = close.rolling(365).std()
    df["MVRV_Z_4h"] = (df["MarketCap_4h"] - df["RealizedCap_4h"]) / (rolling_std+1e-9)
    df["NVT_4h"]    = df["MarketCap_4h"]/(vol+1e-9)

    # 17) future_return + Label
    df["future_ret_4h"] = (close.shift(-1) - close)/(close+1e-9)
    df["Label_4h"]      = (df["future_ret_4h"]>0).astype(int)
    
    df['DI_plus_4h'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_4h'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_4h'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_4h'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_4h'] = df['Close'].ewm(span=20, adjust=False).mean()
    holy_grail_all_timeframes(df,"_4h")

    return df


def calculate_indicators_1d(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 günlük veri, config => RSI(14), ADX(14), ATR(21) + full set 
    (fibo, pivot, cmo, stochRSI, MVRV, Candle Patterns, Heikin, SuperTrend, T3, 
     Vortex, MACD, Boll vs.).
    """
    cfg = INDICATOR_CONFIG["1d"]
    df = df.copy()

    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "1d",
        14,
        20,
        2,
         10)
    # 1) RSI, ADX, ATR
    df["RSI_1d"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_1d"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_1d"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # 2) MACD
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=cfg["MACD_fast"],
        slowperiod=cfg["MACD_slow"],
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_1d"]     = macd
    df["MACDSig_1d"]  = macd_signal
    df["MACDHist_1d"] = macd_hist

    # 3) Bollinger
    up, mid, lw = talib.BBANDS(
        close,
        timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"],
        nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_1d"]  = up
    df["BBMid_1d"] = mid
    df["BBLow_1d"] = lw

    # 4) Stoch
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_1d"] = slowk
    df["StochD_1d"] = slowd

    # 5) Ichimoku
    tenkan_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    tenkan_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_1d"] = (tenkan_high + tenkan_low)/2

    kijun_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    kijun_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_1d"] = (kijun_high + kijun_low)/2

    span_a = (df["Ichi_Tenkan_1d"] + df["Ichi_Kijun_1d"])/2
    df["Ichi_SpanA_1d"] = span_a.shift(cfg["ICHIMOKU_kijun"])

    spb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    spb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_1d"] = ((spb_high + spb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # 6) SuperTrend
    std_ = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_1d"] = std_["SuperTrend"]
    df["SuperTD_1d"]    = std_["SuperTrend_Direction"]

    # 7) Vortex
    vi_ = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_1d"] = vi_["VI+"]
    df["VI-_1d"] = vi_["VI-"]

    # 8) T3
    df["T3_1d"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # 9) Heikin-Ashi
    ha1d = heikin_ashi(df)
    df["HA_Open_1d"]  = ha1d["HA_Open"]
    df["HA_Close_1d"] = ha1d["HA_Close"]
    df["HA_High_1d"]  = ha1d["HA_High"]
    df["HA_Low_1d"]   = ha1d["HA_Low"]
    df["HA_Trend_1d"] = ha1d["HA_Trend"]

    # 10) StochRSI
    rsi_ = talib.RSI(close, 14)
    stoch_rsi = (rsi_ - rsi_.rolling(14).min()) / (rsi_.rolling(14).max() - rsi_.rolling(14).min() +1e-9)
    df["StochRSI_1d"] = stoch_rsi

    # 11) CMO
    df["CMO_1d"] = talib.CMO(close, timeperiod=14)

    # 12) OBV, CMF
    df["OBV_1d"] = talib.OBV(close, vol)
    cmf_val = ((2*close - high - low)/(high - low+1e-9)) * vol
    df["CMF_1d"] = cmf_val.rolling(window=20).mean()

    # 13) Momentum, ROC
    df["MOM_1d"] = talib.MOM(close, timeperiod=10)
    df["ROC_1d"] = talib.ROC(close, timeperiod=10)

    # 14) Candle Patterns
    df["Candle_Body_1d"] = (close - open_).abs()
    df["Upper_Wick_1d"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_1d"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_1d"]   = ((df["Lower_Wick_1d"] > 2*df["Candle_Body_1d"]) & (df["Upper_Wick_1d"]< df["Candle_Body_1d"])).astype(int)

    # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_1d"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # 16) On-chain sim (MVRV, NVT)
    df["MarketCap_1d"]   = close*vol
    df["RealizedCap_1d"] = close*(vol.cumsum()/max(1,len(df)))
    rolling_std = close.rolling(365).std()
    df["MVRV_Z_1d"] = (df["MarketCap_1d"] - df["RealizedCap_1d"]) / (rolling_std+1e-9)
    df["NVT_1d"]    = df["MarketCap_1d"]/(vol+1e-9)

    # 17) future_return + Label
    df["future_ret_1d"] = (close.shift(-1) - close)/(close+1e-9)
    df["Label_1d"]      = (df["future_ret_1d"]>0).astype(int)
    df['DI_plus_1d'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_1d'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_1d'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_1d'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_1d'] = df['Close'].ewm(span=20, adjust=False).mean()
    holy_grail_all_timeframes(df,"_1d")

    return df



def calculate_indicators_1w(df: pd.DataFrame) -> pd.DataFrame:
    """
    1 günlük veri, config => RSI(14), ADX(14), ATR(21) + full set 
    (fibo, pivot, cmo, stochRSI, MVRV, Candle Patterns, Heikin, SuperTrend, T3, 
     Vortex, MACD, Boll vs.).
    """
    cfg = INDICATOR_CONFIG["1w"]
    df = df.copy()

    high = df["High"]
    low  = df["Low"]
    close= df["Close"]
    vol  = df["Volume"]
    open_= df["Open"]
    add_oi_indicators(
        df,
        "1w",
        14,
        20,
        2,
         10)
    # 1) RSI, ADX, ATR
    df["RSI_1w"] = talib.RSI(close, timeperiod=cfg["RSI"])
    df["ADX_1w"] = talib.ADX(high, low, close, timeperiod=cfg["ADX"])
    df["ATR_1w"] = talib.ATR(high, low, close, timeperiod=cfg["ATR"])

    # 2) MACD
    macd, macd_signal, macd_hist = talib.MACD(
        close,
        fastperiod=cfg["MACD_fast"],
        slowperiod=cfg["MACD_slow"],
        signalperiod=cfg["MACD_signal"]
    )
    df["MACD_1w"]     = macd
    df["MACDSig_1w"]  = macd_signal
    df["MACDHist_1w"] = macd_hist

    # 3) Bollinger
    up, mid, lw = talib.BBANDS(
        close,
        timeperiod=cfg["BOLL_period"],
        nbdevup=cfg["BOLL_stddev"],
        nbdevdn=cfg["BOLL_stddev"]
    )
    df["BBUp_1w"]  = up
    df["BBMid_1w"] = mid
    df["BBLow_1w"] = lw

    # 4) Stoch
    slowk, slowd = talib.STOCH(
        high, low, close,
        fastk_period=cfg["STOCH_fastk"],
        slowk_period=cfg["STOCH_slowk"],
        slowd_period=cfg["STOCH_slowd"]
    )
    df["StochK_1w"] = slowk
    df["StochD_1w"] = slowd

    # 5) Ichimoku
    tenkan_high = high.rolling(window=cfg["ICHIMOKU_tenkan"]).max()
    tenkan_low  = low.rolling(window=cfg["ICHIMOKU_tenkan"]).min()
    df["Ichi_Tenkan_1w"] = (tenkan_high + tenkan_low)/2

    kijun_high = high.rolling(window=cfg["ICHIMOKU_kijun"]).max()
    kijun_low  = low.rolling(window=cfg["ICHIMOKU_kijun"]).min()
    df["Ichi_Kijun_1w"] = (kijun_high + kijun_low)/2

    span_a = (df["Ichi_Tenkan_1w"] + df["Ichi_Kijun_1w"])/2
    df["Ichi_SpanA_1w"] = span_a.shift(cfg["ICHIMOKU_kijun"])

    spb_high = high.rolling(window=cfg["ICHIMOKU_spanB"]).max()
    spb_low  = low.rolling(window=cfg["ICHIMOKU_spanB"]).min()
    df["Ichi_SpanB_1w"] = ((spb_high + spb_low)/2).shift(cfg["ICHIMOKU_kijun"])

    # 6) SuperTrend
    std_ = supertrend(df, period=cfg["SUPER_period"], multiplier=cfg["SUPER_multiplier"])
    df["SuperTrend_1w"] = std_["SuperTrend"]
    df["SuperTD_1w"]    = std_["SuperTrend_Direction"]

    # 7) Vortex
    vi_ = vortex_indicator(df, period=cfg["VORTEX_period"])
    df["VI+_1w"] = vi_["VI+"]
    df["VI-_1w"] = vi_["VI-"]

    # 8) T3
    df["T3_1w"] = T3_moving_average(close, period=cfg["T3_period"], v_factor=cfg["T3_vfactor"])

    # 9) Heikin-Ashi
    ha1w = heikin_ashi(df)
    df["HA_Open_1w"]  = ha1w["HA_Open"]
    df["HA_Close_1w"] = ha1w["HA_Close"]
    df["HA_High_1w"]  = ha1w["HA_High"]
    df["HA_Low_1w"]   = ha1w["HA_Low"]
    df["HA_Trend_1w"] = ha1w["HA_Trend"]

    # 10) StochRSI
    rsi_ = talib.RSI(close, 14)
    stoch_rsi = (rsi_ - rsi_.rolling(14).min()) / (rsi_.rolling(14).max() - rsi_.rolling(14).min() +1e-9)
    df["StochRSI_1w"] = stoch_rsi

    # 11) CMO
    df["CMO_1w"] = talib.CMO(close, timeperiod=14)

    # 12) OBV, CMF
    df["OBV_1w"] = talib.OBV(close, vol)
    cmf_val = ((2*close - high - low)/(high - low+1e-9)) * vol
    df["CMF_1w"] = cmf_val.rolling(window=20).mean()

    # 13) Momentum, ROC
    df["MOM_1w"] = talib.MOM(close, timeperiod=10)
    df["ROC_1w"] = talib.ROC(close, timeperiod=10)

    # 14) Candle Patterns
    df["Candle_Body_1w"] = (close - open_).abs()
    df["Upper_Wick_1w"]  = df[["Close","Open"]].max(axis=1).combine(df["High"], lambda x,y: y - x)
    df["Lower_Wick_1w"]  = df[["Close","Open"]].min(axis=1) - df["Low"]
    df["Is_Hammer_1w"]   = ((df["Lower_Wick_1w"] > 2*df["Candle_Body_1w"]) & (df["Upper_Wick_1w"]< df["Candle_Body_1w"])).astype(int)

    # 15) Pivot & Fibo
    if len(df)>0:
        last_bar = df.iloc[-1]
        pivot_ = (last_bar["High"] + last_bar["Low"] + last_bar["Close"])/3
        df["Pivot_1w"] = pivot_
    hi_ = df["High"].max()
    lo_ = df["Low"].min()
    di_ = hi_ - lo_
    recent_high = df["High"].iloc[-100:].max()  # Son 100 bar içindeki en yüksek seviye
    recent_low  = df["Low"].iloc[-100:].min()   # Son 100 bar içindeki en düşük seviye
    di_ = recent_high - recent_low
    #df["Fibo_61.8"] = recent_high - di_ * 0.618
    df["Fibo_23.6"] = recent_high - di_*0.236
    df["Fibo_38.2"] = recent_high - di_*0.382
    df["Fibo_61.8"] = recent_high - di_*0.618
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band']  = df['Middle_Band'] + 2 * df['Close'].rolling(window=20).std()
    df['Lower_Band']  = df['Middle_Band'] - 2 * df['Close'].rolling(window=20).std()
    df['R1'] = 2 * pivot_ - low
    df['S1'] = 2 * pivot_ - high
    df['R2'] = pivot_ + (high - low)
    df['S2'] = pivot_ - (high - low)
   # Destek ve direnç seviyelerini birleştir
    df['Support'] = df[['S1', 'Lower_Band', 'Fibo_61.8']].min(axis=1)
    df['Resistance'] = df[['R1', 'Upper_Band','Fibo_23.6']].max(axis=1)
    #print("---------", df['R1'] , df['Upper_Band'], df['Fibo_61.8'])
    # 16) On-chain sim (MVRV, NVT)
    df["MarketCap_1w"]   = close*vol
    df["RealizedCap_1w"] = close*(vol.cumsum()/max(1,len(df)))
    rolling_std = close.rolling(365).std()
    df["MVRV_Z_1w"] = (df["MarketCap_1w"] - df["RealizedCap_1w"]) / (rolling_std+1e-9)
    df["NVT_1w"]    = df["MarketCap_1w"]/(vol+1e-9)

    # 17) future_return + Label
    df["future_ret_1w"] = (close.shift(-1) - close)/(close+1e-9)
    df["Label_1w"]      = (df["future_ret_1w"]>0).astype(int)
    df['DI_plus_1w'] = talib.PLUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['DI_minus_1w'] = talib.MINUS_DI(df['High'], df['Low'], df['Close'], timeperiod=14)
    # Momentum
    df['SMA_20_1w'] = df['Close'].rolling(window=20).mean()
    df['SMA_50_1w'] = df['Close'].rolling(window=50).mean()
    df['EMA_20_1w'] = df['Close'].ewm(span=20, adjust=False).mean()
    holy_grail_all_timeframes(df,"_1w")

    return df

###############################################################
# tf_indicators.py - V6 (Gerçekçi Fractal + ATR Renko + MTF)
###############################################################

########################################
# 1) ATR Tabanlı Renko Fonksiyonu
########################################

def compute_atr_renko_signals(df: pd.DataFrame, tf:str="1m",
                              atr_period=14, 
                              brick_size_multiplier=1.0) -> pd.DataFrame:
    """
    ATR tabanlı Renko sinyali (gerçekçi yaklaşıma yakın):
      1) ATR(atr_period) hesapla.
      2) Brick size = ATR * brick_size_multiplier
      3) Her bar kapanışında, son renkoClose'a göre 
         + (birden çok) tuğla veya - (birden çok) tuğla oluşabilir.
      4) En son tuğla up => RenkoTrend=+1, down => -1; 
         henüz ATR yok => 0.
    
    - df: en az 'High','Low','Close' kolonlarını içermeli.
    - Dönüş: df["RenkoClose"], df["RenkoTrend"]
      ("RenkoTrend" => +1/-1/0) 
    """
    # 1) ATR => brick size
    high = df[f"High_{tf}"]
    low  = df[f"Low_{tf}"]
    close= df[f"Close_{tf}"]

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    atr_ = tr.rolling(atr_period).mean()  # basit ATR, dilerseniz talib.ATR

    df[f"RenkoClose_{tf}"] = np.nan
    df[f"RenkoTrend_{tf}"] = 0  # +1/-1/0

    current_renko_close = close.iloc[0]
    current_dir = 0
    df.loc[df.index[0],f"RenkoClose_{tf}"] = current_renko_close
    df.loc[df.index[0],f"RenkoTrend_{tf}"] = 0

    # 2) her bar
    for i in range(1, len(df)):
        c = close.iloc[i]
        bsize = atr_.iloc[i] * brick_size_multiplier if not np.isnan(atr_.iloc[i]) else None
        if bsize is None or bsize<=0:
            # ATR yok => 0
            df.loc[df.index[i],f"RenkoClose_{tf}"] = current_renko_close
            df.loc[df.index[i],f"RenkoTrend_{tf}"] = 0
            continue

        up_diff   = c - current_renko_close
        down_diff = current_renko_close - c
        direction = current_dir

        # up bricks
        while up_diff >= bsize:
            current_renko_close += bsize
            up_diff -= bsize
            direction = +1
        
        # down bricks
        while down_diff >= bsize:
            current_renko_close -= bsize
            down_diff -= bsize
            direction = -1

        df.loc[df.index[i],f"RenkoClose_{tf}"] = current_renko_close
        df.loc[df.index[i],f"RenkoTrend_{tf}"] = direction
        current_dir = direction
        print("Renko Sinyals",current_renko_close,direction)
    return df

########################################
# 2) Bill Williams Fractal (5 Bar Kuralı)
########################################

TF_WEIGHTS = {
    "1m": 1.0,
    "5m": 1.0,
    "15m": 1.5,
    "1h": 2.0,
    "4h": 2.5,
    "1d": 3.0,
      "1w": 3.0
}
TF_BAR_COUNTS = {
    "1m": 2000,
    "5m": 1000,
    "15m": 500,
    "1h": 300,
    "4h": 200,
    "1d": 100,
      "1w": 100
}

def compute_billwilliams_fractals(df: pd.DataFrame, tf:str="1m", confirm_bars: int = 1) -> pd.DataFrame:
    """
    5-bar fractal tespiti:
      - Up fractal: High[i-2] > High[i-4..i-3..i-1..i]
      - Down fractal: Low[i-2] < Low[i-4..i-3..i-1..i]
    confirm_bars: fractal görüldükten sonra en az kaç bar beklenip sinyalin 'onaylanacağı'.

    df["Fractal_Up"], df["Fractal_Down"] = (0/1)
    df["Fractal_Up_Confirmed"], df["Fractal_Down_Confirmed"] = (0/1)
    """
    n = len(df)
    fractal_up = [0]*n
    fractal_down = [0]*n

    # 1) Klasik fractal
    for i in range(4, n):
        mid = i-2
        #print(df["High"].iloc[mid])
        if (df[f"High_{tf}"].iloc[mid] > df[f"High_{tf}"].iloc[mid-1] and
            df[f"High_{tf}"].iloc[mid] > df[f"High_{tf}"].iloc[mid-2] and
            df[f"High_{tf}"].iloc[mid] > df[f"High_{tf}"].iloc[mid+1] and
            df[f"High_{tf}"].iloc[mid] > df[f"High_{tf}"].iloc[mid+2]):
            fractal_up[mid] = 1

        if (df[f"Low_{tf}"].iloc[mid] < df[f"Low_{tf}"].iloc[mid-1] and
            df[f"Low_{tf}"].iloc[mid] < df[f"Low_{tf}"].iloc[mid-2] and
            df[f"Low_{tf}"].iloc[mid] < df[f"Low_{tf}"].iloc[mid+1] and
            df[f"Low_{tf}"].iloc[mid] < df[f"Low_{tf}"].iloc[mid+2]):
            fractal_down[mid] = 1

    df[f"Fractal_Up_{tf}"] = fractal_up
    df[f"Fractal_Down_{tf}"] = fractal_down

    # 2) Confirmation (confirm_bars sonra onay ver)
    #    fractal_up  -> fractal_up_confirmed
    #    fractal_down-> fractal_down_confirmed
    conf_up = [0]*n
    conf_down = [0]*n
    for i in range(n):
        if fractal_up[i] == 1:
            # i barında fractal up -> i+confirm_bars barında onay
            if i+confirm_bars < n:
                conf_up[i+confirm_bars] = 1

        if fractal_down[i] == 1:
            if i+confirm_bars < n:
                conf_down[i+confirm_bars] = 1

    df[f"Fractal_Up_Confirmed_{tf}"] = conf_up
    df[f"Fractal_Down_Confirmed_{tf}"] = conf_up
    print("Confirm_Fractals_{tf}",conf_up ,conf_up)
    return df

def find_pivots(df: pd.DataFrame, col_name: str, left_right=2) -> pd.DataFrame:
    dup_cols = df.columns[df.columns.duplicated()]

    df = df.reset_index(drop=True)
    pivot_low_col = f"PivotLow_{col_name}"
    pivot_high_col = f"PivotHigh_{col_name}"
    
    df[pivot_low_col] = 0
    df[pivot_high_col] = 0
    
    # Pivot hesaplaması
    for i in range(left_right, len(df) - left_right):
        # 'iloc' => i. satır, col_name sütunu (pozisyon bazlı)
        val = df.iloc[i][col_name]  
        
        left_vals = [df.iloc[i - j][col_name] for j in range(1, left_right+1)]
        right_vals = [df.iloc[i + j][col_name] for j in range(1, left_right+1)]
        
        if all(val < x for x in left_vals) and all(val <= x for x in right_vals):
            # df.iloc[i, ...] => i. satırın pivot_low_col sütununu güncelle
            df.iloc[i, df.columns.get_loc(pivot_low_col)] = 1
        
        if all(val > x for x in left_vals) and all(val >= x for x in right_vals):
            df.iloc[i, df.columns.get_loc(pivot_high_col)] = 1
    
    return df

def store_rsi_at_pivots(df: pd.DataFrame, tf: str, left_right=2) -> pd.DataFrame:
    """
    'PivotLow_Close_{tf}' ve 'PivotHigh_Close_{tf}' sütunları = 1 olan satırlarda,
    o anki 'RSI_{tf}' değerini kaydeder. Örneğin 'PivotLow_RSI_{tf}' gibi.
    Böylece pivot noktalarındaki RSI'ları görmek kolaylaşır.
    
    Ayrıca isterseniz pivot index'leri saklayarak 'bir önceki pivot' ile 
    karşılaştırma yapmayı kolaylaştırabilirsiniz.
    """
    pivot_low_col  = f"PivotLow_Close_{tf}"
    pivot_high_col = f"PivotHigh_Close_{tf}"
    rsi_col        = f"RSI_{tf}"
    
    # Aşağıdaki sütunlarda pivot RSI değerleri saklanacak
    df[f"PivotLow_RSI_{tf}"]  = None
    df[f"PivotHigh_RSI_{tf}"] = None
    
    for i in range(left_right, len(df) - left_right):
        if df.loc[i, pivot_low_col] == 1:
            df.loc[i, f"PivotLow_RSI_{tf}"] = df.loc[i, rsi_col]
        if df.loc[i, pivot_high_col] == 1:
            df.loc[i, f"PivotHigh_RSI_{tf}"] = df.loc[i, rsi_col]
    
    return df
def calculate_divergence_advanced(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Belirli bir tf (örn. '1h') için:
     - PivotLow_Close_{tf}, PivotHigh_Close_{tf} sütunlarını okuyup
     - RSI_{tf}, PivotLow_RSI_{tf}, PivotHigh_RSI_{tf} değerlerini kullanarak
       Bullish/Bearish Divergence tespit eder.
    Sonuç: Divergence_Bull_{tf} ve Divergence_Bear_{tf} sütunları eklenir.
    
    Not: Önce 'find_pivots' ve 'store_rsi_at_pivots' çağırarak pivot sütunlarını oluşturmalısınız!
    """
    bull_div_col = f"Divergence_Bull_{tf}"
    bear_div_col = f"Divergence_Bear_{tf}"
    
    df[ bull_div_col ] = 0
    df[ bear_div_col ] = 0
    
    # Pivot sütun isimleri
    pivot_low_col  = f"PivotLow_Close_{tf}"
    pivot_high_col = f"PivotHigh_Close_{tf}"
    
    # RSI pivot sütunları
    pivot_low_rsi_col  = f"PivotLow_RSI_{tf}"
    pivot_high_rsi_col = f"PivotHigh_RSI_{tf}"
    
    close_col = f"Close_{tf}"
    
    # Pivot Low'lar listesi (index'leri alalım)
    pivot_lows_idx  = df.index[df[pivot_low_col] == 1].tolist()
    pivot_highs_idx = df.index[df[pivot_high_col] == 1].tolist()
    
    # -- Bullish Divergence (Dip -> Dip) --
    for i in range(1, len(pivot_lows_idx)):
        prev_i = pivot_lows_idx[i-1]  # önceki pivot low index
        curr_i = pivot_lows_idx[i]    # şimdiki pivot low index
        
        prev_close = df.loc[prev_i, close_col]
        curr_close = df.loc[curr_i, close_col]
        
        prev_rsi = df.loc[prev_i, pivot_low_rsi_col]
        curr_rsi = df.loc[curr_i, pivot_low_rsi_col]
        
        # Şart: fiyat düşmüş (curr_close < prev_close), RSI yükselmiş (curr_rsi > prev_rsi)
        if (curr_close < prev_close) and (curr_rsi > prev_rsi):
            df.loc[curr_i, bull_div_col] = 1
    
    # -- Bearish Divergence (Tepe -> Tepe) --
    for i in range(1, len(pivot_highs_idx)):
        prev_i = pivot_highs_idx[i-1]
        curr_i = pivot_highs_idx[i]
        
        prev_close = df.loc[prev_i, close_col]
        curr_close = df.loc[curr_i, close_col]
        
        prev_rsi = df.loc[prev_i, pivot_high_rsi_col]
        curr_rsi = df.loc[curr_i, pivot_high_rsi_col]
        
        # Şart: fiyat yükselmiş (curr_close > prev_close), RSI düşmüş (curr_rsi < prev_rsi)
        if (curr_close > prev_close) and (curr_rsi < prev_rsi):
            df.loc[curr_i, bear_div_col] = 1

    return df
def calculate_divergence_for_all_timeframes(df: pd.DataFrame, 
                                            tf_list=None, 
                                            left_right=2,
                                            tf_bar_counts=None) -> pd.DataFrame:
    """
    Her TF için, sadece "tf_bar_counts[tf]" kadar son bar üzerinde pivot/fark vb. analiz yapar.
    Sonra bu analiz sonucu ana df'ye işlenir.
    """
    if tf_list is None:
        tf_list = ["1m","5m","15m","1h","4h","1d"]
    if tf_bar_counts is None:
        tf_bar_counts = {
            "1m": 2000,
            "5m": 1000,
            "15m": 500,
            "1h": 300,
            "4h": 200,
            "1d": 100
        }
    
    for tf in tf_list:
        # Bu TF ilgili 'Close_{tf}', 'RSI_{tf}' vs. sütunları var mı?
        close_col = f"Close_{tf}"
        rsi_col   = f"RSI_{tf}"
        if close_col not in df.columns or rsi_col not in df.columns:
            continue
        
        # Son X bar'lık bir alt-DF alalım:
        needed_bars = tf_bar_counts.get(tf, 500)  # default 500
        # slice => df.iloc[-needed_bars:] (ama eğer df çok kısa ise?)
        if len(df) > needed_bars:
            mini = df.iloc[-needed_bars:].copy()
        else:
            mini = df.copy()

        # 1) Pivot bul
        mini = find_pivots(mini, col_name=f"Close_{tf}", left_right=left_right)
        
        # 2) Pivot RSI kaydet
        mini = store_rsi_at_pivots(mini, tf=tf, left_right=left_right)
        
        # 3) Gelişmiş Divergence hesapla
        mini = calculate_divergence_advanced(mini, tf=tf)
        #calculate_divergence_for_all_timeframes
        # Şimdi mini'de "PivotLow_Close_{tf}", "PivotHigh_Close_{tf}", 
        # "Divergence_Bull_{tf}", "Divergence_Bear_{tf}" vb. sütunlar güncel.
        # Bunu ana df'ye geri yazmalıyız. 
        # Overlap indices => son "needed_bars" satırlar.

        # Ekle:
        for col_ in [f"PivotLow_Close_{tf}", f"PivotHigh_Close_{tf}",
                     f"PivotLow_RSI_{tf}", f"PivotHigh_RSI_{tf}",
                     f"Divergence_Bull_{tf}", f"Divergence_Bear_{tf}"]:
            if col_ in mini.columns:
                #print(col_)
                df.loc[mini.index, col_] = mini[col_]
          
    return df


########################################
# 3) Her TF İçin (Gerçek) Renko & Fractal
########################################

def add_real_synergy_indicators_for_all_tfs(
    df: pd.DataFrame,
    tf_list = ["1m","5m","15m","30m","1h","4h","1d"],
    atr_period=14,
    brick_size_multiplier=1.0
) -> pd.DataFrame:
    """
    Her TF için:
      - ATR renko => "RenkoTrend" => +1/-1/0
      - Bill Williams fractal => "Fractal_Up", "Fractal_Down" => +1
    Sonra "Power_Fractal_{tf} = fractal_up - fractal_down"
          "Power_Renko_{tf}   = renkoTrend"
    
    Varsayım: df["Close_{tf}"], df["High_{tf}"], df["Low_{tf}"] mevcuttur.
    """

    for tf in tf_list:
        ccol = f"Close_{tf}"
        hcol = f"High_{tf}"
        lcol = f"Low_{tf}"

        if ccol not in df.columns or hcol not in df.columns or lcol not in df.columns:
            # Bu TF yok => geç
            continue
        
        # mini DataFrame
        mini = pd.DataFrame({
            "Close": df[ccol],
            "High":  df[hcol],
            "Low":   df[lcol]
        }, index=df.index)

        # 1) ATR Renko
        mini = compute_atr_renko_signals(mini, tf,
                                         atr_period=atr_period, 
                                         brick_size_multiplier=brick_size_multiplier)

        # 2) Bill Williams Fractal
        mini = compute_billwilliams_fractals(df,mini,tf)

        # 3) "Power_Fractal_{tf}" => fractal_up - fractal_down
        mini[f"Power_Fractal_{tf}"] = mini[f"Fractal_Up_{tf}"] - mini[f"Fractal_Down_{tf}"]

        # 4) "Power_Renko_{tf}" => RenkoTrend
        mini[f"Power_Renko_{tf}"] = mini[f"RenkoTrend_{tf}"]

        # 5) Orijinal df'ye ekliyoruz
        df[f"Power_Fractal_{tf}"] = mini[f"Power_Fractal_{tf}"]
        df[f"Power_Renko_{tf}"]   = mini[f"Power_Renko_{tf}"]

    return df

########################################
# 4) synergy (örnek) => evaluate_all_indicators_v6
########################################
def evaluate_all_indicators_v6(row: pd.Series) -> float:
    synergy = 0.0
    tf_list = ["1m","5m","15m","1h","4h","1d"]
     
    for tf in tf_list:
        w = TF_WEIGHTS.get(tf, 1.0)
        # OI_BBUp => eğer sumOpenInterestValue barın OI_BBUp üstündeyse synergy +=1
        oi_val = row.get(f"sumOpenInterestValue_{tf}", None)
        oi_bb_up = row.get(f"OI_BBUp_{tf}", None)
        oi_bb_low=row.get(f"OI_BBLow_{tf}", None)
        if oi_val and oi_bb_up and oi_val > oi_bb_up:
            synergy += 1
        elif oi_val and oi_bb_up and oi_val < oi_bb_low:
            synergy -= 1
        
        # OI_ROC => eğer > 20 => synergy +=1, < -20 => synergy -=1
        oi_roc_ = row.get(f"OI_ROC_{tf}", 0)
        #print(f"degerler!{tf}", oi_roc_,oi_val,oi_bb_up,oi_bb_low)
        if oi_roc_ > 20:
            synergy +=1
        elif oi_roc_ < -20:
            synergy -=1
            # Renko
        renko_key = f"Power_Renko_{tf}"
        pr = row.get(renko_key, 0)
        #print("renko",pr)
        synergy += pr * w  # pr=+1 => synergy += +1*w

        # Fractal Confirmed
        up_conf_key = f"Fractal_Up_Confirmed_{tf}"
        dn_conf_key = f"Fractal_Down_Confirmed_{tf}"
        #print("up_conf_key",up_conf_key)

        upc = row.get(up_conf_key, 0)
        dnc = row.get(dn_conf_key, 0)
        
        # eğer up_conf=1 ise synergy += 2*w
        if upc == 1:
            synergy += (2 * w)
        if dnc == 1:
            synergy -= (2 * w)
        # Divergence
        div_bull = row.get(f"Divergence_Bull_{tf}", 0)
        div_bear = row.get(f"Divergence_Bear_{tf}", 0)
        #print("div_bull",div_bull,div_bear)

        synergy += div_bull * 2
        synergy -= div_bear * 2
        #print("sinerji",synergy)
   
    # Örnek: Divergence eklerseniz
   
    
    return synergy

########################################
# 5) Diğer Yardımcı (TF Score, Onchain, Macro, vs.)
########################################

def calc_tf_score_v6(row, tf: str) -> float:
    """
    Örnek MTF skor => RSI_{tf}, MACD_{tf}, ADX_{tf}, Candle Engulf vs.
    """
    score = 0.0
    # RSI => +1/-1
    rsi_val = row.get(f"RSI_{tf}", 50)
    if rsi_val>60:
        score +=1
    elif rsi_val<40:
        score -=1

    # MACD => +1/-1
    macd_  = row.get(f"MACD_{tf}", 0)
    macds_ = row.get(f"MACDSig_{tf}", 0)
    if macd_>macds_:
        score+=1
    else:
        score-=1

    # ADX>25 => +1
    adx_ = row.get(f"ADX_{tf}", 0)
    if adx_>25:
        score+=1

    # Candle Engulf => +1/-1
    cdl_ = row.get(f"CDL_ENGULFING_{tf}", 0)
    if cdl_>0:
        score+=1
    elif cdl_<0:
        score-=1
    holy_grail_= row.get(f"HolyGrail_FinalSignal_{tf}", 0)
    
    if holy_grail_>0:
       score+=1
    elif holy_grail_<0:
        score-=1

    return score

def calc_onchain_v6(row: pd.Series) -> float:
    """MVRV, Funding, Order_Book_Num vb."""
    score = 0.0
    mvrv = row.get("MVRV_Z_1d", 0)
    if mvrv:
        if mvrv < -0.5:
            score +=2
        elif mvrv>0.7:
            score -=2
        
    fund = row.get("Funding_Rate", 0)
    
    if fund:
        if fund>0.01:
            score +=1
        elif fund<-0.01:
            score-=1

    ob = row.get("Order_Book_Num", 0)
    if ob:
        if ob>0:
            score+=1
        elif ob<0:
            score-=1

    return score

def calc_macro_v6(row: pd.Series) -> float:
    """
    SP500, DXY, VIX verilerini baz alarak
    basit bir makro skor hesaplar.
    row içerisinde:
      - "SPX_Change" => S&P 500 günlük % değişim (ör. 1.2 => %1.2)
      - "DXY_Change" => DXY günlük % değişim
      - "VIX"        => VIX endeksi seviyesi
    """
    score = 0.0

    # ---- SP500 ----
    spx_chg = row.get("SPX_Change", 0)
    if spx_chg > 1.0:
        score += 1
    elif spx_chg < -1.0:
        score -= 1

    # ---- DXY ----
    dxy_chg = row.get("DXY_Change", 0)
    if dxy_chg > 0.3:
        score -= 1
    elif dxy_chg < -0.3:
        score += 1

    # ---- VIX ----
    vix_val = row.get("VIX", 20)
    if vix_val > 30:
        score -= 2
    elif vix_val > 20:
        score -= 1
    elif vix_val < 15:
        score += 1

    return score


def calc_sentiment_v6(row: pd.Series) -> float:
    """FearGreed + News => +1/-1"""
    sc = 0
    fgi = row.get("Fear_Greed_Index", 0.5)
    if fgi<0.3:
        sc +=1
    elif fgi>0.7:
        sc -=1

    news_ = row.get("News_Headlines", 0.0)
    if news_>0.2:
        sc +=1
    elif news_<-0.2:
        sc -=1

    return sc

########################################
# 6) 4h Rejim (TREND_UP, TREND_DOWN, RANGE vs.)
########################################
def detect_volatility_regime(df: pd.DataFrame, period=14, threshold_low=0.5, threshold_high=1.5) -> str:
    """
    ATR veya Bollinger Band genişliği ile basit bir volatilite rejimi.
    threshold_low, threshold_high => ATR değerini son X barın ortalamasına göre normalize edebilirsiniz.
    Burada basitçe:
      ATR / Close ortalama >= threshold_high => "HIGH_VOL"
      ATR / Close ortalama <= threshold_low  => "LOW_VOL"
      aksi => "MID_VOL"
    """
    if len(df) < period:
        return "MID_VOL"
    high = df["High_4h"]
    low  = df["Low_4h"]
    close= df["Close_4h"]
    
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    atr_val = tr.rolling(period).mean().iloc[-1]
    
    avg_close = close.rolling(period).mean().iloc[-1]
    ratio = atr_val / avg_close if avg_close != 0 else 0.0
    
    if ratio >= threshold_high:
        return "HIGH_VOL"
    elif ratio <= threshold_low:
        return "LOW_VOL"
    else:
        return "MID_VOL"

def detect_regime_4h_v6(row_4h) -> str:
    """
    Basit trend tespiti => ADX_4h, RSI_4h, MACD_4h
    """
    adx_ = row_4h.get("ADX_4h", 0)
    rsi_ = row_4h.get("RSI_4h", 50)
    macd_  = row_4h.get("MACD_4h", 0)
    macds_ = row_4h.get("MACDSig_4h", 0)
    oi_val = row_4h.get("sumOpenInterestValue_4h", 0)
    oi_bb_up = row_4h.get("OI_BBUp_4h", 0)
    oi_bb_mid= row_4h.get("OI_BBMid_4h", 0)
    oi_roc_ = row_4h.get("OI_ROC_4h", 0)
    oi_rsi_ = row_4h.get("OI_RSI_4h", 50)
    #print("-----sdfdsfsdfdf---",oi_val,oi_bb_mid,oi_rsi_)
    if adx_ > 20:
       # Trend var
        # Koşul => RSI>55, MACD>MACDSig 
        # + OI Bollinger 
        if (rsi_ > 55 and macd_ > macds_) and (oi_val > oi_bb_mid)  and oi_roc_ > 0:  # OI ortalama üstünde => trend teyidi
           
            if oi_rsi_ > 70:
            # OI overbought => belki "TREND_UP_OVERHEATED" vs.
                return "TREND_UP_OVERHEATED"
            else:
                 return "TREND_UP"
        elif (rsi_ < 45 and macd_ < macds_) \
             and (oi_val < oi_bb_mid):
            return "TREND_DOWN"
        else:
            return "TREND_FLAT"
    else:
        return "RANGE"

def detect_regime_1h_v6(row: pd.Series) -> str:
    """
    1 Saatlik veride hesaplanmış indikatör sütunlarına bakarak 
    rejim tespiti: TREND_UP / TREND_DOWN / TREND_FLAT / RANGE / BREAKOUT_SOON
    """
    # Değerleri row'dan okuyalım:
    
    adx_   = row.get("ADX_1h", 0.0)
    rsi_   = row.get("RSI_1h", 50.0)
    macd_  = row.get("MACD_1h", 0.0)
    macds_ = row.get("MACDSig_1h", 0.0)
    #bbw_   = row.get("BBWidth_1h", 999.0)
    up=row.get("BBUp_1h", 999.0)
    mid=row.get("BBMid_1h", 999.0)
    low=row.get("BBLow_1h", 999.0)
    bbw_=(up-low)/mid
    
    # Eşik değerler (örnek)
    ADX_THRESHOLD    = 20.0
    RSI_BULL_LEVEL   = 55.0
    RSI_BEAR_LEVEL   = 45.0
    BB_SQUEEZE_LEVEL = 2.0  # bant genişliği 2.0'ın altındaysa sıkışma

    if adx_ > ADX_THRESHOLD:
        # Bir trend var
        if rsi_ > RSI_BULL_LEVEL and macd_ > macds_:
            return "TREND_UP"
        elif rsi_ < RSI_BEAR_LEVEL and macd_ < macds_:
            return "TREND_DOWN"
        else:
            # ADX yüksek, ama RSI ve MACD yön vermiyor => kararsız
            return "TREND_FLAT"
    else:
        # ADX düşük => Trend zayıf => Range veya Sıkışma
        if bbw_ < BB_SQUEEZE_LEVEL:
            return "BREAKOUT_SOON"
        else:
            return "RANGE"

def combine_regimes(vol_regime: str, trend_regime: str) -> str:
    return f"{vol_regime}_{trend_regime}"

def confirm_advanced_v6(df: pd.DataFrame, row: pd.Series,synergy_val) -> (bool, bool):
    """
    Gelişmiş Bull/Bear teyit (confirmation) örneği:
      - 5m zaman diliminde:
         1) 'volume spike' + barın yukarı kapanışı => potansiyel bull
         2) fractal up => potansiyel bull
         (Tersi => bear)
         (Bu iki ipucu “veya” (OR) ile birleştiriliyor; 
          isterseniz “ve” (AND) yapabilirsiniz)
      - synergy > 0 => bull'a ek destek, synergy < 0 => bear'a ek destek
      - 4h T3 slope => son 3 barlık (daha stabil) eğime bakıyoruz.
         slope > 0 => up, slope < 0 => down
      - Sonuç: bull_conf = True (hem kısa vade sinyali hem 4h slope + synergy +)
                bear_conf = True (hem kısa vade sinyali hem 4h slope - synergy -)

    Parametreler:
      - df: tüm veri DataFrame'i (asof-merge edilmiş, “Volume_5m”, “Close_5m”, “Power_Fractal_5m”, “T3_4h”, “synergy” vb. sütunları içermeli)
      - row: son bar'a ait Series (df.iloc[-1]), 
             row.get(...) ile son değerlere ulaşıyoruz.

    Dönüş:
      (confirm_bull, confirm_bear) => (bool, bool)
    """

    confirm_bull = False
    confirm_bear = False

    # ------------ 1) 5m Verisinden Bull/Bear İpucu ------------
    # a) Volume Spike (5m) => son bar hacmi son 20 barın ortalamasının X katı üzerindeyse
    vol_5m_now = row.get("Volume_5m", 0.0)
    price_5m_now = row.get("Close_5m", 0.0)
    open_5m_now  = row.get("Open_5m", 0.0)

    if len(df) >= 20:
        vol_5m_window = df["Volume_5m"].iloc[-20:]
        avg_vol_5m = vol_5m_window.mean()
        # Eşik değeri 1.5 gibi alabilirsiniz, testlerle ayarlayabilirsiniz.
        volume_spike_5m = (vol_5m_now > 1.5 * avg_vol_5m)
    else:
        volume_spike_5m = False

    # 5m bar yukarı kapanmış mı? (Close > Open)
    bar_up_5m = (price_5m_now > open_5m_now)
    bar_down_5m = (price_5m_now < open_5m_now)

    # b) 5m fractal
    pf_5m = row.get("Power_Fractal_5m", 0)  # +1 => fractal up, -1 => fractal down
    fractal_up_5m = (pf_5m > 0)
    fractal_down_5m = (pf_5m < 0)

    # c) Bullish ipucu => volume_spike + bar_up veya fractal_up
    #    Aşağıdaki mantık "OR" ile birleştirilmiş. 
    #    Daha katı koşullar isterseniz ("AND") vs. değiştirebilirsiniz.
    bullish_signal_5m = (
        (volume_spike_5m and bar_up_5m)  # hacim patlaması + yeşil bar
        or fractal_up_5m                # fractal up
    )

    # d) Bearish ipucu => volume_spike + bar_down veya fractal_down
    #    "volume spike" her zaman yukarı demek değil; 
    #    eğer bar aşağı kapandıysa, ayı hareketi olabilir.
    bearish_signal_5m = (
        (volume_spike_5m and bar_down_5m)
        or fractal_down_5m
    )

    # ------------ 2) synergy Desteği (opsiyonel) ------------
    #synergy_val = row.get("synergy", 0.0)
    synergy_bull = (synergy_val > 1)   # Eşik örnek
    synergy_bear = (synergy_val < -1)

    # ------------ 3) 4H T3 Slope (Son 3 Bar) ------------
    # df["T3_4h"] => 4h T3
    # Biz son 3 barın T3 değerine bakarak ortalama slope hesaplayalım.
    t3_slope_4h = 0.0
    t3_values = df["T3_4h"].dropna()
    # Son 3 bar varsa:
    if len(t3_values) >= 3:
        # Örnek: slope = (T3[-1] - T3[-3]) / 2 
        # (bar başına ortalama değişim)
        t3_slope_4h = (t3_values.iloc[-1] - t3_values.iloc[-3]) / 2.0

    bull_t3 = (t3_slope_4h > 0)
    bear_t3 = (t3_slope_4h < 0)

    # ------------ 4) Kısa Vade + Orta Vade Birleşimi ------------
    # Bull => (5m bullish sinyal) ve (T3 slope > 0) ve (synergy_bull opsiyonel)
    # Burada "AND" ile birleştiriyoruz. synergy'yi "isteğe bağlı" da kullanabilirsiniz.
    if bullish_signal_5m and bull_t3:
        # synergy'yi tamamen zorunlu kılmak istemiyorsanız "or synergy_bull" diyebilirsiniz.
        if synergy_bull:
            confirm_bull = True

    # Bear => (5m bearish sinyal) ve (T3 slope < 0) ve synergy_bear
    if bearish_signal_5m and bear_t3:
        if synergy_bear:
            confirm_bear = True

    return (confirm_bull, confirm_bear)

def calculate_atr_stoploss(df: pd.DataFrame, row: pd.Series, atr_period=14, multiplier=1.5) -> float:
    """
    ATR tabanlı basit stop loss:
      stop_distance = ATR(14) * multiplier
      stop_loss_price = row['Close'] - stop_distance (long için)
    """
    if len(df) < atr_period:
        return 0.0
    high = df["High_4h"]
    low  = df["Low_4h"]
    close= df["Close_4h"]
    
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr  = tr1.combine(tr2, np.maximum).combine(tr3, np.maximum)
    atr_series = tr.rolling(atr_period).mean()
    atr_val = atr_series.iloc[-1]  # son bar ATR
    
    stop_distance = atr_val * multiplier
    stop_loss_price = row["Close_4h"] - stop_distance  # long senaryosu
    return stop_loss_price


########################################
# 7) Final Decision
########################################
def final_decision_v6(regime: str, total_score: float, bull_conf: bool, bear_conf: bool) -> (int, str):
    """
    0=SELL, 1=BUY, 2=HOLD => reason
    regime şu değerleri alabilir:
      - TREND_UP, TREND_UP_OVERHEATED, LOW_VOL_TREND_UP, LOW_VOL_TREND_UP_OVERHEATED
      - TREND_DOWN, TREND_DOWN_OVERHEATED, LOW_VOL_TREND_DOWN, ...
      - BREAKOUT_SOON
      - RANGE, vb.

    total_score: hesapladığınız MTF skor + synergy + onchain + macro + sentiment
    bull_conf, bear_conf: confirm_advanced_v6 vb. teyit bool'ları
    """
    act = 2
    reason = ""

    # -------- 1) Yukarı Trend Grupları --------
    if regime in ["TREND_UP", 
                  "TREND_UP_OVERHEATED", 
                  "LOW_VOL_TREND_UP", 
                  "LOW_VOL_TREND_UP_OVERHEATED"]:
        # Örnek kural: total_score > 3 + bull_conf => BUY
        if total_score > 20:
            act = 1
            reason = f"{regime}_BULLCONF"
        else:
            # Aşırı ısınma (overheated) ya da bull_conf yetersiz => HOLD
            act = 2
            reason = f"{regime}_BUT_WEAK"

    # -------- 2) Aşağı Trend Grupları --------
    elif regime in ["TREND_DOWN", 
                    "TREND_DOWN_OVERHEATED", 
                    "LOW_VOL_TREND_DOWN", 
                    "LOW_VOL_TREND_DOWN_OVERHEATED"]:
        if total_score < 0 :
            act = 0
            reason = f"{regime}_BEARCONF"
        else:
            act = 2
            reason = f"{regime}_BUT_WEAK"

    # -------- 3) Breakout Soon --------
    elif regime == "BREAKOUT_SOON":
        if total_score > 10:
            act = 1
            reason = "BREAKOUT_SOON_BULLISH"
        elif total_score < -5:
            act = 0
            reason = "BREAKOUT_SOON_BEARISH"
        else:
            act = 2
            reason = "BREAKOUT_SOON_UNCERTAIN"

    # -------- 4) Diğer Durumlar => RANGE --------
    else:
        # RANGE
        if total_score > 5:
            act = 1
            reason = "RANGE_BULLISH"
        elif total_score < -5:
            act = 0
            reason = "RANGE_BEARISH"
        else:
            act = 2
            reason = "SIDEWAYS"

    return (act, reason)

########################################
# 8) Delayed Signals
########################################

def dynamic_delay_based_on_synergy(val: float) -> int:
    """synergy>5 => 1 bar, >2 =>2 bar, >0 =>3 bar, aksi =>5 bar"""
    if val>5:
        return 1
    elif val>2:
        return 2
    elif val>0:
        return 3
    else:
        return 5

def generate_delayed_signals_v6(final_act: int, synergy: float, row) -> list:
    """
    synergy tabanlı bar gecikmesi => 
    eğer BUY => partial SELL plan
    eğer SELL => contrarian BUY plan
    """
    signals = []
    if final_act==1:  # BUY
        d1 = dynamic_delay_based_on_synergy(synergy)
        signals.append({
            "bars_delay": d1,
            "action": "SELL",
            "qty_ratio": 0.5,
            "condition": {
                "type": "SYNERGY",
                "operator": ">",
                "value": synergy+1
            },
            "reason": f"Partial SELL if synergy>{synergy+1}"
        })
    elif final_act==0: # SELL
        d2 = dynamic_delay_based_on_synergy(synergy)
        signals.append({
            "bars_delay": d2,
            "action": "BUY",
            "qty_ratio": 0.3,
            "condition": {
                "type": "SYNERGY",
                "operator": "<",
                "value": synergy-2
            },
            "reason": f"Contrarian BUY if synergy<{synergy-2}"
        })
    return signals

########################################
# 9) Nihai MTF v6 Sinyal Fonksiyonu
########################################
def analyze_trends_and_signals_v6(df: pd.DataFrame) -> dict:
    """
    MTF + synergy + onchain + macro + sentiment + rejim + ATR Stop => final sinyal (0/1/2)
    0 = SELL, 1 = BUY, 2 = HOLD
    """
    if len(df) < 20:
        return {
            "immediate_action": 2,
            "immediate_reason": "NOT_ENOUGH_DATA",
            "total_score": 0.0,
            "detail_scores": {},
            "regime": "NONE",
            "delayed_signals": [],
            "stop_loss_price": None
        }
    #holy grail  stratejisinden 1 saatlik signalleri al ve stoploss  15 dakika rsi teyyitli
    row = df.iloc[-1]
    tf_list = ["1m","5m","15m","1h","4h","1d"]
    df = calculate_divergence_for_all_timeframes(
    df, 
    tf_list=tf_list,  # sadece 1h için örnek
    left_right=2
        )
    calculate_divergence_for_all_timeframes(df, 
                                            tf_list, 
                                            left_right=2,
                                            tf_bar_counts=None)
    df =holy_grail_all_timeframes(df)

    # 1) TF skorlar (örnek)
    sc_1m = calc_tf_score_v6(row, "1m")
    sc_5m = calc_tf_score_v6(row, "5m")
    sc_15m = calc_tf_score_v6(row, "15m")
    sc_30m = calc_tf_score_v6(row, "30m")
    sc_1h= calc_tf_score_v6(row, "1h")

    sc_4h = calc_tf_score_v6(row, "4h")
    sc_1d = calc_tf_score_v6(row, "1d")
   
    # 2) Gelişmiş synergy (fractal confirmed vs.)
    synergy_val = evaluate_all_indicators_v6(row)
    
    # 3) On-chain, macro, sentiment
    onchain_s = calc_onchain_v6(row)
    macro_s   = calc_macro_v6(row)
    senti_s   = calc_sentiment_v6(row)
    
    # 4) Rejimler
    regime_4h  = detect_regime_4h_v6(row)
    regime_1h  = detect_regime_1h_v6(row)

    vol_regime = detect_volatility_regime(df)
    regime_combined_4h = combine_regimes(vol_regime, regime_4h)
    regime_combined_1h = combine_regimes(vol_regime, regime_1h)

    # 5) Temel MTF skor => (1m+5m) + 2*(4h) + 2*(1d)
    base_mtf = (sc_1m + sc_5m + sc_15m + sc_30m) \
           + 1.5 * sc_1h \
           + 2.0 * sc_4h \
           + 2.5 * sc_1d  
    total_s = base_mtf + synergy_val + onchain_s + macro_s + senti_s
    
    # 6) Teyit
    bull_conf, bear_conf = confirm_advanced_v6(df, row,synergy_val)
    
    # 7) Karar => 0/1/2
    final_act, reason_ = final_decision_v6(regime_4h, total_s, bull_conf, bear_conf)
    
    # 8) Stop loss (örnek)
    sl_price = None
    if final_act == 1:
        sl_price = calculate_atr_stoploss(df, row, atr_period=14, multiplier=1.5)
    elif final_act == 0:
        # Short iseniz SL mantığı terstir
        sl_dist = calculate_atr_stoploss(df, row, atr_period=14, multiplier=1.5)
        # burda "sl_price" mantığı (Close + sl_dist) gibi ayarlanmalı
        if sl_dist is not None:
            sl_price = row["Close"] + (row["Close"] - sl_dist)
    
    # 9) Delayed
    delayed_signals = generate_delayed_signals_v6(final_act, synergy_val, row)
    df.columns.tolist()
    detail_scores = {
        "sc_1m": sc_1m,
        "sc_5m": sc_5m,
         "sc_15m": sc_15m,
          "sc_30m": sc_30m,
           "sc_1h": sc_1h,

        "sc_4h": sc_4h,
        "sc_1d": sc_1d,
        "synergy": synergy_val,
        "onchain": onchain_s,
        "macro": macro_s,
        "sentiment": senti_s
    }
    
    return {
        "immediate_action": final_act,
        "immediate_reason": reason_,
        "total_score": total_s,
        "detail_scores": detail_scores,
        "1h_regime": regime_combined_1h,
        "4h_regime": regime_combined_4h,
        "delayed_signals": delayed_signals,
        "stop_loss_price": sl_price
    }
