import math
import numpy as np
import pandas as pd
import concurrent.futures
import warnings

##############################
# GANN ek import (opsiyonel)
##############################
try:
    import swisseph
except ImportError:
    swisseph = None

##############################
# GENEL HELPER
##############################
def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def line_equation(x1, y1, x2, y2):
    """
    Returns slope (m) and intercept (b) of the line passing through (x1, y1) and (x2, y2).
    If vertical => returns (None, None).
    """
    if (x2 - x1) == 0:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return m, b

def line_intersection(m1, b1, m2, b2):
    """
    Intersection (x, y) of lines y=m1*x+b1 and y=m2*x+b2.
    If parallel or invalid => returns (None, None).
    """
    if (m1 is None) or (m2 is None):
        return None, None
    if m1 == m2:
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y


##############################
# ATR / Volume Hazırlık
##############################
def prepare_atr(df: pd.DataFrame, time_frame: str = "1m", period: int = 14):
    """
    Prepares Average True Range (ATR) in the DataFrame with rolling mean of TR over `period`.
    Columns used: High, Low, Close with the time_frame suffix (e.g. High_1m).
    """
    high_col = get_col_name("High", time_frame)
    low_col  = get_col_name("Low",  time_frame)
    close_col= get_col_name("Close",time_frame)
    atr_col  = get_col_name("ATR",  time_frame)

    if atr_col in df.columns:
        return
    df[f"H-L_{time_frame}"]  = df[high_col] - df[low_col]
    df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
    df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()
    df[f"TR_{time_frame}"]   = df[[f"H-L_{time_frame}",
                                  f"H-PC_{time_frame}",
                                  f"L-PC_{time_frame}"]].max(axis=1)
    df[atr_col] = df[f"TR_{time_frame}"].rolling(period).mean()

def prepare_volume_ma(df: pd.DataFrame, time_frame: str="1m", period: int=20):
    """
    Prepares a rolling moving average of Volume over `period`.
    """
    vol_col = get_col_name("Volume", time_frame)
    ma_col  = f"Volume_MA_{period}_{time_frame}"
    if (vol_col in df.columns) and (ma_col not in df.columns):
        df[ma_col] = df[vol_col].rolling(period).mean()


##############################
# RSI, MACD 
##############################
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Standard RSI calculation over `period`.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    MACD line, Signal line, and Histogram.
    """
    fast_ema = series.ewm(span=fastperiod).mean()
    slow_ema = series.ewm(span=slowperiod).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signalperiod).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def indicator_checks(df: pd.DataFrame,
                     idx: int,
                     time_frame: str="1m",
                     rsi_check: bool = True,
                     macd_check: bool = True,
                     rsi_period=14,
                     macd_fast=12,
                     macd_slow=26,
                     macd_signal=9) -> dict:
    """
    Example combined RSI & MACD check at a specific index.
    Returns a dict with 'verdict' and messages.
    """
    res = {
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "verdict": True,
        "msgs": []
    }
    close_col = get_col_name("Close", time_frame)
    if close_col not in df.columns:
        res["verdict"] = False
        res["msgs"].append("Close column not found, indicator checks skipped.")
        return res

    # RSI
    if rsi_check:
        if f"RSI_{time_frame}" not in df.columns:
            df[f"RSI_{time_frame}"] = compute_rsi(df[close_col])
        if idx < len(df):
            rsi_val = df[f"RSI_{time_frame}"].iloc[idx]
            res["rsi"] = rsi_val
            if (not pd.isna(rsi_val)) and (rsi_val < 50):
                res["verdict"]= False
                res["msgs"].append(f"RSI {rsi_val:.2f} <50 => negative.")
        else:
            res["verdict"]= False
            res["msgs"].append("RSI idx out of range")

    # MACD
    if macd_check:
        macd_col   = f"MACD_{time_frame}"
        macds_col  = f"MACD_signal_{time_frame}"
        if macd_col not in df.columns or macds_col not in df.columns:
            macd_line, macd_signal, _ = compute_macd(df[close_col], macd_fast, macd_slow, macd_signal)
            df[macd_col]  = macd_line
            df[macds_col] = macd_signal
        if idx < len(df):
            macd_val = df[macd_col].iloc[idx]
            macd_sig = df[macds_col].iloc[idx]
            res["macd"] = macd_val
            res["macd_signal"] = macd_sig
            if (macd_val < macd_sig):
                res["verdict"]=False
                res["msgs"].append(f"MACD < Signal at index={idx}")
        else:
            res["verdict"]=False
            res["msgs"].append("MACD idx out of range")

    return res


##############################
# PivotScanner (volume/ATR)
##############################
class PivotScanner:
    """
    A utility class to scan for local maxima/minima (pivots) 
    with optional volume and ATR filters.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 time_frame: str = "1m",
                 left_bars: int = 5,
                 right_bars: int= 5,
                 volume_factor: float = 1.2,
                 atr_factor: float= 0.0,
                 volume_ma_period: int=20,
                 atr_period: int=14,
                 min_distance_bars: int=0):
        self.df = df
        self.time_frame = time_frame
        self.left_bars= left_bars
        self.right_bars= right_bars
        self.volume_factor= volume_factor
        self.atr_factor= atr_factor
        self.volume_ma_period= volume_ma_period
        self.atr_period= atr_period
        self.min_distance_bars= min_distance_bars
        #print("degerler--->",time_frame, self.atr_factor,self.volume_factor,self.left_bars,self.right_bars
            #  )
        # Prepare volume/ATR if needed
        prepare_volume_ma(df, time_frame, volume_ma_period)
        if atr_factor>0:
            prepare_atr(df, time_frame, atr_period)

    def find_pivots(self):
        #print(self.left_bars,self.right_bars)
        """
        Finds local maxima and minima:
          +1 => local max
          -1 => local min
        Then filters them by volume_factor, atr_factor, etc. 
        Returns a list of tuples: (index, price, +1/-1)
        """
        close_col= get_col_name("Close", self.time_frame)
        if close_col not in self.df.columns:
            return []
        price= self.df[close_col]
        n= len(price)
        pivots=[]
        for i in range(self.left_bars, n- self.right_bars):
            val= price.iloc[i]
            left_slice = price.iloc[i- self.left_bars : i]
            right_slice= price.iloc[i+1: i+1+ self.right_bars]
            is_local_max = (all(val> l for l in left_slice) and all(val>= r for r in right_slice))
            is_local_min = (all(val< l for l in left_slice) and all(val<= r for r in right_slice))
            if is_local_max:
                if self._pivot_ok(i,val,+1):
                    pivots.append((i,val,+1))
            elif is_local_min:
                if self._pivot_ok(i,val,-1):
                    pivots.append((i,val,-1))

        # Enforce minimum spacing if needed
        if self.min_distance_bars>0 and len(pivots)>1:
            filtered=[pivots[0]]
            for j in range(1,len(pivots)):
                if pivots[j][0]- filtered[-1][0]>= self.min_distance_bars:
                    filtered.append(pivots[j])
            pivots= filtered
        #print("donen degerler", pivots)
        return pivots

    def _pivot_ok(self, idx,val,ptype):
        """
        Checks volume factor and ATR factor constraints for the pivot at df.index=idx.
        """
        volume_col= get_col_name("Volume", self.time_frame)
        vol_ma_col= f"Volume_MA_{self.volume_ma_period}_{self.time_frame}"
        atr_col   = get_col_name("ATR", self.time_frame)

        # Volume check
        if self.volume_factor>0 and (volume_col in self.df.columns) and (vol_ma_col in self.df.columns):
            vol_now= self.df[volume_col].iloc[idx]
            vol_ma= self.df[vol_ma_col].iloc[idx]
            if (not pd.isna(vol_now)) and (not pd.isna(vol_ma)):
                if vol_now< (self.volume_factor* vol_ma):
                    return False

        # ATR check
        if self.atr_factor>0 and (atr_col in self.df.columns):
            pivot_atr= self.df[atr_col].iloc[idx]
            if not pd.isna(pivot_atr):
                prev_close= self.df[get_col_name("Close", self.time_frame)].iloc[idx-1] if idx>0 else val
                diff= abs(val- prev_close)
                if diff< (self.atr_factor* pivot_atr):
                    return False

        return True


##############################
# Parametre optimizasyon example
##############################
import json
import os
from itertools import product

def save_best_params_to_json(symbol: str,
                             timeframe: str,
                             best_params: dict,
                             best_score: float,
                             filename: str = None):
    """
    Bir symbol.json dosyasında, time_frames[timeframe] kısmına
    best_params ve best_score değerlerini kaydeder.

    Örneğin filename=None ise "btcusdt.json" gibi bir varsayılan isim oluşturabiliriz.
    """
    if filename is None:
        # Örn. "BTCUSDT" => "btcusdt.json"
        filename = f"{symbol.lower()}.json"

    # Eğer dosya yoksa temel bir dict yapısı oluştur
    if not os.path.exists(filename):
        data = {
            "symbol": symbol,
            "time_frames": {}
        }
    else:
        # Dosya varsa yükle
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

    # time_frames altında bu timeframe'e özel bir dict oluşturalım
    if "time_frames" not in data:
        data["time_frames"] = {}

    data["time_frames"][timeframe] = {
        "best_params": best_params,
        "best_score": best_score
    }

    # Yazma
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"[save_best_params_to_json] => {filename} updated with timeframe={timeframe}")


def load_best_params_from_json(symbol: str, timeframe: str, filename: str = None) -> dict:
    """
    Kaydedilmiş symbol.json dosyasından time_frames[timeframe].best_params'ı yükleyip döndürür.
    """
    if filename is None:
        filename = f"{symbol.lower()}.json"

    if not os.path.exists(filename):
        print(f"[load_best_params_from_json] => File {filename} not found!")
        return None

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "time_frames" not in data:
        print("[load_best_params_from_json] => 'time_frames' key not found in JSON!")
        return None

    tf_info = data["time_frames"].get(timeframe, None)
    if tf_info is None:
        print(f"[load_best_params_from_json] => timeframe '{timeframe}' not found!")
        return None

    return tf_info.get("best_params", None)


def optimize_parameters(
    df,
    symbol: str = "BTCUSDT",
    time_frame: str = "1m",
    param_grid: dict = None
) -> dict:
    """
    Gelişmiş optimizasyon:
      - symbol + time_frame => hangi dataseti optimize ettiğimizi belirtiyor
      - param_grid => pivot params
      - Strateji => calc_strategy_pnl(...) 
      - 'score' => basit formül => PnL - alpha*drawdown
      - Sonuçta best_params + best_score => symbol.json'a kaydedilir (time_frames[time_frame]).
    """
    if param_grid is None:
        param_grid= {
            "left_bars": [5,10],
            "right_bars":[5,10],
            "volume_factor":[1.0,1.2],
            "atr_factor":[0.0,0.2]
        }

    best_score = -999999
    best_params = None

    all_results = []

    for vals in product(*[param_grid[k] for k in param_grid.keys()]):
        params = dict(zip(param_grid.keys(), vals))

        # 1) PivotScanner => pivots
        scanner = PivotScanner(
            df, 
            time_frame,
            left_bars    = params["left_bars"],
            right_bars   = params["right_bars"],
            volume_factor= params["volume_factor"],
            atr_factor   = params["atr_factor"]
        )
        pivs = scanner.find_pivots()

        # 2) Strateji metriği
        metrics = calc_strategy_pnl(
            df, 
            pivs, 
            time_frame=time_frame,
            commission_rate=0.0005,
            slippage=0.0,
            use_stop_loss=True,
            stop_loss_atr_factor=2.0,
            allow_short=True
        )
        # { "pnl":..., "max_drawdown":..., "trade_count":..., ...}

        # 3) Score => PnL - 0.1 * drawdown
        score = metrics["pnl"] - 0.1 * abs(metrics["max_drawdown"])

        result_item = {
            "params": params,
            "pnl": metrics["pnl"],
            "max_dd": metrics["max_drawdown"],
            "trade_count": metrics["trade_count"],
            "score": score,
            "pivot_count": len(pivs)
        }
        all_results.append(result_item)

        if score > best_score:
            best_score = score
            best_params = params

    # SONUÇ
    print(f"[optimize_parameters] => Best Score for {symbol} - {time_frame} = {best_score:.2f}")
    print(f"[optimize_parameters] => Best Params = {best_params}")

    # Best parametreleri JSON dosyasına kaydedelim
    save_best_params_to_json(
        symbol      = symbol,
        timeframe   = time_frame,
        best_params = best_params,
        best_score  = best_score,
        filename    = None  # Otomatik "btcusdt.json"
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results
    }


# ##############################
# # Örnek Kullanım
# ##############################
# if __name__ == "__main__":
#     # Elinizdeki DataFrame (örnek)
#     df_15m = ...  # BTCUSDT 15m datası
#     df_1h  = ...  # BTCUSDT 1h datası

#     # 1) 15m Optimize
#     res15 = optimize_parameters(df_15m, symbol="BTCUSDT", time_frame="15m")
#     # => best_params kaydedildi => "btcusdt.json" => time_frames["15m"]

#     # 2) 1h Optimize
#     res1h = optimize_parameters(df_1h, symbol="BTCUSDT", time_frame="1h")
#     # => best_params kaydedildi => "btcusdt.json" => time_frames["1h"]

#     # Sonra istediğiniz zaman "best_params"’ı okuyabilirsiniz:
#     best_15m = load_best_params_from_json("BTCUSDT", "15m")
#     print("Loaded best params for 15m =>", best_15m)

#     best_1h = load_best_params_from_json("BTCUSDT", "1h")
#     print("Loaded best params for 1h =>", best_1h)


def calc_strategy_pnl(
    df: pd.DataFrame,
    pivots: list,
    time_frame: str = "1m",
    commission_rate: float = 0.0005,
    slippage: float = 0.0,
    use_stop_loss: bool = True,
    stop_loss_atr_factor: float = 2.0,
    atr_period: int = 14,
    allow_short: bool = True
) -> dict:
    """
    Geliştirilmiş pivot tabanlı strateji:
      - local min (-1) => long pozisyona giriş (yoksa)
      - local max (+1) => short pozisyona giriş (yoksa), 
                         eğer long pozisyon açıksa kapat
      - Komisyon ve slipaj => gerçekçi maliyet
      - Opsiyonel ATR tabanlı stop-loss
      - Basit PnL, max drawdown, trade count döndürür.

    DÖNÜŞ:
      {
        "pnl": float,               # nihai net kar-zarar
        "max_drawdown": float,      # basit peak-to-trough
        "trade_count": int,
        "long_trades": int,
        "short_trades": int
      }
    """

    # Gerekliyse ATR hesapla (stop-loss için)
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        return {"pnl": 0.0, "max_drawdown": 0.0, "trade_count": 0, "long_trades":0, "short_trades":0}

    if use_stop_loss:
        high_col = f"High_{time_frame}"
        low_col  = f"Low_{time_frame}"
        atr_col  = f"ATR_{time_frame}"
        if atr_col not in df.columns:
            # ATR hesapla
            from copy import deepcopy
            # Basit bir ATR fonksiyonu ekleyebilirsiniz veya yeniden
            # prepare_atr(...)'i çağırın
            # (Kodu yeniden kopyalamamak adına kısaltıyorum)
            pass

    sorted_piv = sorted(pivots, key=lambda x: x[0])  # bar_index'e göre sırala

    position = 0    # 0 => flat, +1 => long, -1 => short
    position_price = 0.0
    realized_pnl = 0.0

    # Trade metriği takibi
    trade_count = 0
    long_trades = 0
    short_trades= 0

    # equity takibi => max_drawdown ölçümü basit
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0

    for (bar_idx, price, ptype) in sorted_piv:
        # ptype = -1 => local min (potansiyel dip)
        # ptype = +1 => local max (potansiyel tepe)

        current_close = price
        # slipaj => alışta fiyatı biraz yukarı, satışta biraz aşağı
        # basit model => slipaj: current_close += slippage (long alımda)
        # short açarken => current_close -= slippage
        # Bu elbette abartılı / basit bir model

        ### STOP-LOSS KONTROLÜ (örnek, bar bazında)
        if position != 0 and use_stop_loss:
            # ATR tabanlı stop => position_price +/- (stop_loss_atr_factor * ATR)
            # Basitçe bar_idx'teki ATR'yi al
            # ...
            pass  # (Detay implementasyonu size kalmış)

        # 1) LONG GİRİŞ (dip pivot) => ptype==-1
        if ptype == -1:
            # Eğer halihazırda pozisyonumuz yoksa (veya short'taysak?), long açabiliriz
            if position <= 0:  # Pozisyon flat veya short
                # Kapat short (varsa)
                if position == -1:
                    # short kapat => realized pnl
                    close_price = current_close - slippage
                    trade_pnl = (position_price - close_price)  # short => giriş - çıkış
                    # komisyon
                    trade_pnl -= commission_rate * abs(position_price + close_price)/2
                    realized_pnl += trade_pnl
                    equity += trade_pnl
                    trade_count += 1
                    short_trades +=1
                    position = 0
                    position_price = 0.0

                # şimdi long aç
                open_price = current_close + slippage
                # komisyon
                cost = commission_rate * open_price
                position_price = open_price
                position = 1
                equity -= cost

        # 2) SHORT GİRİŞ (tepe pivot) => ptype==+1
        elif ptype == +1:
            # Eğer halihazırda pozisyonumuz yoksa (veya long'taysak?), short açabiliriz
            if allow_short and position >= 0:
                # Kapat long (varsa)
                if position == 1:
                    close_price = current_close - slippage
                    trade_pnl = (close_price - position_price)  # long => çıkış - giriş
                    trade_pnl -= commission_rate * abs(position_price + close_price)/2
                    realized_pnl += trade_pnl
                    equity += trade_pnl
                    trade_count += 1
                    long_trades +=1
                    position = 0
                    position_price = 0.0

                # Şimdi short aç
                open_price = current_close - slippage
                cost = commission_rate * open_price
                position_price = open_price
                position = -1
                equity -= cost

    # DÖNEM SONUNDA => açık pozisyonu kapatalım (opsiyonel)
    if position != 0:
        last_close = df[close_col].iloc[-1]
        if position == 1:   # longta isek
            trade_pnl = (last_close - position_price)
            trade_pnl -= commission_rate * abs(position_price + last_close)/2
            realized_pnl += trade_pnl
            equity += trade_pnl
            trade_count += 1
            long_trades +=1
        else:  # short
            trade_pnl = (position_price - last_close)
            trade_pnl -= commission_rate * abs(position_price + last_close)/2
            realized_pnl += trade_pnl
            equity += trade_pnl
            trade_count += 1
            short_trades+=1

        position = 0
        position_price= 0.0

    # Max drawdown basit hesap: 
    # her işlem sonrası equity'yi güncelleyip peak_equity takip edin
    # Yukarıda her trade sonrası equity güncelledik => orada peak_equity & dd bakabiliriz
    # Bu örnekte dd = peak - en düşük seviye. 
    # Tam implementasyon size kalmış.

    return {
        "pnl": round(realized_pnl, 4),
        "max_drawdown": round(max_drawdown, 4),  # (örnek: 0.0, uygulamadık)
        "trade_count": trade_count,
        "long_trades": long_trades,
        "short_trades": short_trades
    }

##############################################################################
# 2) HEAD & SHOULDERS (Advanced) - Retest & RSI/MACD onay dahil
##############################################################################
def detect_head_and_shoulders_advanced(
        pivots,
    df: pd.DataFrame,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_distance_bars: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0,
    check_rsi_macd: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01,
) -> list:
    """
    Advanced Head & Shoulders detection with:
      - volume_decline check
      - neckline_break confirmation
      - optional RSI/MACD check
      - optional neckline retest check
    """
    high_col   = get_col_name("High",  time_frame)
    low_col    = get_col_name("Low",   time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume",time_frame)
    atr_col    = get_col_name("ATR",   time_frame)

    # Prepare ATR if needed
    if atr_filter>0:
        prepare_atr(df, time_frame)

    # 1) Find top pivots (using a simple PivotScanner or direct logic)
    piv_scanner = PivotScanner(
        df, time_frame,
        left_bars=left_bars, 
        right_bars=right_bars,
        volume_factor=0,   # pivot detection only; will check volume after
        atr_factor=0,
    )
    #pivot_list = piv_scanner.find_pivots()
    top_pivots = [p for p in pivots if p[2]==+1]
    #print("top pivots",top_pivots)
    results=[]
    for i in range(len(top_pivots)-2):
        L= top_pivots[i]
        H= top_pivots[i+1]
        R= top_pivots[i+2]
        idxL, priceL,_= L
        idxH, priceH,_= H
        idxR, priceR,_= R

        # Basic position check
        if not (idxL< idxH< idxR):
            continue
        # The "head" must be higher than shoulders
        if not (priceH> priceL and priceH> priceR):
            continue

        bars_LH= idxH- idxL
        bars_HR= idxR- idxH
        if bars_LH< min_distance_bars or bars_HR< min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue

        # Shoulder height difference
        diffShoulder= abs(priceL- priceR)/(priceH+ 1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        # ATR filter => optional check on the head pivot
        if atr_filter>0 and atr_col in df.columns:
            head_atr= df[atr_col].iloc[idxH]
            if head_atr>0:
                # e.g. check if priceH - previous close < something
                pass

        # Volume check => Head volume < average(shoulders) * factor
        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            # e.g. "head volume 20% lower than shoulders"
            if volH > (mean_shoulder_vol*0.8):
                vol_check= False

        # Neckline => lowest dips between L-H and H-R
        segment_LH= df[low_col].iloc[idxL: idxH+1]
        segment_HR= df[low_col].iloc[idxH: idxR+1]
        if len(segment_LH)<1 or len(segment_HR)<1:
            continue
        dip1_idx= segment_LH.idxmin()
        dip2_idx= segment_HR.idxmin()
        dip1_val= df[low_col].iloc[dip1_idx]
        dip2_val= df[low_col].iloc[dip2_idx]

        # Neckline break => check a bar below that line
        confirmed= False
        confirmed_bar = None
        if neckline_break:
            if dip1_idx != dip2_idx:
                m_ = (dip2_val - dip1_val)/(dip2_idx - dip1_idx + 1e-9)
                b_ = dip1_val - m_*dip1_idx
                for test_i in range(idxR, len(df)):
                    c = df[close_col].iloc[test_i]
                    line_y = m_* test_i + b_
                    if c < line_y:
                        confirmed= True
                        confirmed_bar = test_i
                        break

        # (Optionally) RSI/MACD check on the break bar
        indicator_res = None
        if check_rsi_macd and confirmed and (confirmed_bar is not None):
            indicator_res = indicator_checks(df, confirmed_bar, time_frame=time_frame,
                                             rsi_check=True, macd_check=True)

        # Retest check
        retest_info = None
        if check_retest and confirmed and (confirmed_bar is not None):
            retest_info = check_retest_levels(
                df, time_frame,
                neckline_points=((dip1_idx, dip1_val),(dip2_idx, dip2_val)),
                break_bar=confirmed_bar,
                tolerance=retest_tolerance
            )

        results.append({
            "L": (idxL, priceL),
            "H": (idxH, priceH),
            "R": (idxR, priceR),
            "shoulder_diff": diffShoulder,
            "volume_check": vol_check,
            "neckline": ((dip1_idx, dip1_val),(dip2_idx, dip2_val)),
            "confirmed": confirmed,
            "confirmed_bar": confirmed_bar,
            "indicator_check": indicator_res,
            "retest_info": retest_info
        })

    return results


def check_retest_levels(df: pd.DataFrame,
                        time_frame: str,
                        neckline_points: tuple,
                        break_bar: int,
                        tolerance: float = 0.01) -> dict:
    """
    After a neckline break, check if there's a retest of that neckline.
    Neckline is defined by 2 points => line_equation => search if subsequent 
    close is near the line within `tolerance`.
    """
    if not neckline_points:
        return {"retest_done": False, "retest_bar": None}

    (x1, y1), (x2, y2) = neckline_points
    m_, b_ = line_equation(x1, y1, x2, y2)
    if m_ is None:
        return {"retest_done": False, "retest_bar": None}

    close_col = get_col_name("Close", time_frame)
    retest_done = False
    retest_bar = None

    for i in range(break_bar+1, len(df)):
        c = df[close_col].iloc[i]
        line_y = m_*i + b_
        diff_perc = abs(c - line_y)/(line_y+1e-9)
        if diff_perc <= tolerance:
            retest_done = True
            retest_bar = i
            break

    return {
        "retest_done": retest_done,
        "retest_bar": retest_bar
    }


##############################################################################
# 5) Örnek: Tamamlanmamış (Devam Eden) Head & Shoulders Tespiti
##############################################################################
def detect_incomplete_head_and_shoulders(pivots,df: pd.DataFrame,
                                         time_frame: str="1m",
                                         left_bars: int=10,
                                         right_bars: int=10,
                                         potential_right_shoulder: bool=True) -> list:
    """
    Looks for a partial H&S: we have left shoulder + head, but no right shoulder yet.
    This is an 'early warning' approach and can have more false positives.
    """
    #pivot_scanner = PivotScanner(df, time_frame, left_bars, right_bars, volume_factor=0, atr_factor=0)
    #pivots = pivot_scanner.find_pivots()
    top_pivots = [p for p in pivots if p[2]==+1]

    incomplete_list = []
    for i in range(len(top_pivots)-1):
        L= top_pivots[i]
        H= top_pivots[i+1]
        if L[0] < H[0] and H[1] > L[1]:
            incomplete_list.append({
                "potential_pattern": "incomplete_HS",
                "L": L,
                "H": H,
                "comment": "Right shoulder not formed yet."
            })
    return incomplete_list


##############################################################################
# 3) INVERSE HEAD & SHOULDERS (Advanced)
##############################################################################
def detect_inverse_head_and_shoulders_advanced(
        pivots,
    df: pd.DataFrame,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_distance_bars: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0,
    check_rsi_macd: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01,
) -> list:
    """
    Advanced Inverse Head & Shoulders detection.
    Similar logic to normal H&S but inverted.
    """
    low_col   = get_col_name("Low", time_frame)
    close_col = get_col_name("Close", time_frame)
    volume_col= get_col_name("Volume", time_frame)
    atr_col   = get_col_name("ATR", time_frame)

    # ATR if needed
    if atr_filter>0:
        prepare_atr(df, time_frame)

    # Pivot scan => find dips
    pivot_scanner= PivotScanner(
        df, time_frame,
        left_bars= left_bars,
        right_bars= right_bars,
        volume_factor=0,
        atr_factor=0
    )
    #pivot_list= pivot_scanner.find_pivots()
    dip_pivots= [p for p in pivots if p[2]== -1]

    results=[]
    for i in range(len(dip_pivots)-2):
        L= dip_pivots[i]
        H= dip_pivots[i+1]
        R= dip_pivots[i+2]
        idxL, priceL,_= L
        idxH, priceH,_= H
        idxR, priceR,_= R

        if not (idxL< idxH< idxR):
            continue
        if not (priceH< priceL and priceH< priceR):
            continue

        bars_LH= idxH- idxL
        bars_HR= idxR- idxH
        if bars_LH< min_distance_bars or bars_HR< min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue

        diffShoulder= abs(priceL- priceR)/ ((priceL+priceR)/2 +1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        # Volume check
        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            if volH> (mean_shoulder_vol*0.8):
                vol_check= False

        # Neckline => local max
        seg_LH= df[close_col].iloc[idxL: idxH+1]
        seg_HR= df[close_col].iloc[idxH: idxR+1]
        if len(seg_LH)<1 or len(seg_HR)<1:
            continue
        T1_idx= seg_LH.idxmax()
        T2_idx= seg_HR.idxmax()
        T1_val= df[close_col].iloc[T1_idx]
        T2_val= df[close_col].iloc[T2_idx]

        confirmed= False
        confirmed_bar= None
        if neckline_break:
            m_, b_= line_equation(T1_idx, T1_val, T2_idx, T2_val)
            if m_ is not None:
                for test_i in range(idxR, len(df)):
                    c = df[close_col].iloc[test_i]
                    line_y = m_* test_i + b_
                    if c > line_y:
                        confirmed= True
                        confirmed_bar= test_i
                        break

        indicator_res = None
        if check_rsi_macd and confirmed and (confirmed_bar is not None):
            indicator_res = indicator_checks(df, confirmed_bar, time_frame=time_frame,
                                             rsi_check=True, macd_check=True)

        retest_info = None
        if check_retest and confirmed and (confirmed_bar is not None):
            retest_info = check_retest_levels(
                df, time_frame,
                neckline_points=((T1_idx, T1_val),(T2_idx, T2_val)),
                break_bar=confirmed_bar,
                tolerance=retest_tolerance
            )

        results.append({
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "shoulder_diff": diffShoulder,
          "volume_check": vol_check,
          "confirmed": confirmed,
          "confirmed_bar": confirmed_bar,
          "neckline": ((T1_idx,T1_val), (T2_idx,T2_val)),
          "indicator_check": indicator_res,
          "retest_info": retest_info
        })
    return results


##############################################################################
# 4) DOUBLE / TRIPLE TOP - BOTTOM (Advanced)
##############################################################################
def detect_double_top(
    pivots,
    df: pd.DataFrame = None,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Detects Double or Triple Tops using pivot list (pivots with +1 => top).
    - If triple_variation=True, it can extend to a 3rd top if present.
    - volume_check => optional volume-based filter
    - neckline_break => checks for 'confirmed' if price breaks below local dip
    - check_retest => if there's a retest of the neckline after the break
    """
    top_pivots= [p for p in pivots if p[2]== +1]
    if len(top_pivots)<2:
        return []

    volume_col= get_col_name("Volume", time_frame)
    close_col = get_col_name("Close", time_frame)

    results=[]
    i=0
    while i< len(top_pivots)-1:
        t1= top_pivots[i]
        t2= top_pivots[i+1]
        idx1,price1= t1[0], t1[1]
        idx2,price2= t2[0], t2[1]

        avgp= (price1+ price2)/2
        pdiff= abs(price1- price2)/(avgp+1e-9)
        bar_diff= idx2- idx1

        # Are these two tops close enough in price & far enough in time?
        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
              "tops": [(idx1,price1),(idx2,price2)],
              "neckline": None,
              "confirmed": False,
              "start_bar": idx1,
              "end_bar": idx2,
              "pattern": "double_top",
              "retest_info": None
            }

            used_third= False
            # triple top check
            if triple_variation and (i+2< len(top_pivots)):
                t3= top_pivots[i+2]
                idx3,price3= t3[0], t3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found["tops"]= [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found["end_bar"]= idx3
                    found["pattern"]= "triple_top"
                    used_third= True

            # volume check
            if volume_check and df is not None and volume_col in df.columns:
                vol1= df[volume_col].iloc[idx1]
                vol2= df[volume_col].iloc[idx2]
                # e.g. 2nd top volume < 80% of 1st => custom rule
                if vol2 > (vol1 * 0.8):
                    i+= (2 if used_third else 1)
                    continue

            # neckline
            if neckline_break and df is not None and close_col in df.columns:
                seg_end= found["end_bar"]
                # find lowest dip pivot in that range
                dips_for_neck = [pp for pp in pivots if pp[2]== -1 and (pp[0]> idx1 and pp[0]< seg_end)]
                if dips_for_neck:
                    dips_sorted= sorted(dips_for_neck, key=lambda x: x[1])  # ascending by price
                    neck = dips_sorted[0]
                    found["neckline"]= (neck[0], neck[1])

                    # confirm => last close < neckline => simplistic example
                    last_close = df[close_col].iloc[-1]
                    if last_close < neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1

                        # retest?
                        if check_retest:
                            retest_info= _check_retest_doubletop(
                                df, time_frame,
                                neckline_price= neck[1],
                                confirm_bar= len(df)-1,
                                retest_tolerance= retest_tolerance
                            )
                            found["retest_info"]= retest_info

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results

def _check_retest_doubletop(df: pd.DataFrame,
                            time_frame: str,
                            neckline_price: float,
                            confirm_bar: int,
                            retest_tolerance: float=0.01) -> dict:
    """
    After a confirmed double-top break, check if there's a retest of the neckline area.
    """
    close_col = get_col_name("Close", time_frame)
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None}

    n= len(df)
    if confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }

    return {"retest_done": False, "retest_bar": None}

def detect_double_bottom(
    pivots,
    df: pd.DataFrame = None,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Similar to detect_double_top but for bottoms.
    """
    bottom_pivots= [p for p in pivots if p[2]== -1]
    if len(bottom_pivots)<2:
        return []

    volume_col= get_col_name("Volume", time_frame)
    close_col = get_col_name("Close", time_frame)

    results=[]
    i=0
    while i< len(bottom_pivots)-1:
        b1= bottom_pivots[i]
        b2= bottom_pivots[i+1]
        idx1,price1= b1[0], b1[1]
        idx2,price2= b2[0], b2[1]

        avgp= (price1+ price2)/2
        pdiff= abs(price1- price2)/(avgp+1e-9)
        bar_diff= idx2- idx1

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
              "bottoms": [(idx1,price1),(idx2,price2)],
              "neckline": None,
              "confirmed": False,
              "start_bar": idx1,
              "end_bar": idx2,
              "pattern": "double_bottom",
              "retest_info": None
            }
            used_third= False

            # triple?
            if triple_variation and (i+2< len(bottom_pivots)):
                b3= bottom_pivots[i+2]
                idx3,price3= b3[0], b3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found["bottoms"]= [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found["end_bar"]= idx3
                    found["pattern"]= "triple_bottom"
                    used_third= True

            if volume_check and df is not None and volume_col in df.columns:
                vol1= df[volume_col].iloc[idx1]
                vol2= df[volume_col].iloc[idx2]
                if vol2 > (vol1 * 0.8):
                    i+= (2 if used_third else 1)
                    continue

            # neckline => highest top pivot between idx1..idx2
            if neckline_break and df is not None and close_col in df.columns:
                seg_end= found["end_bar"]
                top_pivs= [pp for pp in pivots if pp[2]== +1 and pp[0]> idx1 and pp[0]< seg_end]
                if top_pivs:
                    top_sorted= sorted(top_pivs, key=lambda x: x[1], reverse=True)
                    neck= top_sorted[0]
                    found["neckline"]= (neck[0], neck[1])

                    last_close= df[close_col].iloc[-1]
                    if last_close> neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1

                        # retest
                        if check_retest:
                            retest_info = _check_retest_dblbottom(
                                df, time_frame,
                                neckline_price= neck[1],
                                confirm_bar=len(df)-1,
                                retest_tolerance=retest_tolerance
                            )
                            found["retest_info"] = retest_info

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results

def _check_retest_dblbottom(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float=0.01
) -> dict:
    """
    Double Bottom => after break above neckline, does price come back to retest it?
    """
    close_col= get_col_name("Close", time_frame)
    n= len(df)
    if close_col not in df.columns or confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }

    return {"retest_done": False, "retest_bar": None}

def detect_triple_top_advanced(
    pivots,
    df: pd.DataFrame = None,
    time_frame: str = "1m",
    tolerance: float = 0.01,
    min_distance_bars: int = 20,
    volume_check: bool = False,
    volume_col_factor: float = 0.8,
    neckline_break: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> list:
    """
    Gelişmiş Triple Top (Üçlü Tepe) tespiti:
      - 3 adet tepe pivotu, fiyatları birbirine yakın (tolerance)
      - Her tepe arasında min_distance_bars kadar bar olmalı
      - İsteğe bağlı volume check
      - Neckline kırılım onayı (isteğe bağlı)
      - Retest kontrolü (isteğe bağlı)

    DÖNÜŞ:
      - Her tespit edilen Triple Top için dict listesi
        [
          {
            "tops": [(idxT1, priceT1), (idxT2, priceT2), (idxT3, priceT3)],
            "pattern": "triple_top",
            "neckline": (idxNeck, priceNeck) or None,
            "confirmed": bool,
            "retest_info": {...} or None,
            "volume_check": bool,
            "msgs": [],
            ...
          },
          ...
        ]
    """
    # Yalnızca +1 pivotlar = tepe
    top_pivots = [p for p in pivots if p[2] == +1]
    if len(top_pivots) < 3:
        return []

    close_col  = f"Close_{time_frame}"
    volume_col = f"Volume_{time_frame}"
    results = []
    i = 0
    while i < len(top_pivots) - 2:
        t1 = top_pivots[i]
        t2 = top_pivots[i+1]
        t3 = top_pivots[i+2]

        idx1, price1 = t1[0], t1[1]
        idx2, price2 = t2[0], t2[1]
        idx3, price3 = t3[0], t3[1]

        # Zaman aralığı kontrolü (yeterli bar var mı?)
        bar_diff_12 = idx2 - idx1
        bar_diff_23 = idx3 - idx2
        if bar_diff_12 < min_distance_bars or bar_diff_23 < min_distance_bars:
            i += 1
            continue

        # Fiyat yakınlığı kontrolü
        avgp = (price1 + price2 + price3) / 3.0
        pdiff_1 = abs(price1 - avgp) / (avgp + 1e-9)
        pdiff_2 = abs(price2 - avgp) / (avgp + 1e-9)
        pdiff_3 = abs(price3 - avgp) / (avgp + 1e-9)
        if any(p > tolerance for p in [pdiff_1, pdiff_2, pdiff_3]):
            i += 1
            continue

        # Volume check (örnek: 2. veya 3. tepenin hacmi, 1. tepeden daha düşük olsun gibi)
        vol_ok = True
        msgs = []
        if volume_check and df is not None and volume_col in df.columns:
            vol1 = df[volume_col].iloc[idx1] if idx1 < len(df) else None
            vol2 = df[volume_col].iloc[idx2] if idx2 < len(df) else None
            vol3 = df[volume_col].iloc[idx3] if idx3 < len(df) else None
            # Örneğin 3. tepenin hacmi, 1. ve 2. nin ortalamasından düşük mü?
            if vol1 and vol2 and vol3:
                mean_top_vol = (vol1 + vol2) / 2.0
                # volume_col_factor => default 0.8 (örn. %80)
                if vol3 > (mean_top_vol * volume_col_factor):
                    vol_ok = False
                    msgs.append(f"3rd top volume not low enough (vol3={vol3:.2f}, mean12={mean_top_vol:.2f})")

        # Neckline => genellikle 1.-2.-3. tepe aralarında oluşan diplerin en düşüğünün yakınında
        # En basit yaklaşım: idx1..idx3 aralığındaki -1 pivotlardan en düşük olanını al.
        seg_min_pivots = [p for p in pivots if p[2] == -1 and p[0] > idx1 and p[0] < idx3]
        neckline = None
        if seg_min_pivots:
            sorted_dips = sorted(seg_min_pivots, key=lambda x: x[1])  # fiyata göre sıralar
            neckline = (sorted_dips[0][0], sorted_dips[0][1])  # (index, price)
        else:
            msgs.append("No local dip pivot found for neckline.")

        # Neckline break onayı
        conf = False
        retest_data = None
        if neckline_break and neckline is not None and df is not None:
            neck_idx, neck_prc = neckline
            last_close = df[close_col].iloc[-1]
            if last_close < neck_prc:  # triple top => aşağı kırılım arıyoruz
                conf = True
                # Retest ?
                if check_retest:
                    retest_data = _check_retest_triple_top(
                        df, time_frame,
                        neckline_price=neck_prc,
                        confirm_bar=len(df)-1,
                        retest_tolerance=retest_tolerance
                    )
            else:
                msgs.append("Neckline not broken => not confirmed")

        pattern_info = {
            "tops": [(idx1, price1), (idx2, price2), (idx3, price3)],
            "pattern": "triple_top",
            "neckline": neckline,
            "confirmed": conf,
            "volume_check": vol_ok,
            "msgs": msgs,
            "retest_info": retest_data
        }

        # Son olarak volume check başarısızsa pattern geçersiz sayalım (opsiyonel)
        if vol_ok:
            results.append(pattern_info)

        # i'yi 3 adım atlatmak yerine 1 adım atlatıyoruz ki
        # bir pivot seti içinde başka triple-top kombinasyonu varsa da yakalayabilelim
        i += 1

    return results


def _check_retest_triple_top(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float = 0.01
) -> dict:
    """
    Triple Top => neckline kırıldıktan sonra, fiyatın tekrar neckline seviyesini
    yukarıdan test edip etmediğini kontrol eder.
    """
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None}

    n = len(df)
    if confirm_bar >= n - 1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c = df[close_col].iloc[i]
        dist_ratio = abs(c - neckline_price) / (abs(neckline_price) + 1e-9)
        if dist_ratio <= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}


def detect_triple_bottom_advanced(
    pivots,
    df: pd.DataFrame = None,
    time_frame: str = "1m",
    tolerance: float = 0.01,
    min_distance_bars: int = 20,
    volume_check: bool = False,
    volume_col_factor: float = 0.8,
    neckline_break: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> list:
    """
    Gelişmiş Triple Bottom (Üçlü Dip) tespiti:
      - 3 adet dip pivotu, fiyatları birbirine yakın (tolerance)
      - Her dip arasında min_distance_bars kadar bar olmalı
      - İsteğe bağlı volume check
      - Neckline kırılım onayı (isteğe bağlı)
      - Retest kontrolü (isteğe bağlı)

    DÖNÜŞ:
      - Her tespit edilen Triple Bottom için dict listesi
        [
          {
            "bottoms": [(idxB1, priceB1), (idxB2, priceB2), (idxB3, priceB3)],
            "pattern": "triple_bottom",
            "neckline": (idxNeck, priceNeck) or None,
            "confirmed": bool,
            "retest_info": {...} or None,
            "volume_check": bool,
            "msgs": [],
            ...
          },
          ...
        ]
    """
    # Yalnızca -1 pivotlar = dip
    bot_pivots = [p for p in pivots if p[2] == -1]
    if len(bot_pivots) < 3:
        return []

    close_col  = f"Close_{time_frame}"
    volume_col = f"Volume_{time_frame}"
    results = []
    i = 0
    while i < len(bot_pivots) - 2:
        b1 = bot_pivots[i]
        b2 = bot_pivots[i+1]
        b3 = bot_pivots[i+2]

        idx1, price1 = b1[0], b1[1]
        idx2, price2 = b2[0], b2[1]
        idx3, price3 = b3[0], b3[1]

        # Zaman aralığı kontrolü
        bar_diff_12 = idx2 - idx1
        bar_diff_23 = idx3 - idx2
        if bar_diff_12 < min_distance_bars or bar_diff_23 < min_distance_bars:
            i += 1
            continue

        # Fiyat yakınlığı kontrolü
        avgp = (price1 + price2 + price3) / 3.0
        pdiff_1 = abs(price1 - avgp) / (avgp + 1e-9)
        pdiff_2 = abs(price2 - avgp) / (avgp + 1e-9)
        pdiff_3 = abs(price3 - avgp) / (avgp + 1e-9)
        if any(p > tolerance for p in [pdiff_1, pdiff_2, pdiff_3]):
            i += 1
            continue

        # Volume check
        vol_ok = True
        msgs = []
        if volume_check and df is not None and volume_col in df.columns:
            vol1 = df[volume_col].iloc[idx1] if idx1 < len(df) else None
            vol2 = df[volume_col].iloc[idx2] if idx2 < len(df) else None
            vol3 = df[volume_col].iloc[idx3] if idx3 < len(df) else None
            if vol1 and vol2 and vol3:
                mean_bot_vol = (vol1 + vol2) / 2.0
                if vol3 > (mean_bot_vol * volume_col_factor):
                    vol_ok = False
                    msgs.append(f"3rd bottom volume not low enough (vol3={vol3:.2f}, mean12={mean_bot_vol:.2f})")

        # Neckline => local max pivot(lar) arasından en yükseği
        seg_max_pivots = [p for p in pivots if p[2] == +1 and p[0] > idx1 and p[0] < idx3]
        neckline = None
        if seg_max_pivots:
            sorted_tops = sorted(seg_max_pivots, key=lambda x: x[1], reverse=True)
            neckline = (sorted_tops[0][0], sorted_tops[0][1])
        else:
            msgs.append("No local top pivot found for neckline.")

        # Neckline break onayı
        conf = False
        retest_data = None
        if neckline_break and neckline is not None and df is not None:
            neck_idx, neck_prc = neckline
            last_close = df[close_col].iloc[-1]
            if last_close > neck_prc:  # triple bottom => yukarı kırılım arıyoruz
                conf = True
                # Retest ?
                if check_retest:
                    retest_data = _check_retest_triple_bottom(
                        df, time_frame,
                        neckline_price=neck_prc,
                        confirm_bar=len(df)-1,
                        retest_tolerance=retest_tolerance
                    )
            else:
                msgs.append("Neckline not broken => not confirmed")

        pattern_info = {
            "bottoms": [(idx1, price1), (idx2, price2), (idx3, price3)],
            "pattern": "triple_bottom",
            "neckline": neckline,
            "confirmed": conf,
            "volume_check": vol_ok,
            "msgs": msgs,
            "retest_info": retest_data
        }

        if vol_ok:
            results.append(pattern_info)

        i += 1

    return results


def _check_retest_triple_bottom(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float = 0.01
) -> dict:
    """
    Triple Bottom => neckline kırıldıktan sonra, fiyatın tekrar neckline seviyesini
    aşağıdan test edip etmediğini kontrol eder.
    """
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None}

    n = len(df)
    if confirm_bar >= n - 1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c = df[close_col].iloc[i]
        dist_ratio = abs(c - neckline_price) / (abs(neckline_price) + 1e-9)
        if dist_ratio <= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}

##############################################################################
# 6) WOLFE WAVE (Advanced)
##############################################################################
def detect_wolfe_wave_advanced(
    wave,
    df: pd.DataFrame = None,
    time_frame: str = "1m",
    price_tolerance: float = 0.03,
    strict_lines: bool = False,
    breakout_confirm: bool = True,
    line_projection_check: bool = True,
    check_2_4_slope: bool = True,
    check_1_4_intersection_time: bool = True,
    check_time_symmetry: bool = True,
    max_time_ratio: float = 0.3,
    check_retest: bool = False,          
    retest_tolerance: float = 0.01       
) -> dict:
    """
    Gelişmiş Wolfe Wave dedektörü (v2) + 'retest' eklendi.
    `wave` => at least 5 pivot points in order (1..5).
    """
    result= {
      "found": False,
      "msgs": [],
      "breakout": False,
      "intersection": None,
      "time_symmetry_ok": True,
      "sweet_zone": None,
      "wolfe_line": None,
      "retest_info": None
    }

    if len(wave)<5:
        result["msgs"].append("Not enough pivots (need 5).")
        return result

    # 1..5
    w1= wave[-5]
    w2= wave[-4]
    w3= wave[-3]
    w4= wave[-2]
    w5= wave[-1]
    x1,y1,_= w1
    x2,y2,_= w2
    x3,y3,_= w3
    x4,y4,_= w4
    x5,y5,_= w5

    # line(1->3), line(3->5)
    m13,b13= line_equation(x1,y1, x3,y3)
    m35,b35= line_equation(x3,y3, x5,y5)
    if (m13 is None) or (m35 is None):
        result["msgs"].append("Line(1->3) or (3->5) vertical => fail.")
        return result

    # slope difference
    diff_slope= abs(m35- m13)/(abs(m13)+1e-9)
    if diff_slope> price_tolerance:
        result["msgs"].append(f"Slope difference(1->3 vs 3->5) too big => {diff_slope:.3f}")

    # optional check_2_4_slope
    if check_2_4_slope:
        m24,b24= line_equation(x2,y2, x4,y4)
        if strict_lines and (m24 is not None):
            slope_diff= abs(m24- m13)/(abs(m13)+1e-9)
            if slope_diff> 0.3:
                result["msgs"].append("Line(2->4) slope differs from line(1->3).")

    # sweet zone => check if w5 is between line(1->3) & line(2->4)
    m24_,b24_= line_equation(x2,y2, x4,y4)
    if m24_ is not None:
        line13_y5= m13*x5+ b13
        line24_y5= m24_*x5+ b24_
        low_  = min(line13_y5, line24_y5)
        high_ = max(line13_y5, line24_y5)
        result["sweet_zone"]= (low_, high_)
        if not (low_<= y5<= high_):
            result["msgs"].append("W5 not in sweet zone")

    # Time symmetry => bar counts
    if check_time_symmetry:
        bars_23= x3- x2
        bars_34= x4- x3
        bars_45= x5- x4
        def ratio(a,b): return abs(a-b)/(abs(b)+1e-9)
        r1= ratio(bars_23,bars_34)
        r2= ratio(bars_34,bars_45)
        if (r1> max_time_ratio) or (r2> max_time_ratio):
            result["time_symmetry_ok"]= False
            result["msgs"].append(f"Time symmetry fail => r1={r1:.2f}, r2={r2:.2f}")

    # line_projection => intersection(1->4 & 2->3)
    if line_projection_check:
        m14,b14= line_equation(x1,y1, x4,y4)
        m23,b23= line_equation(x2,y2, x3,y3)
        if (m14 is not None) and (m23 is not None):
            ix,iy= line_intersection(m14,b14, m23,b23)
            if ix is not None:
                result["intersection"]= (ix, iy)
                if check_1_4_intersection_time and ix< x5:
                    result["msgs"].append("Intersection(1->4 & 2->3) < w5 => degrade")

    # breakout => last close above line(1->4) 
    if breakout_confirm and df is not None:
        close_col= get_col_name("Close", time_frame)
        if close_col in df.columns:
            last_close= df[close_col].iloc[-1]
            m14,b14= line_equation(x1,y1, x4,y4)
            if m14 is not None:
                last_i= len(df)-1
                line_y= m14* last_i + b14
                if last_close> line_y:
                    result["breakout"]= True
                    result["wolfe_line"] = ((x1,y1), (x4,y4))
                else:
                    result["msgs"].append("No breakout => last_close below line(1->4).")

    result["found"]= True

    # Retest check
    if check_retest and result["breakout"] and result["wolfe_line"]:
        (ixA,pxA),(ixB,pxB) = result["wolfe_line"]
        m_, b_= line_equation(ixA, pxA, ixB, pxB)
        if m_ is not None and df is not None:
            close_col= get_col_name("Close", time_frame)
            if close_col in df.columns:
                last_i= len(df)-1
                retest_done= False
                retest_bar = None
                for i in range(last_i+1, len(df)):
                    c= df[close_col].iloc[i]
                    line_val= m_* i + b_
                    dist_perc= abs(c - line_val)/(abs(line_val)+1e-9)
                    if dist_perc<= retest_tolerance:
                        retest_done= True
                        retest_bar= i
                        break
                result["retest_info"]= {
                    "retest_done": retest_done,
                    "retest_bar": retest_bar
                }

    return result


def detect_elliott_5wave_advanced(
    wave,
    df: pd.DataFrame = None,
    time_frame: str = "1m",
    fib_tolerance: float = 0.1,
    wave_min_bars: int = 5,
    extended_waves: bool = True,
    rule_3rdwave_min_percent: float = 1.618,
    rule_5thwave_ext_range: tuple = (1.0, 1.618),
    check_alt_scenarios: bool = True,
    check_abc_correction: bool = True,
    allow_4th_overlap: bool = False,
    min_bar_distance: int = 3,
    check_fib_retracements: bool = True,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> dict:
    """
    Elliott 5 Wave detection (v2) + optional retest of wave4 pivot.
    """
    result = {
        "found": False,
        "trend": None,
        "pivots": [],
        "check_msgs": [],
        "abc": None,
        "extended_5th": False,
        "wave4_level": None,
        "retest_info": None
    }

    if len(wave) < wave_min_bars:
        result["check_msgs"].append("Not enough pivots for Elliott 5-wave.")
        return result

    last5 = wave[-5:]
    types = [p[2] for p in last5]
    up_pattern   = [+1, -1, +1, -1, +1]
    down_pattern = [-1, +1, -1, +1, -1]

    # Detect up or down pattern
    if types == up_pattern:
        trend = "UP"
    elif check_alt_scenarios and (types == down_pattern):
        trend = "DOWN"
    else:
        result["check_msgs"].append("Pivot pattern not matching up or down Elliott.")
        return result

    result["trend"] = trend

    # Label waves
    p0i,p0p,_= last5[0]
    p1i,p1p,_= last5[1]
    p2i,p2p,_= last5[2]
    p3i,p3p,_= last5[3]
    p4i,p4p,_= last5[4]
    result["pivots"] = [(p0i,p0p),(p1i,p1p),(p2i,p2p),(p3i,p3p),(p4i,p4p)]

    def wave_len(a,b): 
        return abs(b-a)

    w1= wave_len(p0p,p1p)
    w2= wave_len(p1p,p2p)
    w3= wave_len(p2p,p3p)
    w4= wave_len(p3p,p4p)

    d1= p1i- p0i
    d2= p2i- p1i
    d3= p3i- p2i
    d4= p4i- p3i
    if any(d< min_bar_distance for d in [d1,d2,d3,d4]):
        result["check_msgs"].append("Bar distance too small between waves.")
        return result

    # 3rd wave length check
    if w3< (rule_3rdwave_min_percent* w1):
        result["check_msgs"].append("3rd wave not long enough vs wave1.")
        return result

    # 4th wave overlap
    if not allow_4th_overlap:
        if trend=="UP" and (p4p< p1p):
            result["check_msgs"].append("4th wave overlap in UP trend.")
            return result
        if trend=="DOWN" and (p4p> p1p):
            result["check_msgs"].append("4th wave overlap in DOWN trend.")
            return result

    # Fib retracements for wave2 & wave4
    if check_fib_retracements:
        w2r= w2/(w1+1e-9)
        w4r= w4/(w3+1e-9)
        typical_min= 0.382- fib_tolerance
        typical_max= 0.618+ fib_tolerance
        if not (typical_min<= w2r<= typical_max):
            result["check_msgs"].append("Wave2 retracement ratio not in typical range.")
        if not (typical_min<= w4r<= typical_max):
            result["check_msgs"].append("Wave4 retracement ratio not in typical range.")

    wave5_ratio= w4/ (w1+1e-9)
    if (wave5_ratio>= rule_5thwave_ext_range[0]) and (wave5_ratio<= rule_5thwave_ext_range[1]):
        result["extended_5th"]= True

    # ABC check => if last 3 pivots fit an ABC after wave5
    if extended_waves and check_abc_correction and (len(wave)>=8):
        maybe_abc= wave[-3:]
        abc_types= [p[2] for p in maybe_abc]
        if trend=="UP":
            if abc_types== [-1,+1,-1]:
                result["abc"]= True
        else:
            if abc_types== [+1,-1,+1]:
                result["abc"]= True

    # Found
    result["found"]= True
    wave4_price= p4p
    wave4_index= p4i
    result["wave4_level"]= wave4_price

    # Retest => wave4 pivot
    if check_retest and df is not None:
        retest_info= _check_retest_elliott_wave4(
            df, time_frame,
            wave4_index= wave4_index,
            wave4_price= wave4_price,
            tolerance= retest_tolerance,
            trend= trend
        )
        result["retest_info"]= retest_info

    return result

def _check_retest_elliott_wave4(
    df: pd.DataFrame,
    time_frame: str,
    wave4_index: int,
    wave4_price: float,
    tolerance: float=0.01,
    trend: str="UP"
) -> dict:
    """
    Simple retest check for wave4 pivot level => see if subsequent bars 
    come back near wave4_price within `tolerance`.
    """
    close_col= get_col_name("Close", time_frame)
    n= len(df)
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    retest_done= False
    retest_bar= None
    retest_dist= None
    for i in range(wave4_index+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - wave4_price)/(abs(wave4_price)+1e-9)
        if dist_ratio<= tolerance:
            retest_done= True
            retest_bar= i
            retest_dist= dist_ratio
            break

    return {
        "retest_done": retest_done,
        "retest_bar": retest_bar,
        "retest_price": wave4_price,
        "distance_ratio": retest_dist
    }


def detect_harmonic_pattern_advanced(
    wave,
    df: pd.DataFrame= None,
    time_frame: str = "1m",
    fib_tolerance: float=0.02,
    patterns: list = None,
    check_volume: bool=False,
    volume_factor: float=1.3,
    check_retest: bool = False,
    retest_tolerance: float=0.01
) -> dict:
    """
    Advanced Harmonic Pattern detection with optional retest check at D.
    """
    if patterns is None:
        patterns= ["gartley","bat","crab","butterfly","shark","cipher"]
    result= {
      "found": False,
      "pattern_name": None,
      "xabc": [],
      "msgs": [],
      "retest_info": None
    }

    if len(wave)<5:
        result["msgs"].append("Not enough pivot for harmonic (need 5).")
        return result

    # X,A,B,C,D
    X= wave[-5]
    A= wave[-4]
    B= wave[-3]
    C= wave[-2]
    D= wave[-1]
    idxX, pxX,_= X
    idxA, pxA,_= A
    idxB, pxB,_= B
    idxC, pxC,_= C
    idxD, pxD,_= D
    result["xabc"]= [X,A,B,C,D]

    def length(a,b): 
        return abs(b-a)
    XA= length(pxX, pxA)
    AB= length(pxA, pxB)
    BC= length(pxB, pxC)
    CD= length(pxC, pxD)

    harmonic_map= {
        "gartley": {
            "AB_XA": (0.618, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.13, 1.618)
        },
        "bat": {
            "AB_XA": (0.382, 0.5),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.618)
        },
        "crab": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (2.24, 3.618)
        },
        "butterfly": {
            "AB_XA": (0.786, 0.786),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.24)
        },
        "shark": {
            "AB_XA": (0.886,1.13),
            "BC_AB": (1.13, 1.618),
            "CD_BC": (0.886,1.13)
        },
        "cipher": {
            "AB_XA": (0.382,0.618),
            "BC_AB": (1.27,2.0),
            "CD_BC": (1.13,1.414)
        }
    }

    def in_range(val, rng, tol):
        mn, mx= rng
        if abs(mn-mx)< 1e-9:  # exact single value
            return abs(val- mn)<= abs(mn)* tol
        else:
            low_= mn- abs(mn)* tol
            high_= mx+ abs(mx)* tol
            return (val>= low_) and (val<= high_)

    AB_XA= AB/(XA+1e-9)
    BC_AB= BC/(AB+1e-9)
    CD_BC= CD/(BC+1e-9)

    found_any= False
    matched_pattern= None

    for pat in patterns:
        if pat not in harmonic_map:
            continue
        spec= harmonic_map[pat]
        rngAB_XA= spec["AB_XA"]
        rngBC_AB= spec["BC_AB"]
        rngCD_BC= spec["CD_BC"]

        ok1= in_range(AB_XA, rngAB_XA, fib_tolerance)
        ok2= in_range(BC_AB, rngBC_AB, fib_tolerance)
        ok3= in_range(CD_BC, rngCD_BC, fib_tolerance)
        if ok1 and ok2 and ok3:
            found_any= True
            matched_pattern= pat
            break

    if found_any:
        result["found"]= True
        result["pattern_name"]= matched_pattern

        # volume check
        volume_col= get_col_name("Volume", time_frame)
        if check_volume and df is not None and volume_col in df.columns and idxD<len(df):
            vol_now= df[volume_col].iloc[idxD]
            prepare_volume_ma(df, time_frame, period=20)
            ma_col= f"Volume_MA_20_{time_frame}"
            if ma_col in df.columns:
                v_mean= df[ma_col].iloc[idxD]
                if (v_mean>0) and (vol_now> volume_factor*v_mean):
                    pass  # e.g. "strong volume at D"

        # retest => D pivot
        if check_retest and df is not None:
            close_col= get_col_name("Close", time_frame)
            if close_col in df.columns:
                retest_done= False
                retest_bar = None
                for i in range(idxD+1, len(df)):
                    c= df[close_col].iloc[i]
                    dist_ratio = abs(c - pxD)/(abs(pxD)+1e-9)
                    if dist_ratio <= retest_tolerance:
                        retest_done= True
                        retest_bar = i
                        break
                if retest_done:
                    result["retest_info"] = {
                        "retest_done": True,
                        "retest_bar": retest_bar,
                        "retest_price": df[close_col].iloc[retest_bar],
                        "comment": "Price re-tested the D pivot region"
                    }
                else:
                    result["retest_info"] = {
                        "retest_done": False
                    }
    else:
        result["msgs"].append("No harmonic pattern match in given list.")

    return result


def detect_wedge_advanced(
    wave,
    df: pd.DataFrame=None,
    time_frame:str="1m",
    wedge_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
) -> dict:
    """
    Advanced Wedge detection (rising/falling), optional breakout & retest.
    """
    result={
      "found":False,
      "wedge_type":None,
      "breakout":False,
      "breakout_line":None,
      "retest_info":None,
      "msgs":[]
    }
    if len(wave)<5:
        result["msgs"].append("Not enough pivot for wedge (need>=5).")
        return result
    last5=wave[-5:]
    types=[p[2]for p in last5]
    rising_pat=[+1,-1,+1,-1,+1]
    falling_pat=[-1,+1,-1,+1,-1]
    if types==rising_pat:
        wedge_type="rising"
    elif types==falling_pat:
        wedge_type="falling"
    else:
        result["msgs"].append("Pivot pattern not matching rising/falling wedge.")
        return result

    x1,y1=last5[0][0],last5[0][1]
    x3,y3=last5[2][0],last5[2][1]
    x5,y5=last5[4][0],last5[4][1]
    slope_top=(y5-y1)/((x5-x1)+1e-9)

    x2,y2=last5[1][0],last5[1][1]
    x4,y4=last5[3][0],last5[3][1]
    slope_bot=(y4-y2)/((x4-x2)+1e-9)

    if wedge_type=="rising":
        if(slope_top<0)or(slope_bot<0):
            result["msgs"].append("Expected positive slopes for rising wedge.")
            return result
        if not(slope_bot> slope_top):
            result["msgs"].append("slope(2->4)<= slope(1->3) => not wedge shape.")
            return result
    else:
        if(slope_top>0)or(slope_bot>0):
            result["msgs"].append("Expected negative slopes for falling wedge.")
            return result
        if not(slope_bot> slope_top):
            result["msgs"].append("Dip slope <= top slope => not wedge shape.")
            return result

    ratio=abs(slope_bot-slope_top)/(abs(slope_top)+1e-9)
    if ratio<wedge_tolerance:
        result["msgs"].append(f"Wedge slope difference ratio {ratio:.3f}<tolerance => might be channel.")

    df_len=len(df) if df is not None else 0
    brk=False
    if check_breakout and df is not None and df_len>0:
        close_col=get_col_name("Close",time_frame)
        last_close=df[close_col].iloc[-1]
        m_,b_= line_equation(x2,y2,x4,y4)
        if m_ is not None:
            last_i=df_len-1
            line_y=m_*last_i+b_
            if wedge_type=="rising":
                if last_close<line_y:
                    brk=True
            else:
                if last_close>line_y:
                    brk=True

    if brk:
        result["breakout"]=True
        if wedge_type=="rising":
            result["breakout_line"]=((x2,y2),(x4,y4))
        else:
            result["breakout_line"]=((x1,y1),(x3,y3))

    result["found"]=True
    result["wedge_type"]=wedge_type

    if check_retest and brk and result["breakout_line"]:
        close_col=get_col_name("Close",time_frame)
        (ixA,pxA),(ixB,pxB)=result["breakout_line"]
        mW,bW=line_equation(ixA,pxA,ixB,pxB)
        if mW is not None:
            retest_done=False
            retest_bar=None
            for i in range(ixB+1,df_len):
                c=df[close_col].iloc[i]
                line_y=mW*i+bW
                diff_perc=abs(c-line_y)/(abs(line_y)+1e-9)
                if diff_perc<=retest_tolerance:
                    retest_done=True
                    retest_bar=i
                    break
            result["retest_info"]={
                "retest_done":retest_done,
                "retest_bar":retest_bar
            }

    return result


def detect_triangle_advanced(
    wave,
    df: pd.DataFrame=None,
    time_frame: str="1m",
    triangle_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01,
    triangle_types: list=None
) -> dict:
    """
    Advanced Triangle detection (ascending, descending, symmetrical)
    with optional breakout & retest checks.
    """
    result={
      "found": False,
      "triangle_type": None,
      "breakout": False,
      "breakout_line": None,
      "retest_info": None,
      "msgs": []
    }
    if triangle_types is None:
        triangle_types=["ascending","descending","symmetrical"]
    if len(wave)<4:
        result["msgs"].append("Not enough pivot for triangle (need >=4).")
        return result

    last4=wave[-4:]
    p1,p2,p3,p4=last4
    t_list=[p[2]for p in last4]
    up_zig=[+1,-1,+1,-1]
    down_zig=[-1,+1,-1,+1]
    if t_list not in[up_zig,down_zig]:
        result["msgs"].append("Zigzag pattern not matching triangle requirement.")
        return result

    if t_list==up_zig:
        x1,y1=p1[0],p1[1]
        x3,y3=p3[0],p3[1]
        x2,y2=p2[0],p2[1]
        x4,y4=p4[0],p4[1]
    else:
        x1,y1=p2[0],p2[1]
        x3,y3=p4[0],p4[1]
        x2,y2=p1[0],p1[1]
        x4,y4=p3[0],p3[1]

    m_top,b_top=line_equation(x1,y1,x3,y3)
    m_bot,b_bot=line_equation(x2,y2,x4,y4)
    if m_top is None or m_bot is None:
        result["msgs"].append("Line top/bot eq fail => vertical slope.")
        return result

    def is_flat(m):
        return(abs(m)<triangle_tolerance)

    top_type=None
    bot_type=None
    if is_flat(m_top):
        top_type="flat"
    elif m_top>0:
        top_type="rising"
    else:
        top_type="falling"

    if is_flat(m_bot):
        bot_type="flat"
    elif m_bot>0:
        bot_type="rising"
    else:
        bot_type="falling"

    tri_type=None
    if top_type=="flat"and bot_type=="rising"and("ascending"in triangle_types):
        tri_type="ascending"
    elif top_type=="falling"and bot_type=="flat"and("descending"in triangle_types):
        tri_type="descending"
    elif top_type=="falling"and bot_type=="rising"and("symmetrical"in triangle_types):
        tri_type="symmetrical"

    if not tri_type:
        result["msgs"].append("No matching triangle type.")
        return result

    df_len=len(df) if df is not None else 0
    brk=False
    if check_breakout and df is not None and df_len>0:
        close_col=get_col_name("Close",time_frame)
        if close_col in df.columns:
            last_close=df[close_col].iloc[-1]
            last_i=df_len-1
            line_y_top=m_top*last_i+b_top
            line_y_bot=m_bot*last_i+b_bot
            if tri_type=="ascending":
                if last_close>line_y_top:
                    brk=True
            elif tri_type=="descending":
                if last_close<line_y_bot:
                    brk=True
            else:
                if(last_close>line_y_top)or(last_close<line_y_bot):
                    brk=True

    if brk:
        result["breakout"]=True
        if tri_type=="ascending":
            result["breakout_line"]=((x1,y1),(x3,y3))
        elif tri_type=="descending":
            result["breakout_line"]=((x2,y2),(x4,y4))
        else:
            result["breakout_line"]=((x1,y1),(x3,y3))

    result["found"]=True
    result["triangle_type"]=tri_type

    if check_retest and brk and result["breakout_line"]is not None:
        close_col=get_col_name("Close",time_frame)
        xA,pA=result["breakout_line"][0]
        xB,pB=result["breakout_line"][1]
        m_,b_=line_equation(xA,pA,xB,pB)
        if m_ is not None:
            retest_done=False
            retest_bar=None
            for i in range(xB+1,df_len):
                c=df[close_col].iloc[i]
                line_y=m_*i+b_
                diff_perc=abs(c-line_y)/(abs(line_y)+1e-9)
                if diff_perc<=retest_tolerance:
                    retest_done=True
                    retest_bar=i
                    break
            result["retest_info"]={
                "retest_done":retest_done,
                "retest_bar":retest_bar
            }

    return result


##############################################################################
# 7) CUP & HANDLE (Advanced)
##############################################################################
def detect_cup_and_handle_advanced(df: pd.DataFrame,
                                   time_frame: str = "1m",
                                   pivots=None,
                                   tolerance: float = 0.02,
                                   volume_drop_check: bool = True,
                                   volume_drop_ratio: float = 0.2,
                                   cup_min_bars: int = 20,
                                   cup_max_bars: int = 300,
                                   handle_ratio: float = 0.3,
                                   handle_max_bars: int = 50,
                                   close_above_rim: bool = True,
                                   check_retest: bool = False,
                                   retest_tolerance: float = 0.01
                                  ) -> dict:
    """
    Advanced Cup&Handle detection with optional volume checks and retest detection.
    """
    result = {
        "found": False,
        "cup_left_top": None,
        "cup_bottom": None,
        "cup_right_top": None,
        "cup_bars": 0,
        "cup_volume_drop": None,
        "handle_found": False,
        "handle_top": None,
        "handle_bars": 0,
        "confirmed": False,
        "rim_line": None,
        "msgs": [],
        "retest_info": None
    }

    close_col  = f"Close_{time_frame}"
    volume_col = f"Volume_{time_frame}"
    if close_col not in df.columns:
        result["msgs"].append(f"Missing col: {close_col}")
        return result

    if pivots is None:
        # Could run PivotScanner or a custom pivot approach
        pass

    top_pivots = [p for p in pivots if p[2]==+1]
    bot_pivots = [p for p in pivots if p[2]==-1]
    if len(top_pivots)<2 or len(bot_pivots)<1:
        result["msgs"].append("Not enough top/dip pivots for Cup&Handle.")
        return result

    # Find a 'cup' => left_top -> bottom -> right_top
    sorted_p = sorted(pivots, key=lambda x: x[0])
    best_cup = None
    for i in range(1, len(sorted_p)-1):
        if sorted_p[i][2]== -1:  # dip
            idxDip, pxDip = sorted_p[i][0], sorted_p[i][1]
            left_candidates  = [tp for tp in sorted_p[:i]   if tp[2]==+1]
            right_candidates = [tp for tp in sorted_p[i+1:] if tp[2]==+1]
            if (not left_candidates) or (not right_candidates):
                continue
            left_top  = left_candidates[-1]
            right_top = right_candidates[0]
            bars_cup = right_top[0] - left_top[0]
            if bars_cup< cup_min_bars or bars_cup> cup_max_bars:
                continue

            avg_top = (left_top[1] + right_top[1]) / 2
            top_diff = abs(left_top[1] - right_top[1]) / (avg_top+1e-9)
            if top_diff> tolerance:
                continue
            if pxDip> avg_top:
                continue

            best_cup = (left_top, (idxDip, pxDip), right_top, bars_cup)
            break

    if not best_cup:
        result["msgs"].append("No valid cup found.")
        return result

    l_top, cup_dip, r_top, cup_bars = best_cup
    result["found"] = True
    result["cup_left_top"] = l_top
    result["cup_bottom"]   = cup_dip
    result["cup_right_top"]= r_top
    result["cup_bars"]     = cup_bars

    # Volume drop check
    if volume_drop_check and (volume_col in df.columns):
        idxL, pxL = l_top[0], l_top[1]
        idxR, pxR = r_top[0], r_top[1]
        cup_vol_series= df[volume_col].iloc[idxL : idxR+1]
        if len(cup_vol_series)>5:
            start_vol = cup_vol_series.iloc[0]
            min_vol   = cup_vol_series.min()
            drop_percent = (start_vol - min_vol)/(start_vol+1e-9)
            result["cup_volume_drop"]= drop_percent
            if drop_percent< volume_drop_ratio:
                result["msgs"].append(
                    f"Cup volume drop {drop_percent:.2f} < target {volume_drop_ratio:.2f}"
                )

    # Handle => small correction after the cup forms
    rim_idxL, rim_pxL= l_top[0], l_top[1]
    rim_idxR, rim_pxR= r_top[0], r_top[1]
    if rim_idxR<= rim_idxL:
        return result

    slope_rim= (rim_pxR- rim_pxL)/(rim_idxR- rim_idxL+1e-9)
    intercept= rim_pxL - slope_rim* rim_idxL

    dip_price= cup_dip[1]
    cup_height= ((l_top[1] + r_top[1])/2) - dip_price
    if cup_height<=0:
        return result

    handle_start= rim_idxR
    handle_end  = min(rim_idxR + handle_max_bars, len(df)-1)
    handle_found= False
    handle_top  = None
    handle_bars = 0

    if handle_start< handle_end:
        seg = df[close_col].iloc[handle_start : handle_end+1]
        loc_max_val= seg.max()
        loc_max_idx= seg.idxmax()
        handle_bars = handle_end - handle_start
        handle_depth= ((r_top[1] + l_top[1])/2) - loc_max_val
        if handle_depth>0:
            ratio= handle_depth / cup_height
            if ratio <= handle_ratio:
                handle_found= True
                handle_top= (loc_max_idx, loc_max_val)

    result["handle_found"]= handle_found
    result["handle_top"]= handle_top
    result["handle_bars"]= handle_bars

    # Confirmation => breakout above the rim line
    close_vals= df[close_col]
    last_i = len(df)-1
    last_price= close_vals.iloc[-1]
    rim_line_val= slope_rim * last_i + intercept
    if close_above_rim:
        if last_price> rim_line_val:
            result["confirmed"]= True
            result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))
    else:
        high_col= f"High_{time_frame}"
        if high_col in df.columns:
            last_high= df[high_col].iloc[-1]
            if last_high> rim_line_val:
                result["confirmed"]= True
                result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))

    # Retest => if confirmed
    if check_retest and result["confirmed"]:
        retest_info = _check_retest_cup_handle(
            df, time_frame,
            rim_line=((rim_idxL, rim_pxL), (rim_idxR, rim_pxR)),
            break_bar= last_i,
            tolerance= retest_tolerance
        )
        result["retest_info"] = retest_info

    return result


def _check_retest_cup_handle(
    df: pd.DataFrame,
    time_frame: str,
    rim_line: tuple,
    break_bar: int,
    tolerance: float = 0.01
) -> dict:
    """
    Cup&Handle retest of the rim line after breakout.
    """
    if not rim_line or len(rim_line)!=2:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    (xL, pL), (xR, pR) = rim_line
    m, b = line_equation(xL, pL, xR, pR)
    if m is None:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    close_col= f"Close_{time_frame}"
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    for i in range(break_bar+1, len(df)):
        c = df[close_col].iloc[i]
        line_y = m*i + b
        dist_ratio = abs(c - line_y)/(abs(line_y)+1e-9)
        if dist_ratio <= tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "distance_ratio": dist_ratio
            }

    return {"retest_done": False, "retest_bar": None, "distance_ratio": None}


##############################################################################
# 8) FLAG / PENNANT (Advanced)
##############################################################################
def detect_flag_pennant_advanced(
    df: pd.DataFrame,
    time_frame: str="1m",
    pivots=None,
    min_flagpole_bars: int=15,
    impulse_pct: float=0.05,
    max_cons_bars: int=40,
    pivot_channel_tolerance: float=0.02,
    pivot_triangle_tolerance: float=0.02,
    require_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
) -> dict:
    """
    Detects Flag or Pennant after an impulse move (>= impulse_pct in last min_flagpole_bars).
    Then sees if pivot slopes form a mini-channel (flag) or mini-triangle (pennant).
    Optional breakout & retest checks.
    """
    result={
        "found":False,
        "direction":None,
        "pattern_type":None,
        "consolidation_pivots":[],
        "upper_line":None,
        "lower_line":None,
        "confirmed":False,
        "breakout_bar":None,
        "breakout_line":None,
        "retest_info":None,
        "msgs":[]
    }
    close_col=get_col_name("Close",time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"Missing {close_col}")
        return result
    n=len(df)
    if n<min_flagpole_bars:
        result["msgs"].append("Not enough bars for flagpole check.")
        return result

    # Flagpole => last X bars
    start_i=n-min_flagpole_bars
    price_start=df[close_col].iloc[start_i]
    price_end=df[close_col].iloc[-1]
    pct_chg=(price_end-price_start)/(price_start+1e-9)
    if abs(pct_chg)<impulse_pct:
        result["msgs"].append(f"No strong impulse (<{impulse_pct*100}%).")
        return result

    direction="bull" if (pct_chg>0) else "bear"
    result["direction"]=direction

    # Consolidation zone => might be after the main move or overlapping
    cons_start=n-min_flagpole_bars
    cons_end=min(n-1,cons_start+max_cons_bars)
    if cons_end<=cons_start:
        result["msgs"].append("Consolidation not enough bars.")
        return result

    if pivots is None:
        pass
    cons_piv=[p for p in pivots if p[0]>=cons_start and p[0]<=cons_end]
    result["consolidation_pivots"]=cons_piv

    # Need at least 2 top & 2 bottom pivots
    top_pivs=[p for p in cons_piv if p[2]==+1]
    bot_pivs=[p for p in cons_piv if p[2]==-1]
    if len(top_pivs)<2 or len(bot_pivs)<2:
        result["msgs"].append("Not enough top/bottom pivots => can't form mini channel or triangle.")
        return result

    # Take the first two top pivots, first two bottom pivots
    top_sorted=sorted(top_pivs,key=lambda x:x[0])
    bot_sorted=sorted(bot_pivs,key=lambda x:x[0])
    up1,up2=top_sorted[0],top_sorted[1]
    dn1,dn2=bot_sorted[0],bot_sorted[1]

    def slope(x1,y1,x2,y2):
        if(x2-x1)==0:return None
        return(y2-y1)/(x2-x1)
    s_up=slope(up1[0],up1[1],up2[0],up2[1])
    s_dn=slope(dn1[0],dn1[1],dn2[0],dn2[1])
    if(s_up is None)or(s_dn is None):
        result["msgs"].append("Channel lines vertical => cannot form slope.")
        return result

    slope_diff=abs(s_up-s_dn)/(abs(s_up)+1e-9)
    is_parallel=(slope_diff<pivot_channel_tolerance)
    is_opposite_sign=(s_up*s_dn<0)

    upper_line=((up1[0],up1[1]),(up2[0],up2[1]))
    lower_line=((dn1[0],dn1[1]),(dn2[0],dn2[1]))
    result["upper_line"]=upper_line
    result["lower_line"]=lower_line

    pattern_type=None
    if is_parallel:
        pattern_type="flag"
    elif is_opposite_sign and slope_diff>pivot_triangle_tolerance:
        pattern_type="pennant"

    if not pattern_type:
        result["msgs"].append("No definitive mini-flag or mini-pennant from pivot slopes.")
        return result

    result["pattern_type"]=pattern_type
    result["found"]=True

    # Breakout check?
    if not require_breakout:
        return result

    last_i=n-1
    last_close=df[close_col].iloc[-1]
    def line_val(p1,p2,x):
        if(p2[0]-p1[0])==0:
            return p1[1]
        m=(p2[1]-p1[1])/(p2[0]-p1[0])
        b=p1[1]-m*p1[0]
        return m*x+b

    up_line_last=line_val(up1,up2,last_i)
    dn_line_last=line_val(dn1,dn2,last_i)
    conf=False
    brk_bar=None
    if direction=="bull":
        if last_close>up_line_last:
            conf=True
            brk_bar=last_i
    else:
        if last_close<dn_line_last:
            conf=True
            brk_bar=last_i

    result["confirmed"]=conf
    result["breakout_bar"]=brk_bar
    if conf:
        if direction=="bull":
            result["breakout_line"]=upper_line
        else:
            result["breakout_line"]=lower_line

        if check_retest and result["breakout_line"]:
            (ixA,pxA),(ixB,pxB)=result["breakout_line"]
            mF,bF=line_equation(ixA,pxA,ixB,pxB)
            if mF is not None:
                retest_done=False
                retest_bar=None
                for i in range(ixB+1,n):
                    c=df[close_col].iloc[i]
                    line_y=mF*i+bF
                    diff_perc=abs(c-line_y)/(abs(line_y)+1e-9)
                    if diff_perc<=retest_tolerance:
                        retest_done=True
                        retest_bar=i
                        break
                result["retest_info"]={
                    "retest_done":retest_done,
                    "retest_bar":retest_bar
                }

    return result


##############################################################################
# 9) CHANNEL (Advanced)
##############################################################################
def detect_channel_advanced(
    df: pd.DataFrame,
    time_frame: str="1m",
    pivots=None,
    parallel_thresh: float=0.02,
    min_top_pivots: int=3,
    min_bot_pivots: int=3,
    max_iter: int=10,
    check_retest: bool=False,
    retest_tolerance: float=0.01
)->dict:
    """
    Regression-based approach: best-fit lines for top and bottom pivots.
    Checks if they are sufficiently parallel => channel.
    Optionally checks for breakout & retest.
    """
    import numpy as np
    result={
        "found":False,
        "channel_type":None,
        "upper_line_points":[],
        "lower_line_points":[],
        "upper_line_eq":None,
        "lower_line_eq":None,
        "breakout":False,
        "breakout_line":None,
        "retest_info":None,
        "msgs":[]
    }
    close_col=get_col_name("Close",time_frame)
    if close_col not in df.columns:
        result["msgs"].append("No close col found.")
        return result
    if pivots is None or len(pivots)==0:
        result["msgs"].append("No pivots given.")
        return result

    top_piv=[p for p in pivots if p[2]==+1]
    bot_piv=[p for p in pivots if p[2]==-1]
    if len(top_piv)<min_top_pivots or len(bot_piv)<min_bot_pivots:
        result["msgs"].append("Not enough top/bottom pivots.")
        return result

    def best_fit_line(pivot_list):
        xs=np.array([p[0] for p in pivot_list],dtype=float)
        ys=np.array([p[1] for p in pivot_list],dtype=float)
        if len(xs)<2:
            return(0.0,float(ys.mean()))
        m=(np.mean(xs*ys)-np.mean(xs)*np.mean(ys))/ \
          (np.mean(xs**2)-(np.mean(xs))**2+1e-9)
        b=np.mean(ys)-m*np.mean(xs)
        return(m,b)

    # regression for top & bottom
    m_top,b_top=best_fit_line(top_piv)
    m_bot,b_bot=best_fit_line(bot_piv)
    slope_diff=abs(m_top-m_bot)/(abs(m_top)+1e-9)
    if slope_diff>parallel_thresh:
        msg=f"Slope diff {slope_diff:.3f}>threshold=>not channel."
        result["msgs"].append(msg)
        return result

    result["found"]=True
    result["upper_line_points"]=top_piv
    result["lower_line_points"]=bot_piv
    result["upper_line_eq"]=(m_top,b_top)
    result["lower_line_eq"]=(m_bot,b_bot)

    avg_slope=(m_top+m_bot)/2
    if abs(avg_slope)<0.01:
        result["channel_type"]="horizontal"
    elif avg_slope>0:
        result["channel_type"]="ascending"
    else:
        result["channel_type"]="descending"

    last_i=len(df)-1
    last_close=df[close_col].iloc[-1]
    top_line_val=m_top*last_i+b_top
    bot_line_val=m_bot*last_i+b_bot
    breakout_up=(last_close>top_line_val)
    breakout_down=(last_close<bot_line_val)
    if breakout_up or breakout_down:
        result["breakout"]=True
        def line_points_from_regression(m,b,pivot_list):
            xvals=[p[0] for p in pivot_list]
            x_min,x_max=min(xvals),max(xvals)
            y_min=m*x_min+b
            y_max=m*x_max+b
            return((x_min,y_min),(x_max,y_max))

        if breakout_up:
            line2d=line_points_from_regression(m_top,b_top,top_piv)
            result["breakout_line"]=line2d
        else:
            line2d=line_points_from_regression(m_bot,b_bot,bot_piv)
            result["breakout_line"]=line2d

        if check_retest and result["breakout_line"]is not None:
            (ixA,pxA),(ixB,pxB)=result["breakout_line"]
            mC,bC=line_equation(ixA,pxA,ixB,pxB)
            if mC is not None:
                retest_done=False
                retest_bar=None
                for i in range(ixB+1,len(df)):
                    c=df[close_col].iloc[i]
                    line_y=mC*i+bC
                    diff_perc=abs(c-line_y)/(abs(line_y)+1e-9)
                    if diff_perc<=retest_tolerance:
                        retest_done=True
                        retest_bar=i
                        break
                result["retest_info"]={
                    "retest_done":retest_done,
                    "retest_bar":retest_bar
                }

    return result


##############################################################################
#  GANN Helpers
##############################################################################
def get_planet_angle(date, planet_name="SUN"):
    """
    Placeholder for real astro calculations (pyswisseph, etc.).
    """
    if swisseph is None:
        # fallback
        day = date.day
        return (day* 15)% 360
    else:
        # real ephemeris calculation
        pass
    return 0

def get_astro_angle(date):
    """
    Another placeholder for astro angle with day-based logic.
    """
    day = date.day
    angle = (day * 12) % 360
    return angle

def is_local_min(df: pd.DataFrame, bar_i: int, close_col: str,
                 left_bars: int, right_bars: int) -> bool:
    if bar_i< left_bars or bar_i>(len(df)- right_bars-1):
        return False
    val= df[close_col].iloc[bar_i]
    left_slice= df[close_col].iloc[bar_i-left_bars: bar_i]
    right_slice= df[close_col].iloc[bar_i+1: bar_i+1+ right_bars]
    return (all(val< x for x in left_slice) and
            all(val<= x for x in right_slice))

def is_local_max(df: pd.DataFrame, bar_i: int, close_col: str,
                 left_bars: int, right_bars: int) -> bool:
    if bar_i< left_bars or bar_i>(len(df)- right_bars-1):
        return False
    val= df[close_col].iloc[bar_i]
    left_slice= df[close_col].iloc[bar_i-left_bars: bar_i]
    right_slice= df[close_col].iloc[bar_i+1: bar_i+1+ right_bars]
    return (all(val> x for x in left_slice) and
            all(val>= x for x in right_slice))

def advanced_wheel_of_24_variants(anchor_price: float, variant: str = "typeA",
                                  steps: int=5):
    """
    Example ways to compute 'Wheel of 24' price levels from an anchor price.
    This is largely placeholder / conceptual.
    """
    levels=[]
    if anchor_price<=0:
        return levels

    if variant=="typeA":
        # anchor_price * (1 + n*(24/100))
        for n in range(1, steps+1):
            uv= anchor_price*(1+ n*(24/100))
            dv= anchor_price*(1- n*(24/100)) if n*(24/100)<1 else None
            if uv>0: levels.append(uv)
            if dv and dv>0: levels.append(dv)

    elif variant=="typeB":
        # sqrt-based approach, purely an example
        base= math.sqrt(24)
        anc_sqrt= math.sqrt(anchor_price)
        for n in range(1, steps+1):
            upv= (anc_sqrt+ n*base)**2
            dnv= (anc_sqrt- n*base)**2 if (anc_sqrt> n*base) else None
            if upv>0: levels.append(upv)
            if dnv and dnv>0: levels.append(dnv)

    else:
        # typeC => anchor ± n*15
        for n in range(1, steps+1):
            uv= anchor_price + n*15
            dv= anchor_price - n*15 if (anchor_price> n*15) else None
            if uv>0: levels.append(uv)
            if dv and dv>0: levels.append(dv)

    return sorted(list(set(levels)))


def detect_gann_pattern_ultra_v7(
    df: pd.DataFrame,
    time_frame: str="1m",

    # Anchor/pivot param
    pivot_window: int = 200,
    anchor_count: int = 3,
    pivot_select_mode: str = "extremes_vol",

    # Gann fan param
    angles = None,   
    bars_per_unit: float= 1.0,
    price_per_unit: float= 1.0,
    line_tolerance: float=0.005,
    min_line_respects: int=3,

    # SQ9 param
    sq9_variant="sqrt_plus_360",
    sq9_steps=5,
    sq9_tolerance=0.01,

    # Wheel of 24 param
    w24_variant="typeB",
    w24_steps=5,
    w24_tolerance=0.01,

    # Time cycles
    cycles=None,          
    astro_cycles=None,    
    cycle_pivot_tolerance=2,

    # pivot left/right
    pivot_left_bars=3,
    pivot_right_bars=3,

    # ATR filter
    atr_filter=True,
    atr_period=14,
    atr_factor=0.5,

    # Volume filter
    volume_filter=False,
    volume_ratio=1.3,

    # Additional angles
    additional_angle_shift: float=180.0,

    debug: bool=False,

    # --- New param: retest check
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> dict:
    """
    'Ultra++' Gann detection (v7) with multiple concepts:
      - best anchor pivot
      - gann fan lines
      - square of 9
      - wheel of 24
      - time cycles / astro cycles
    + optional retest detection of the chosen Gann line.
    """
    result= {
        "anchors": [],
        "best_anchor": None,
        "msgs": [],
        "found": False,
        "gann_line": None,
        "retest_info": None,
    }

    close_col= f"Close_{time_frame}"
    high_col= f"High_{time_frame}"
    low_col= f"Low_{time_frame}"
    volume_col= f"Volume_{time_frame}"

    if angles is None:
        angles= [45.0, 22.5, 67.5, 90.0, 135.0, 180.0]

    if additional_angle_shift>0:
        add_list=[]
        for a in angles:
            shifted= a+ additional_angle_shift
            add_list.append(shifted)
        angles= sorted(list(set(angles+ add_list)))

    if cycles is None:
        cycles= [30,90,180]
    if astro_cycles is None:
        astro_cycles= [90,180,360]

    if close_col not in df.columns or len(df)< pivot_window:
        result["msgs"].append("Insufficient data or missing close column.")
        return result

    # Prepare ATR if needed
    atr_col= f"ATR_{time_frame}"
    if atr_filter and (atr_col not in df.columns):
        if (high_col in df.columns) and (low_col in df.columns):
            df["_H-L_"]  = df[high_col]- df[low_col]
            df["_H-PC_"] = (df[high_col]- df[close_col].shift(1)).abs()
            df["_L-PC_"] = (df[low_col] - df[close_col].shift(1)).abs()
            df["_TR_"]   = df[["_H-L_","_H-PC_","_L-PC_"]].max(axis=1)
            df[atr_col]  = df["_TR_"].rolling(atr_period).mean()

    # 1) Choose anchor pivots
    anchor_pivots=[]
    seg= df[close_col].iloc[-pivot_window:]
    smin, smax= seg.min(), seg.max()
    i_min, i_max= seg.idxmin(), seg.idxmax()
    if pivot_select_mode=="extremes_vol":
        anchor_pivots.append( (i_min, smin) )
        anchor_pivots.append( (i_max, smax) )
        if volume_col in df.columns:
            vseg= df[volume_col].iloc[-pivot_window:]
            iv= vseg.idxmax()
            pv= df[close_col].loc[iv]
            anchor_pivots.append( (iv,pv) )
    else:
        anchor_pivots.append( (i_min,smin) )
        anchor_pivots.append( (i_max,smax) )

    anchor_pivots= list(dict.fromkeys(anchor_pivots))  # unique
    if len(anchor_pivots)> anchor_count:
        anchor_pivots= anchor_pivots[: anchor_count]

    def slope_from_angle(ang_deg: float) -> float:
        rad= math.radians(ang_deg)
        raw_slope= math.tan(rad)
        return raw_slope

    def build_fan_lines(anc_idx, anc_val) -> list:
        fan=[]
        for a_deg in angles:
            m= slope_from_angle(a_deg)
            label= f"{a_deg}°"
            fan.append({
                "angle_deg": a_deg,
                "slope": m,
                "label": label,
                "respects": 0,
                "confidence": 0.0,
                "points": []
            })
        return fan

    def check_fan_respects(fan_lines, anchor_idx, anchor_val):
        for b_i in range(len(df)):
            px= df[close_col].iloc[b_i]
            # Optionally skip bars with very small range if ATR filtering
            if atr_filter and (atr_col in df.columns):
                av= df[atr_col].iloc[b_i]
                if not math.isnan(av):
                    rng= df[high_col].iloc[b_i]- df[low_col].iloc[b_i]
                    if rng< (av* atr_factor):
                        continue

            xdiff= b_i- anchor_idx
            for fl in fan_lines:
                line_y= fl["slope"]* xdiff + anchor_val
                dist= abs(px- line_y)/ (abs(line_y)+1e-9)
                if dist< line_tolerance:
                    # Check if local pivot?
                    ptype=None
                    if is_local_min(df,b_i,close_col,pivot_left_bars, pivot_right_bars):
                        ptype="min"
                    elif is_local_max(df,b_i,close_col,pivot_left_bars, pivot_right_bars):
                        ptype="max"

                    fl["respects"]+=1
                    fl["points"].append( (b_i, line_y, px, ptype) )

        for fl in fan_lines:
            c=0.0
            piv_count= sum(1 for p in fl["points"] if p[3] is not None)
            if fl["respects"]>= min_line_respects:
                c= 0.5+ min(0.5, 0.05*(fl["respects"]- min_line_respects))
            c+= piv_count*0.1
            if c>1.0: c=1.0
            fl["confidence"]= round(c,2)

    def compute_sq9_levels(anchor_price: float):
        out=[]
        if anchor_price<=0:
            return out
        base_val= 0
        if sq9_variant in ["sqrt_basic","sqrt_plus_360","default"]:
            base_val= math.sqrt(anchor_price)
        elif sq9_variant=="log_spiral":
            if anchor_price>1:
                base_val= math.log(anchor_price)
            else:
                return out

        for s in range(1, sq9_steps+1):
            if sq9_variant=="sqrt_basic":
                upv= (base_val+ s)**2
                dnv= (base_val- s)**2 if base_val> s else None
            elif sq9_variant=="sqrt_plus_360":
                upv= (base_val+ s + (180/10))**2
                dnv= (base_val- s + (180/10))**2 if base_val> s else None
            elif sq9_variant=="log_spiral":
                upv= math.exp(base_val+ s)
                dnv= math.exp(base_val- s) if base_val> s else None
            else:
                upv= (base_val+ s)**2
                dnv= (base_val- s)**2 if base_val> s else None

            if upv and upv>0: out.append(upv)
            if dnv and dnv>0: out.append(dnv)
        return sorted(list(set(out)))

    def check_levels_respects(level_list, tolerance):
        out=[]
        for lv in level_list:
            res=0
            plist=[]
            for b_i in range(len(df)):
                px_b= df[close_col].iloc[b_i]
                dist= abs(px_b- lv)/ (abs(lv)+1e-9)
                if dist< tolerance:
                    res+=1
                    plist.append((b_i, px_b))
            c= min(1.0, res/10)
            out.append( (lv, res, c, plist) )
        return out

    def build_time_cycles(anchor_idx):
        cyc_data=[]
        # normal cycles
        for cyc in cycles:
            tbar= anchor_idx+ cyc
            cyc_date= df.index[tbar] if (tbar>=0 and tbar<len(df)) else None
            cyc_data.append({
                "bars": cyc,
                "astro": None,
                "target_bar": tbar,
                "approx_date": str(cyc_date) if cyc_date else None,
                "cycle_confidence": 0.0,
                "pivot_detected": None
            })
        # astro
        anchor_date= df.index[anchor_idx] if anchor_idx>=0 and anchor_idx<len(df) else None
        if anchor_date is not None:
            anchor_planet_angle= get_planet_angle(anchor_date)
        else:
            anchor_planet_angle= 0

        for deg in astro_cycles:
            cyc_bar= anchor_idx + deg
            cyc_date= df.index[cyc_bar] if (cyc_bar>=0 and cyc_bar<len(df)) else None
            cyc_data.append({
                "bars": None,
                "astro": deg,
                "target_bar": cyc_bar,
                "approx_date": str(cyc_date) if cyc_date else None,
                "cycle_confidence": 0.0,
                "pivot_detected": None
            })
        return cyc_data

    def check_cycle_pivots(cyc_data):
        for ci in cyc_data:
            tb= ci["target_bar"]
            if tb>=0 and tb<len(df):
                lb= max(0, tb- cycle_pivot_tolerance)
                rb= min(len(df)-1, tb+ cycle_pivot_tolerance)
                piv=None
                for b in range(lb, rb+1):
                    if is_local_min(df,b,close_col,pivot_left_bars,pivot_right_bars):
                        piv= (b, df[close_col].iloc[b], "min")
                        break
                    elif is_local_max(df,b,close_col,pivot_left_bars,pivot_right_bars):
                        piv= (b, df[close_col].iloc[b], "max")
                        break
                if piv:
                    ci["pivot_detected"]= piv
                    ci["cycle_confidence"]=1.0
                else:
                    ci["cycle_confidence"]=0.0
            else:
                ci["cycle_confidence"]=0.0

    def compute_wheel24(anchor_price: float, variant: str="typeB"):
        return advanced_wheel_of_24_variants(anchor_price, variant, steps=w24_steps)

    def build_confluence_points(
        fan_lines, sq9_data, w24_data, cyc_data, anchor_idx, anchor_val
    ):
        conf=[]
        for fl in fan_lines:
            if fl["confidence"]<=0:
                continue
            for (b_i, line_y, px, ptype) in fl["points"]:
                sq9_match= None
                for (lvl,rescount,cc,pl) in sq9_data:
                    dist= abs(px- lvl)/ (abs(lvl)+1e-9)
                    if dist< (sq9_tolerance*2):
                        sq9_match= lvl
                        break
                w24_match= None
                for (wl,resc,cc,pl) in w24_data:
                    dist= abs(px- wl)/ (abs(wl)+1e-9)
                    if dist< (w24_tolerance*2):
                        w24_match= wl
                        break
                cyc_found= None
                for ci in cyc_data:
                    if ci["cycle_confidence"]>0 and ci["pivot_detected"]:
                        if abs(ci["pivot_detected"][0]- b_i)<= cycle_pivot_tolerance:
                            cyc_found= ci
                            break
                if (sq9_match or w24_match or cyc_found):
                    cboost= fl["confidence"]
                    if sq9_match: cboost+= 0.3
                    if w24_match: cboost+= 0.2
                    if cyc_found: cboost+= 0.4
                    if cboost>2.0: cboost=2.0
                    conf.append({
                        "bar_index": b_i,
                        "price": px,
                        "fan_line_label": fl["label"],
                        "ptype": ptype,
                        "sq9_level": sq9_match,
                        "w24_level": w24_match,
                        "cycle_bar": cyc_found["target_bar"] if cyc_found else None,
                        "confidence_boost": round(cboost,2)
                    })
        return conf

    anchor_list=[]
    for (a_idx,a_val) in anchor_pivots:
        item={
            "anchor_idx": a_idx,
            "anchor_price": a_val,
            "fan_lines": [],
            "sq9_levels": [],
            "wheel24_levels": [],
            "time_cycles": [],
            "confluence_points": [],
            "score": 0.0
        }
        fl= build_fan_lines(a_idx,a_val)
        check_fan_respects(fl,a_idx,a_val)
        item["fan_lines"]= fl

        sq9_lvls= compute_sq9_levels(a_val)
        sq9_data= check_levels_respects(sq9_lvls, sq9_tolerance)
        item["sq9_levels"]= sq9_data

        w24_lvls= compute_wheel24(a_val, variant=w24_variant)
        w24_data= check_levels_respects(w24_lvls, w24_tolerance)
        item["wheel24_levels"]= w24_data

        cyc_data= build_time_cycles(a_idx)
        check_cycle_pivots(cyc_data)
        item["time_cycles"]= cyc_data

        conf_pts= build_confluence_points(fl, sq9_data, w24_data, cyc_data, a_idx,a_val)
        item["confluence_points"]= conf_pts

        best_fan_conf= max([f["confidence"] for f in fl]) if fl else 0
        ccount= len(conf_pts)
        item["score"]= round(best_fan_conf + ccount*0.2,2)

        anchor_list.append(item)

    if anchor_list:
        best_anch= max(anchor_list, key=lambda x: x["score"])
        result["best_anchor"]= best_anch
        result["anchors"]= anchor_list
        if best_anch["score"]> 0:
            result["found"] = True

            # pick best fan line
            best_fan_line = None
            best_c = -999
            for fan_line in best_anch["fan_lines"]:
                if fan_line["confidence"]> best_c:
                    best_c = fan_line["confidence"]
                    best_fan_line= fan_line

            if best_fan_line and best_fan_line["respects"]>=3:
                anchor_idx= best_anch["anchor_idx"]
                anchor_price= best_anch["anchor_price"]
                m= best_fan_line["slope"]
                x2= anchor_idx+ 100
                y2= anchor_price+ m*100
                result["gann_line"]= ((anchor_idx, anchor_price), (x2, y2))
    else:
        result["anchors"]= []
        result["best_anchor"]= None

    # Retest check on the chosen Gann line
    if check_retest and result["found"] and result["gann_line"]:
        (ixA,pxA),(ixB,pxB)=result["gann_line"]
        m_, b_= line_equation(ixA, pxA, ixB, pxB)
        if m_ is not None:
            close_col= f"Close_{time_frame}"
            if close_col in df.columns:
                retest_done=False
                retest_bar=None
                start_bar= int(max(ixA, ixB))
                for i in range(start_bar, len(df)):
                    c= df[close_col].iloc[i]
                    line_y= m_*i + b_
                    dist_perc= abs(c- line_y)/(abs(line_y)+1e-9)
                    if dist_perc<= retest_tolerance:
                        retest_done=True
                        retest_bar=i
                        break
                result["retest_info"]={
                    "retest_done": retest_done,
                    "retest_bar": retest_bar
                }

    return result


##############################################################################
# 8) TÜM PATTERNLERİ TEK FONKSİYONLA ÇAĞIRMA (v2)
##############################################################################
def detect_all_patterns_v2(
    pivots, 
    wave, 
    df: pd.DataFrame = None, 
    time_frame: str = "1m", 
    config: dict = None
) -> dict:
    """
    Master function to run all pattern detections in a single call.
    
    - pivots: local pivot list (index, price, +1 or -1)
    - wave:   can be the same pivot list or a specialized wave pivot list
    - df:     main DataFrame
    - config: a dict with sub-keys for each pattern's parameters
    
    Returns a dict of results for each pattern type.
    """
    if config is None:
        config= {}

    # 1) Head & Shoulders
    hs_cfg = config.get("headshoulders", {})
    hs_result = detect_head_and_shoulders_advanced(pivots,df=df, 
                                                   time_frame=time_frame,
                                                   **hs_cfg)

    # 2) Inverse Head & Shoulders
    inv_hs_cfg = config.get("inverse_headshoulders", {})
    inv_hs_result = detect_inverse_head_and_shoulders_advanced(pivots,df=df, 
                                                               time_frame=time_frame,
                                                               **inv_hs_cfg)

    # 3) Double/Triple Top-Bottom
    dt_cfg = config.get("doubletriple", {})
    dtops = detect_double_top(pivots,
                              df, 
                              time_frame,
                              tolerance= dt_cfg.get("tolerance", 0.01),
                              min_distance_bars= dt_cfg.get("min_distance_bars", 20),
                              triple_variation= dt_cfg.get("triple_variation", True),
                              volume_check= dt_cfg.get("volume_check", False),
                              neckline_break= dt_cfg.get("neckline_break", False),
                              check_retest= dt_cfg.get("check_retest", False),
                              retest_tolerance= dt_cfg.get("retest_tolerance", 0.01)
                             )

    dbots = detect_double_bottom(pivots,
                                 df, 
                                 time_frame,
                                 tolerance= dt_cfg.get("tolerance", 0.01),
                                 min_distance_bars= dt_cfg.get("min_distance_bars", 20),
                                 triple_variation= dt_cfg.get("triple_variation", True),
                                 volume_check= dt_cfg.get("volume_check", False),
                                 neckline_break= dt_cfg.get("neckline_break", False),
                                 check_retest= dt_cfg.get("check_retest", False),
                                 retest_tolerance= dt_cfg.get("retest_tolerance", 0.01)
                                )
 # 2.1) Triple Top
    triple_top_cfg = config.get("triple_top", {})
    triple_top_results = detect_triple_top_advanced(
        pivots,
        df,
        time_frame,
        **triple_top_cfg
    )

    # 2.2) Triple Bottom
    triple_bottom_cfg = config.get("triple_bottom", {})
    triple_bottom_results = detect_triple_bottom_advanced(
        pivots,
        df,
        time_frame,
        **triple_bottom_cfg
    )
    # 4) Elliott Wave
    elliott_cfg = config.get("elliott",{})
    ell_result  = detect_elliott_5wave_advanced(wave= wave, 
                                                df=df, 
                                                time_frame=time_frame,
                                                **elliott_cfg)

    # 5) Wolfe Wave
    wolfe_cfg = config.get("wolfe",{})
    wolfe_result = detect_wolfe_wave_advanced(wave, 
                                              df, 
                                              time_frame,
                                              **wolfe_cfg)

    # 6) Harmonic
    harmonic_cfg = config.get("harmonic",{})
    harm_result = detect_harmonic_pattern_advanced(
        wave, 
        df, 
        time_frame,
        fib_tolerance= harmonic_cfg.get("fib_tolerance", 0.02),
        patterns= harmonic_cfg.get("patterns", ["gartley","bat","crab","butterfly","shark","cipher"]),
        check_volume= harmonic_cfg.get("check_volume", False),
        volume_factor= harmonic_cfg.get("volume_factor", 1.3),
        check_retest= harmonic_cfg.get("check_retest", False),
        retest_tolerance= harmonic_cfg.get("retest_tolerance", 0.01)
    )

    # 7) Triangle
    tri_cfg= config.get("triangle_wedge",{})
    tri_result= detect_triangle_advanced(
        wave,
        df,
        time_frame,
        triangle_tolerance= tri_cfg.get("triangle_tolerance", 0.02),
        check_breakout= tri_cfg.get("check_breakout", True),
        check_retest= tri_cfg.get("check_retest", False),
        retest_tolerance= tri_cfg.get("retest_tolerance", 0.01),
        triangle_types= tri_cfg.get("triangle_types", ["ascending","descending","symmetrical"])
    )

    # 8) Wedge
    wedge_cfg= config.get("wedge_params",{})
    wedge_result= detect_wedge_advanced(
        wave,
        df,
        time_frame,
        wedge_tolerance= wedge_cfg.get("wedge_tolerance", 0.02),
        check_breakout= wedge_cfg.get("check_breakout", True),
        check_retest= wedge_cfg.get("check_retest", False),
        retest_tolerance= wedge_cfg.get("retest_tolerance", 0.01)
    )

    # 9) Cup & Handle
    cup_cfg= config.get("cuphandle",{})
    cup_result= detect_cup_and_handle_advanced(
        df= df,
        time_frame= time_frame,
        pivots= pivots,
        **cup_cfg
    )

    # 10) Flag/Pennant
    flag_cfg= config.get("flagpennant",{})
    flag_result= detect_flag_pennant_advanced(
        df= df,
        time_frame= time_frame,
        pivots= pivots,
        **flag_cfg
    )

    # 11) Channel
    channel_cfg= config.get("channel",{})
    channel_result= detect_channel_advanced(
        df= df,
        time_frame= time_frame,
        pivots= pivots,
        parallel_thresh= channel_cfg.get("tolerance", 0.02),
        min_top_pivots= channel_cfg.get("min_top_pivots", 3),
        min_bot_pivots= channel_cfg.get("min_bot_pivots", 3),
        max_iter= channel_cfg.get("max_iter", 10),
        check_retest= channel_cfg.get("check_retest", False),
        retest_tolerance= channel_cfg.get("retest_tolerance", 0.01)
    )

    # 12) GANN
    gann_cfg= config.get("gann",{})
    use_ultra= gann_cfg.get("use_ultra", False)
    if use_ultra:
        gann_result= detect_gann_pattern_ultra_v7(
            df, 
            time_frame, 
            **gann_cfg
        )
    else:
        gann_result= {
            "found": False,
            "msgs": ["Gann pattern detection minimal used."],
            "gann_line": None
        }

    return {
        "headshoulders": hs_result,
        "inverse_headshoulders": inv_hs_result,
        "double_top": dtops,
        "double_bottom": dbots,
        "elliott": ell_result,
        "wolfe": wolfe_result,
        "harmonic": harm_result,
        "triangle": tri_result,
        "wedge": wedge_result,
        "cup_handle": cup_result,
        "flag_pennant": flag_result,
        "channel": channel_result,
        "gann": gann_result,
        "triple_top_advanced": triple_top_results,
        "triple_bottom_advanced": triple_bottom_results,
    }


##############################
# 9) RUN PARALLEL SCANS
##############################
def run_parallel_scans(symbols, time_frames, df_map: dict, config: dict):
    """
    Example: parallel scan for multiple symbols & time frames.
    `df_map` => { (symbol, time_frame): DataFrame }
    `config` => global or pattern-specific config.
    """
    results= {}

    def process(sym, tf):
        df= df_map.get((sym,tf), None)
        if df is None:
            return (sym, tf, None)

        # Example: find pivots
        sc= PivotScanner(
            df, tf,
            left_bars= config["system_params"]["pivot_left_bars"],
            right_bars=config["system_params"]["pivot_right_bars"],
            volume_factor=1.2,
            atr_factor=0.0
        )
        pivots= sc.find_pivots()
        wave= pivots  # or a different subset for wave-based patterns

        # Now detect patterns
        pattern_cfg= config["pattern_config"]
        patterns= detect_all_patterns_v2(pivots, wave, df, tf, pattern_cfg)
        return (sym, tf, patterns)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map= {}
        for s in symbols:
            for tf in time_frames:
                f= executor.submit(process, s, tf)
                future_map[f]= (s, tf)

        for f in concurrent.futures.as_completed(future_map):
            s,tf= future_map[f]
            try:
                r= f.result()
                results[(s,tf)]= r[2]
            except Exception as e:
                results[(s,tf)]={"error":str(e)}

    return results
