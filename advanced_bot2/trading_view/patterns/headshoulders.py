############################
# HEAD & SHOULDERS ADV
############################
import pandas as pd 

def detect_head_and_shoulders_advanced(
    df: pd.DataFrame,
    pivots,
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
    retest_tolerance: float = 0.01
) -> list:
    """
    Gelişmiş Head & Shoulders algılama. 
    returns: list of dict (bulunan formasyonlar)
    """
    high_col   = get_col_name("High",  time_frame)
    low_col    = get_col_name("Low",   time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume",time_frame)
    atr_col    = get_col_name("ATR",   time_frame)

    # ATR Filter
    if atr_filter > 0:
        prepare_atr(df, time_frame)

    top_pivots = [p for p in pivots if p[2] == +1]
    results=[]
    for i in range(len(top_pivots)-2):
        L= top_pivots[i]
        H= top_pivots[i+1]
        R= top_pivots[i+2]
        idxL, priceL,_= L
        idxH, priceH,_= H
        idxR, priceR,_= R

        # Sıra kontrolü
        if not (idxL < idxH < idxR):
            continue
        # Head en büyük olmalı
        if not (priceH > priceL and priceH > priceR):
            continue

        bars_LH= idxH - idxL
        bars_HR= idxR - idxH
        if bars_LH < min_distance_bars or bars_HR < min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue
#
        # Omuz yükseklik farkı
        diffShoulder = abs(priceL - priceR)/(priceH+ 1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        # Volume check
        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            if volH > (mean_shoulder_vol*0.8):
                vol_check= False

        # Boyun çizgisi => L-H ve H-R arasındaki min low
        segment_LH= df[low_col].iloc[idxL: idxH+1]
        segment_HR= df[low_col].iloc[idxH: idxR+1]
        if len(segment_LH)<1 or len(segment_HR)<1:
            continue
        dip1_idx= segment_LH.idxmin()
        dip2_idx= segment_HR.idxmin()
        dip1_val= df[low_col].iloc[dip1_idx]
        dip2_val= df[low_col].iloc[dip2_idx]

        confirmed= False
        confirmed_bar= None
        if neckline_break:
            if dip1_idx != dip2_idx:
                m_, b_= line_equation(dip1_idx, dip1_val, dip2_idx, dip2_val)
                if m_ is not None:
                    for test_i in range(idxR, len(df)):
                        c = df[close_col].iloc[test_i]
                        line_y = m_* test_i + b_
                        if c < line_y:
                            confirmed= True
                            confirmed_bar = test_i
                            break

        # RSI-MACD onayı
        indicator_res = None
        if check_rsi_macd and confirmed and confirmed_bar is not None:
            indicator_res = indicator_checks(df, confirmed_bar, time_frame=time_frame,
                                             rsi_check=True, macd_check=True)

        # Retest
        retest_info = None
        if check_retest and confirmed and (confirmed_bar is not None):
            retest_info = check_retest_levels(
                df, time_frame,
                neckline_points=((dip1_idx,dip1_val),(dip2_idx,dip2_val)),
                break_bar=confirmed_bar,
                tolerance=retest_tolerance
            )
###########################
        # GELİŞTİRİLMİŞ FİLTRELER
        ############################
        # Omuzların zaman asimetrisi kontrolü (sol/sağ oranı)
        time_asymmetry = abs(bars_LH - bars_HR)/max(bars_LH, bars_HR)
        if time_asymmetry > 0.6:  # 60%'dan fazla asimetri reddet
            continue

        # Hacim trendi: Head'de zirve hacim, omuzlarda düşüş
        if volume_decline and volume_col in df.columns:
            vol_ratio = (volH / ((volL + volR)/2 + 1e-9))
            if vol_ratio < 1.2:  # Head hacmi omuzlardan en az %20 yüksek olmalı
                continue

        # Boyun çizgisi dinamik eğim kontrolü
        neckline_slope = (dip2_val - dip1_val)/(dip2_idx - dip1_idx + 1e-9)
        if abs(neckline_slope) > 0.002:  # Çok dik boyun çizgisi reddet
            continue
        results.append({
            "pattern": "head_and_shoulders",
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
                        tolerance: float = 0.02) -> dict:
    """
    Neckline retest kontrolü.
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
        diff_perc = abs(c - line_y)/(abs(line_y)+1e-9)
        if diff_perc <= tolerance:
            retest_done = True
            retest_bar = i
            break
    return {
        "retest_done": retest_done,
        "retest_bar": retest_bar
    }

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def prepare_atr(df: pd.DataFrame, time_frame: str = "1m", period: int = 14):
    """
    ATR hesaplama ve df'ye ekleme
    """
    high_col  = get_col_name("High",  time_frame)
    low_col   = get_col_name("Low",   time_frame)
    close_col = get_col_name("Close", time_frame)
    atr_col   = get_col_name("ATR",   time_frame)

    if atr_col in df.columns:
        return
    df[f"H-L_{time_frame}"] = df[high_col] - df[low_col]
    df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
    df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()

    df[f"TR_{time_frame}"] = df[[f"H-L_{time_frame}",
                                 f"H-PC_{time_frame}",
                                 f"L-PC_{time_frame}"]].max(axis=1)
    df[atr_col] = df[f"TR_{time_frame}"].rolling(period).mean()

def prepare_volume_ma(df: pd.DataFrame, time_frame: str="1m", period: int=20):
    """
    Volume_MA_20_{time_frame} hesaplayıp ekler.
    """
    vol_col = get_col_name("Volume", time_frame)
    ma_col  = f"Volume_MA_{period}_{time_frame}"
    if (vol_col in df.columns) and (ma_col not in df.columns):
        df[ma_col] = df[vol_col].rolling(period).mean()

def line_equation(x1, y1, x2, y2):
    """
    Returns slope (m) and intercept (b) of the line y = m*x + b
    If x2 == x1 => returns (None, None)
    """
    if x2 == x1:
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
    if abs(m1 - m2) < 1e-15:  # parallel
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI hesaplama (basit).
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
    Örnek RSI & MACD kontrol fonksiyonu (örnek).
    """
    res = {
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "verdict": True,
        "msgs": []
    }
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        res["verdict"] = False
        res["msgs"].append("Close column not found, indicator checks skipped.")
        return res

    # RSI
    if rsi_check:
        rsi_col = f"RSI_{time_frame}"
         
        if rsi_col not in df.columns:
            df[rsi_col] = compute_rsi(df[close_col])
            df[rsi_col]
        if idx < len(df):
            rsi_val = df[rsi_col].iloc[idx]
           
            res["rsi"] = rsi_val
            if (not pd.isna(rsi_val)) and (rsi_val < 50):
                res["verdict"]= False
                res["msgs"].append(f"RSI {rsi_val:.2f} <50 => negative.")
        else:
            res["verdict"]= False
            res["msgs"].append("RSI idx out of range")
    #print("-----",time_frame,rsi_check,rsi_val)

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
            if macd_val < macd_sig:
                res["verdict"]=False
                res["msgs"].append(f"MACD < Signal at index={idx}")
        else:
            res["verdict"]=False
            res["msgs"].append("MACD idx out of range")

    return res
