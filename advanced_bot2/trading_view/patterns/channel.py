import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

############################
# Gelişmiş Kanal Tespiti
############################

def detect_channel_advanced(
    df: pd.DataFrame,
    pivots: List[Tuple[int, float, int]],
    time_frame: str = "1h",
    parallel_thresh: float = 0.015,
    min_top_pivots: int = 4,
    min_bot_pivots: int = 4,
    max_iter: int = 20,
    check_retest: bool = True,
    retest_tolerance: float = 0.008,
    volume_confirmation: bool = True,
    debug: bool = False
) -> Dict:
    
    result = {
        "pattern": "channel",
        "found": False,
        "channel_type": None,
        "upper_line": None,
        "lower_line": None,
        "breakout": False,
        "breakout_direction": None,
        "volume_confirmed": False,
        "retest_points": [],
        "confidence": 0.0,
        "msgs": []
    }

    # Validasyonlar
    close_col = get_col_name("Close", time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"{close_col} sütunu eksik")
        return result
    
    if not validate_pivots(pivots, min_top_pivots, min_bot_pivots):
        result["msgs"].append("Yetersiz pivot sayısı")
        return result

    # Pivotları ayır
    top_pivots = [p for p in pivots if p[2] == 1]
    bot_pivots = [p for p in pivots if p[2] == -1]
    
    # Robust regresyon ile çizgi hesapla
    upper_line = robust_regression(top_pivots)
    lower_line = robust_regression(bot_pivots)
    
    # Paralellik kontrolü
    if not is_parallel(upper_line, lower_line, parallel_thresh):
        result["msgs"].append("Paralellik şartı sağlanmadı")
        return result
    
    # Kanal tipini belirle
    result.update(identify_channel_type(upper_line, lower_line))
    
    # Breakout analizi
    breakout_info = check_breakout(
        df, 
        upper_line, 
        lower_line, 
        close_col,
        volume_confirmation,
        time_frame  # Eksik parametre eklendi
    )
    result.update(breakout_info)
    
    # # Retest analizi
    # if result["breakout"] and check_retest:
    #     result["retest_points"] = find_retest_points(
    #         df,
    #         result["breakout_line"],
    #         close_col,
    #         retest_tolerance,
    #         time_frame,  # Eksik parametre eklendi
    #         lookback_bars=20
    #     )
    
    # Güven skoru hesapla
    result["confidence"] = calculate_confidence(
        result, 
        len(top_pivots), 
        len(bot_pivots)
    )
    
    # Debug görselleştirme
    if debug:
        plot_channel(df, result, time_frame)
    
    result["found"] = True
    return result

############################
# Yardımcı Fonksiyonlar
############################

def check_breakout(df: pd.DataFrame, upper: Dict, lower: Dict, 
                  close_col: str, volume_check: bool,
                  time_frame: str) -> Dict:  # time_frame parametresi eklendi
    
    last_close = df[close_col].iloc[-1]
    upper_val = upper["slope"]*len(df) + upper["intercept"]
    lower_val = lower["slope"]*len(df) + lower["intercept"]
    
    breakout_info = {
        "breakout": False,
        "breakout_direction": None,
        "breakout_line": None,
        "volume_confirmed": False
    }
    
    if last_close > upper_val:
        breakout_info.update({
            "breakout": True,
            "breakout_direction": "up",
            "breakout_line": upper
        })
    elif last_close < lower_val:
        breakout_info.update({
            "breakout": True,
            "breakout_direction": "down",
            "breakout_line": lower
        })
    
    if breakout_info["breakout"] and volume_check:
        vol_col = get_col_name("Volume", time_frame)  # Artık tanımlı
        if vol_col in df.columns:
            breakout_vol = df[vol_col].iloc[-1]
            avg_vol = df[vol_col].rolling(20).mean().iloc[-1]
            breakout_info["volume_confirmed"] = breakout_vol > avg_vol * 1.5
    
    return breakout_info

def find_retest_points(df: pd.DataFrame, line: Dict, 
                      close_col: str, tolerance: float,
                      time_frame: str,  # Parametre eklendi
                      lookback_bars: int=20) -> List[Dict]:
    
    m, b = line["slope"], line["intercept"]
    retests = []
    
    for i in range(max(0, len(df)-lookback_bars), len(df)):
        price = df[close_col].iloc[i]
        line_price = m*i + b
        diff = abs(price - line_price) / line_price
        
        if diff <= tolerance:
            retests.append({
                "bar": i,
                "price": price,
                "deviation": diff,
                "volume": df[get_col_name("Volume", time_frame)].iloc[i]  # Artık tanımlı
            })
    
    return retests

# Diğer yardımcı fonksiyonlar (robust_regression, is_parallel, identify_channel_type, plot_channel, get_col_name, validate_pivots, calculate_confidence) öncekiyle aynı kalacak
############################
# Yardımcı Fonksiyonlar
############################

def robust_regression(points: List[Tuple[int, float]]) -> Dict:
    """RANSAC algoritması ile gürbüz regresyon"""
    from sklearn.linear_model import RANSACRegressor
    X = np.array([p[0] for p in points]).reshape(-1, 1)
    y = np.array([p[1] for p in points])
    
    model = RANSACRegressor().fit(X, y)
    m = model.estimator_.coef_[0]
    b = model.estimator_.intercept_
    return {"slope": m, "intercept": b, "inliers": model.inlier_mask_}

def is_parallel(line1: Dict, line2: Dict, threshold: float) -> bool:
    """Eğim farkı ile paralellik kontrolü"""
    angle1 = np.degrees(np.arctan(line1["slope"]))
    angle2 = np.degrees(np.arctan(line2["slope"]))
    return abs(angle1 - angle2) < threshold

def identify_channel_type(upper: Dict, lower: Dict) -> Dict:
    """Kanal tipini belirleme"""
    avg_slope = (upper["slope"] + lower["slope"]) / 2
    if abs(avg_slope) < 0.005:
        return {"channel_type": "horizontal"}
    return {"channel_type": "ascending" if avg_slope > 0 else "descending"}


def find_retest_points(df: pd.DataFrame, line: Dict, 
                      close_col: str, tolerance: float,
                      lookback_bars: int=20) -> List[Dict]:
    """Çoklu retest noktalarını bul"""
    m, b = line["slope"], line["intercept"]
    retests = []
    
    for i in range(max(0, len(df)-lookback_bars), len(df)):
        price = df[close_col].iloc[i]
        line_price = m*i + b
        diff = abs(price - line_price) / line_price
        
        if diff <= tolerance:
            retests.append({
                "bar": i,
                "price": price,
                "deviation": diff,
                "volume": df[get_col_name("Volume", time_frame)].iloc[i]
            })
    
    return retests

def plot_channel(df: pd.DataFrame, result: Dict, time_frame: str):
    """Interactive matplotlib plot"""
    plt.figure(figsize=(16,8))
    
    # Fiyat çizgisi
    close_col = get_col_name("Close", time_frame)
    plt.plot(df[close_col], label='Fiyat', alpha=0.5)
    
    # Kanal çizgileri
    x_vals = np.array([0, len(df)-1])
    
    upper_line = result["upper_line"]
    plt.plot(x_vals, upper_line["slope"]*x_vals + upper_line["intercept"], 
             'r--', label='Üst Kanal')
    
    lower_line = result["lower_line"]
    plt.plot(x_vals, lower_line["slope"]*x_vals + lower_line["intercept"], 
             'g--', label='Alt Kanal')
    
    # Breakout noktası
    if result["breakout"]:
        last_close = df[close_col].iloc[-1]
        plt.scatter(len(df)-1, last_close, c='gold', s=100, 
                    marker='*', label='Breakout')
    
    plt.title(f"{result['channel_type'].capitalize()} Kanalı")
    plt.legend()
    plt.show()

############################
# Yardımcı Araçlar
############################

def get_col_name(base_col: str, time_frame: str) -> str:
    """Dinamik sütun adı oluşturma"""
    return f"{base_col}_{time_frame}"

def validate_pivots(pivots: list, min_tops: int, min_bots: int) -> bool:
    """Pivot kalite kontrolü"""
    if not pivots: return False
    top_count = sum(1 for p in pivots if p[2] == 1)
    bot_count = sum(1 for p in pivots if p[2] == -1)
    return top_count >= min_tops and bot_count >= min_bots

def calculate_confidence(result: Dict, top_count: int, bot_count: int) -> float:
    """Kanal güven skoru [0-1] aralığı"""
    confidence = 0.0
    confidence += min(top_count/10, 0.3)  # Max %30
    confidence += min(bot_count/10, 0.3)
    
    if result["volume_confirmed"]:
        confidence += 0.2
    
    if len(result["retest_points"]) >= 2:
        confidence += 0.2
    
    return min(confidence, 1.0)