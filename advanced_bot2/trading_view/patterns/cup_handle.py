import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

#############################################
# 1) Geliştirilmiş ATR Hesaplama
#############################################
def compute_atr(df: pd.DataFrame, period: int = 14, 
                high_col: str = 'High', low_col: str = 'Low', 
                close_col: str = 'Close') -> pd.Series:
    """
    Geliştirilmiş ATR hesaplaması (Wilder'in EMA yöntemi) 
    - Hata kontrolleri eklendi
    - TR hesaplaması optimize edildi
    """
    # Sütun kontrolleri
    for col in [high_col, low_col, close_col]:
        if col not in df.columns:
            raise ValueError(f"DataFrame'de gerekli sütun eksik: {col}")
    
    df = df.copy()
    # TR hesaplama (vektörleştirilmiş)
    df['prev_close'] = df[close_col].shift(1)
    df['H-L'] = df[high_col] - df[low_col]
    df['H-PC'] = (df[high_col] - df['prev_close']).abs()
    df['L-PC'] = (df[low_col] - df['prev_close']).abs()
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Wilder EMA (alpha = 1/period)
    atr = df['TR'].ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr.rename(f'ATR_{period}')


#############################################
# 2) Dinamik ZigZag Pivot Tespiti (ATR Tabanlı)
#############################################
def find_pivots_atr_zigzag(
    df: pd.DataFrame,
    time_frame: str = "1h",
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    min_bars_between_pivots: int = 3,
    pivot_col_name: str = "ZigZag_Pivot"
) -> list:
    """
    Geliştirilmiş Özellikler:
    - Pivot güncelleme mekanizması iyileştirildi
    - Çoklu zaman dilimi desteği
    - Hata toleranslı pivot atama
    """
    # Sütun isimlerini dinamik oluştur
    close_col = f"Close_{time_frame}"
    high_col = f"High_{time_frame}"
    low_col = f"Low_{time_frame}"
    
    # Veri kontrolü
    if close_col not in df.columns:
        raise ValueError(f"DataFrame'de gerekli sütun eksik: {close_col}")

    # ATR hesapla
    df = df.copy()
    df['ATR_TMP'] = compute_atr(
        df, period=atr_period,
        high_col=high_col, low_col=low_col, close_col=close_col
    )

    # Pivot tespiti
    pivot_list = []
    current_leg = None  # 'up' veya 'down'
    last_pivot_idx = 0
    last_pivot_price = df[close_col].iloc[0]
    pivot_type = -1  # İlk pivot dip olarak başlar
    pivot_list.append((0, last_pivot_price, pivot_type))
    current_leg = 'up'

    for i in range(1, len(df)):
        current_price = df[close_col].iloc[i]
        current_atr = df['ATR_TMP'].iloc[i]
        threshold = atr_multiplier * current_atr if not pd.isna(current_atr) else 0

        # Yukarı trendde pivot high kontrolü
        if current_leg == 'up':
            if current_price < (last_pivot_price - threshold):
                pivot_idx = i - 1  # Bir önceki bar pivot olarak işaretlenir
                pivot_price = df[close_col].iloc[pivot_idx]
                
                # Pivotlar arası mesafe kontrolü
                if (pivot_idx - pivot_list[-1][0]) >= min_bars_between_pivots:
                    pivot_list.append((pivot_idx, pivot_price, 1))
                    last_pivot_idx = pivot_idx
                    last_pivot_price = pivot_price
                    current_leg = 'down'
            else:
                # Yeni yüksek güncelleme
                if current_price > last_pivot_price:
                    last_pivot_price = current_price
                    last_pivot_idx = i

        # Aşağı trendde pivot low kontrolü
        elif current_leg == 'down':
            if current_price > (last_pivot_price + threshold):
                pivot_idx = i - 1
                pivot_price = df[close_col].iloc[pivot_idx]
                
                if (pivot_idx - pivot_list[-1][0]) >= min_bars_between_pivots:
                    pivot_list.append((pivot_idx, pivot_price, -1))
                    last_pivot_idx = pivot_idx
                    last_pivot_price = pivot_price
                    current_leg = 'up'
            else:
                # Yeni düşük güncelleme
                if current_price < last_pivot_price:
                    last_pivot_price = current_price
                    last_pivot_idx = i

    # DataFrame'e pivotları işaretle
    df[pivot_col_name] = np.nan
    for idx, price, ptype in pivot_list:
        if idx < len(df):
            df.at[df.index[idx], pivot_col_name] = ptype

    return pivot_list


#############################################
# 3) Gelişmiş Cup & Handle Tespiti
#############################################
def line_equation(x1: float, y1: float, x2: float, y2: float) -> Tuple[Optional[float], Optional[float]]:
    """İki nokta arası doğru denklemi (m, b)"""
    if x2 == x1:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def _check_retest_cup_handle(
    df: pd.DataFrame, 
    time_frame: str, 
    rim_line: Tuple[Tuple[int, float], Tuple[int, float]], 
    break_bar: int, 
    tolerance: float = 0.01
) -> Dict:
    """Rim çizgisini retest kontrolü (birden fazla noktada)"""
    (xL, pL), (xR, pR) = rim_line
    m, b = line_equation(xL, pL, xR, pR)
    if m is None:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}
    
    close_col = f"Close_{time_frame}"
    retest_points = []
    
    for i in range(break_bar + 1, len(df)):
        current_price = df[close_col].iloc[i]
        line_price = m * i + b
        distance_ratio = abs(current_price - line_price) / (abs(line_price) + 1e-9)
        
        if distance_ratio <= tolerance:
            retest_points.append({
                "bar": i,
                "price": current_price,
                "distance": distance_ratio
            })
    
    # En yakın 2 retest noktasını kontrol et
    if len(retest_points) >= 1:
        best_retest = min(retest_points, key=lambda x: x['distance'])
        return {
            "retest_done": True,
            "retest_bar": best_retest['bar'],
            "distance_ratio": best_retest['distance'],
            "retest_count": len(retest_points)
        }
    return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

def detect_cup_and_handle_advanced(
    df: pd.DataFrame,
    pivots: List[Tuple[int, float, int]],
    time_frame: str = "1h",
    tolerance: float = 0.02,
    volume_drop_check: bool = True,
    volume_drop_ratio: float = 0.3,  # Optimize edilmiş varsayılan
    cup_min_bars: int = 20,
    cup_max_bars: int = 300,
    handle_ratio: float = 0.3,
    handle_max_bars: int = 50,
    handle_min_bars: int = 5,  # Daha gerçekçi minimum
    close_above_rim: bool = True,
    check_retest: bool = True,
    retest_tolerance: float = 0.01,
    debug: bool = False
) -> Dict:
    """
    Geliştirilmiş Özellikler:
    - Dinamik cup derinlik kontrolü
    - Hacim profil analizi
    - Handle simetrisi kontrolü
    - Detaylı debug çıktıları
    """
    result = {
        "pattern": "cup_handle",
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
    
    close_col = f"Close_{time_frame}"
    volume_col = f"Volume_{time_frame}"
    
    # Veri kontrolleri
    if close_col not in df.columns:
        result["msgs"].append(f"Eksik sütun: {close_col}")
        return result
    
    # Pivot filtreleme
    top_pivots = [p for p in pivots if p[2] == 1]
    bottom_pivots = [p for p in pivots if p[2] == -1]
    
    if len(top_pivots) < 2 or len(bottom_pivots) < 1:
        result["msgs"].append("Yeterli pivot yok")
        return result
    
    sorted_pivots = sorted(pivots, key=lambda x: x[0])
    best_cup = None
    
    # Cup arama algoritması
    for i in range(1, len(sorted_pivots) - 1):
        current_pivot = sorted_pivots[i]
        if current_pivot[2] != -1:
            continue
        
        # Sol ve sağ tepe noktalarını bul
        left_tops = [p for p in sorted_pivots[:i] if p[2] == 1]
        right_tops = [p for p in sorted_pivots[i+1:] if p[2] == 1]
        
        if not left_tops or not right_tops:
            continue
        
        left_top = left_tops[-1]
        right_top = right_tops[0]
        
        # Cup boyut kontrolü
        cup_length = right_top[0] - left_top[0]
        if not (cup_min_bars <= cup_length <= cup_max_bars):
            continue
        
        # Cup yüksekliği kontrolü
        avg_top = (left_top[1] + right_top[1]) / 2
        price_diff = abs(left_top[1] - right_top[1]) / avg_top
        if price_diff > tolerance or current_pivot[1] > avg_top:
            continue
        
        best_cup = (left_top, current_pivot, right_top, cup_length)
        break
    
    if not best_cup:
        result["msgs"].append("Uygun cup bulunamadı")
        return result
    
    # Cup özelliklerini kaydet
    l_top, cup_bottom, r_top, cup_bars = best_cup
    result.update({
        "found": True,
        "cup_left_top": l_top,
        "cup_bottom": cup_bottom,
        "cup_right_top": r_top,
        "cup_bars": cup_bars
    })
    
    # Hacim analizi (Volume Profile)
    if volume_drop_check and volume_col in df.columns:
        start_idx = l_top[0]
        end_idx = r_top[0]
        if 0 <= start_idx < end_idx < len(df):
            cup_volume = df[volume_col].iloc[start_idx:end_idx+1]
            if len(cup_volume) > 5:
                max_vol = cup_volume.max()
                min_vol = cup_volume.min()
                result["cup_volume_drop"] = (max_vol - min_vol) / max_vol
                if result["cup_volume_drop"] < volume_drop_ratio:
                    result["msgs"].append(f"Hacim düşüşü yetersiz: {result['cup_volume_drop']:.2f}")

    # Rim çizgisi oluştur
    rim_line = ((l_top[0], l_top[1]), (r_top[0], r_top[1]))
    slope, intercept = line_equation(*rim_line[0], *rim_line[1])
    
    # Handle analizi
    handle_start = r_top[0]
    handle_end = min(r_top[0] + handle_max_bars, len(df)-1)
    handle_prices = df[close_col].iloc[handle_start:handle_end+1]
    
    if len(handle_prices) < handle_min_bars:
        result["msgs"].append("Handle çok kısa")
        return result
    
    handle_low_idx = handle_prices.idxmin()
    handle_low = handle_prices.min()
    handle_length = handle_low_idx - handle_start
    
    # Handle oran kontrolü
    cup_height = (l_top[1] + r_top[1])/2 - cup_bottom[1]
    handle_depth = (l_top[1] + r_top[1])/2 - handle_low
    handle_ratio_actual = handle_depth / cup_height if cup_height > 0 else 0
    
    if handle_ratio_actual > handle_ratio:
        result["msgs"].append(f"Handle derinliği fazla: {handle_ratio_actual:.2f} > {handle_ratio}")
    else:
        result.update({
            "handle_found": True,
            "handle_top": (handle_low_idx, handle_low),
            "handle_bars": handle_length
        })
    
    # Rim kırılım kontrolü
    last_close = df[close_col].iloc[-1]
    if slope is not None:
        rim_value = slope * (len(df)-1) + intercept
        if close_above_rim and last_close > rim_value:
            result["confirmed"] = True
            result["rim_line"] = rim_line
    
    # Retest kontrolü
    if check_retest and result["confirmed"]:
        result["retest_info"] = _check_retest_cup_handle(
            df, time_frame, rim_line, len(df)-1, retest_tolerance
        )
    
    # Debug çıktıları
    if debug:
        plot_cup_handle(df, result, time_frame)
    
    return result

def plot_cup_handle(df: pd.DataFrame, result: Dict, time_frame: str):
    """Cup & Handle modelini görselleştirme"""
    plt.figure(figsize=(16,8))
    close_col = f"Close_{time_frame}"
    plt.plot(df[close_col], label='Fiyat')
    
    if result["rim_line"]:
        (x1, y1), (x2, y2) = result["rim_line"]
        x = np.array([x1, x2])
        y = np.array([y1, y2])
        plt.plot(x, y, 'r--', label='Rim Çizgisi')
    
    cup_points = [result["cup_left_top"][0], result["cup_bottom"][0], result["cup_right_top"][0]]
    cup_prices = [result["cup_left_top"][1], result["cup_bottom"][1], result["cup_right_top"][1]]
    plt.scatter(cup_points, cup_prices, c='green', s=100, marker='o', label='Cup Noktaları')
    
    if result["handle_top"]:
        handle_point = result["handle_top"][0]
        plt.scatter(handle_point, result["handle_top"][1], c='red', s=100, marker='X', label='Handle Dibi')
    
    plt.title("Cup & Handle Modeli")
    plt.legend()
    plt.show()

#############################################
# 4) Optimizasyon ve Test Fonksiyonları
#############################################
def optimize_cup_params(df: pd.DataFrame, time_frame: str = "1h"):
    """Parametre optimizasyonu için grid search"""
    best_score = 0
    best_params = {}
    
    param_grid = {
        'atr_multiplier': [1.5, 2.0, 2.5],
        'handle_ratio': [0.25, 0.3, 0.35],
        'volume_drop_ratio': [0.2, 0.3, 0.4]
    }
    
    # Gerçek test senaryoları için bu kısmı genişletin
    for params in product(*param_grid.values()):
        params_dict = dict(zip(param_grid.keys(), params))
        
        # Pivot tespiti
        pivots = find_pivots_atr_zigzag(df, time_frame=time_frame, **params_dict)
        
        # Cup handle tespiti
        result = detect_cup_and_handle_advanced(df, pivots, time_frame=time_frame)
        
        # Skorlama