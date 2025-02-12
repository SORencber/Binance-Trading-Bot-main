import pandas as pd
import numpy as np

#############################################
# 1) ATR Hesaplama
#############################################
def compute_atr(df: pd.DataFrame, period: int=14, 
                high_col='High', low_col='Low', close_col='Close') -> pd.Series:
    """
    Basit ATR hesaplaması (Wilder ewm methodu). 
    Dönüş: ATR serisi
    """
    df = df.copy()
    df['H-L'] = df[high_col] - df[low_col]
    df['H-C'] = (df[high_col] - df[close_col].shift(1)).abs()
    df['L-C'] = (df[low_col]  - df[close_col].shift(1)).abs()
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)

    # Wilder ewm
    atr = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    return atr


#############################################
# 2) ATR Bazlı ZigZag Pivotlar
#############################################
def find_pivots_atr_zigzag(df: pd.DataFrame,
                           time_frame: str="1h",
                           atr_period: int=14,
                           atr_multiplier: float=2.0,
                           min_bars_between_pivots: int=3,
                           pivot_col_name: str="ZigZag_Pivot",
                           ) -> list:
    """
    ATR tabanlı dinamik ZigZag pivot tespiti.
    
    Parametreler:
      - atr_period: ATR hesaplamada kullanılacak periyot (örn.14)
      - atr_multiplier: pivot onayı için gereken hareket çarpanı
      - min_bars_between_pivots: iki pivot arasında en az kaç bar geçmeli
      - pivot_col_name: df'e yazılacak kolon ismi (opsiyonel)
    
    Dönüş => pivot_list: [ (bar_index, price, pivot_type), ... ]
               pivot_type => +1 pivot high, -1 pivot low
    """
    df = df.copy()
    close_col = f"Close_{time_frame}"
    high_col  = f"High_{time_frame}"
    low_col   = f"Low_{time_frame}"

    if close_col not in df.columns:
        raise ValueError(f"DataFrame does not have required column: {close_col}")

    # ATR hesapla
    df['ATR_TMP'] = compute_atr(df,
                                period=atr_period,
                                high_col=high_col,
                                low_col=low_col,
                                close_col=close_col)

    # ZigZag
    pivot_list = []
    current_leg = None  # 'up' or 'down'
    last_pivot_idx = 0
    last_pivot_price = df[close_col].iloc[0]

    # İlk pivot type: -1 (low) veya +1 (high)? 
    # Basitçe ilk bar'ı pivot low gibi alacağız, devreye girdikçe güncelleriz
    pivot_type = -1  
    pivot_list.append((0, last_pivot_price, pivot_type))
    current_leg = 'up'  # ilk bar'dan itibaren yukarı hareket arıyoruz

    for i in range(1, len(df)):
        c_price = df[close_col].iloc[i]
        c_atr   = df['ATR_TMP'].iloc[i]
        threshold = atr_multiplier * c_atr if c_atr is not None else 0

        # Yukarı hareket senaryosu => pivot low'dan threshold kadar yükseldi mi?
        if current_leg == 'up':
            # eğer c_price < (last_pivot_price - threshold) => pivot high onayla, leg'i down'a çevir
            if c_price < (last_pivot_price - threshold):
                # pivot high => bir önceki i-1, i-2 barların en yükseği nerede?
                # Basit yaklaşım: i-1 bar en yüksek miydi vs. 
                # Biraz geriye bakabiliriz. Ama en basit: pivot bar i-1 => close
                pivot_index = i-1
                pivot_price = df[close_col].iloc[pivot_index]

                # min_bars kontrol
                if (pivot_index - pivot_list[-1][0]) >= min_bars_between_pivots:
                    pivot_list.append((pivot_index, pivot_price, +1))  # pivot high
                    # yeni pivot referansı
                    last_pivot_idx = pivot_index
                    last_pivot_price = pivot_price
                    # leg'i down'a çevir
                    current_leg = 'down'
            else:
                # new high => last pivotı güncelle?
                if c_price > last_pivot_price:
                    last_pivot_price = c_price
                    last_pivot_idx = i

        # Aşağı hareket senaryosu => pivot high'dan threshold kadar düştü mü?
        elif current_leg == 'down':
            # eğer c_price > (last_pivot_price + threshold) => pivot low onayla, leg'i up'a çevir
            if c_price > (last_pivot_price + threshold):
                pivot_index = i-1
                pivot_price = df[close_col].iloc[pivot_index]
                # min_bars kontrol
                if (pivot_index - pivot_list[-1][0]) >= min_bars_between_pivots:
                    pivot_list.append((pivot_index, pivot_price, -1))  # pivot low
                    last_pivot_idx = pivot_index
                    last_pivot_price = pivot_price
                    current_leg = 'up'
            else:
                # new low => last pivotı güncelle
                if c_price < last_pivot_price:
                    last_pivot_price = c_price
                    last_pivot_idx = i

    # DataFrame'e opsiyonel pivot kolonunu yazmak istersek:
    df[pivot_col_name] = np.nan
    for idx, price, ptype in pivot_list:
        if idx < len(df):
            df.at[df.index[idx], pivot_col_name] = ptype

    return pivot_list


#############################################
# 3) Cup & Handle Gelişmiş Algılama (Daha Önceki)
#############################################
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

def _check_retest_cup_handle(df: pd.DataFrame, time_frame: str, rim_line: tuple,
                             break_bar: int, tolerance: float=0.01):
    (xL, pL), (xR, pR)= rim_line
    m, b = line_equation(xL, pL, xR, pR)
    if m is None:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    close_col = f"Close_{time_frame}"
    for i in range(break_bar+1, len(df)):
        c = df[close_col].iloc[i]
        line_y= m*i + b
        dist_ratio= abs(c - line_y)/(abs(line_y)+1e-9)
        if dist_ratio<= tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "distance_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None, "distance_ratio": None}


def detect_cup_and_handle_advanced(
    df: pd.DataFrame,
    pivots: list,
    time_frame: str="1h",
    tolerance: float=0.02,
    volume_drop_check: bool=True,
    volume_drop_ratio: float=0.2,
    cup_min_bars: int=20,
    cup_max_bars: int=300,
    handle_ratio: float=0.3,
    handle_max_bars: int=50,
    handle_min_bars: int=3,
    close_above_rim: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Pivots listesinden ( [ (bar_index, price, +1/-1), ... ] ) 
    Cup & Handle aranarak result sözlüğü döner.
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
    close_col= f"Close_{time_frame}"
    volume_col= f"Volume_{time_frame}"
    if close_col not in df.columns:
        result["msgs"].append(f"Missing col: {close_col}")
        return result

    top_pivots= [p for p in pivots if p[2] == +1]
    bot_pivots= [p for p in pivots if p[2] == -1]
    if len(top_pivots)<2 or len(bot_pivots)<1:
        result["msgs"].append("Not enough top/dip pivots for Cup&Handle.")
        return result

    sorted_p= sorted(pivots, key=lambda x: x[0])
    best_cup= None

    # Cup arayışı: pivot low ortada, solda pivot high, sağda pivot high
    for i in range(1, len(sorted_p)-1):
        if sorted_p[i][2]== -1:  # dip
            idxDip, pxDip= sorted_p[i][0], sorted_p[i][1]
            # solda pivot high var mı
            left_candidates= [tp for tp in sorted_p[:i] if tp[2]== +1]
            if not left_candidates:
                continue
            left_top= left_candidates[-1]
            # sağda pivot high
            right_candidates= [tp for tp in sorted_p[i+1:] if tp[2]== +1]
            if not right_candidates:
                continue
            right_top= right_candidates[0]

            bars_cup= right_top[0]- left_top[0]
            if bars_cup< cup_min_bars or bars_cup> cup_max_bars:
                continue

            avg_top= (left_top[1]+ right_top[1])/2
            top_diff= abs(left_top[1]- right_top[1])/(avg_top+1e-9)
            if top_diff> tolerance:
                continue
            if pxDip> avg_top:
                continue

            best_cup= (left_top, (idxDip, pxDip), right_top, bars_cup)
            break

    if not best_cup:
        result["msgs"].append("No valid cup found.")
        return result

    l_top, cup_dip, r_top, cup_bars= best_cup
    result["found"]= True
    result["cup_left_top"]= l_top
    result["cup_bottom"]= cup_dip
    result["cup_right_top"]= r_top
    result["cup_bars"]= cup_bars

    # Volume check
    if volume_drop_check and volume_col in df.columns:
        idxL, pxL = l_top[0], l_top[1]
        idxR, pxR = r_top[0], r_top[1]
        if 0 <= idxL < idxR < len(df):
            cup_vol_series= df[volume_col].iloc[idxL : idxR+1]
            if len(cup_vol_series)>5:
                start_vol= cup_vol_series.iloc[0]
                min_vol= cup_vol_series.min()
                drop_percent= (start_vol- min_vol)/(start_vol+1e-9)
                result["cup_volume_drop"]= drop_percent
                if drop_percent< volume_drop_ratio:
                    result["msgs"].append(f"Cup volume drop {drop_percent:.2f} < ratio({volume_drop_ratio:.2f}).")

    # Rim line
    rim_idxL, rim_pxL= l_top[0], l_top[1]
    rim_idxR, rim_pxR= r_top[0], r_top[1]
    slope_rim, intercept= line_equation(rim_idxL, rim_pxL, rim_idxR, rim_pxR)

    dip_price= cup_dip[1]
    cup_height= ((l_top[1] + r_top[1])/2) - dip_price
    if cup_height<=0:
        return result

    # Handle
    handle_found= False
    handle_top= None
    handle_bars= 0
    handle_start= r_top[0]
    handle_end= min(r_top[0]+ handle_max_bars, len(df)-1)
    if handle_start< handle_end:
        seg= df[close_col].iloc[handle_start: handle_end+1]
        loc_min_val= seg.min()
        loc_min_idx= seg.idxmin()
        handle_bars= (loc_min_idx - handle_start)
        handle_depth= ((r_top[1]+ l_top[1])/2) - loc_min_val
        ratio= handle_depth / (cup_height + 1e-9)
        if handle_bars>= handle_min_bars and handle_bars<= handle_max_bars and ratio<= handle_ratio and ratio>0:
            handle_found= True
            handle_top= (loc_min_idx, loc_min_val)

    result["handle_found"]= handle_found
    result["handle_top"]= handle_top
    result["handle_bars"]= handle_bars

    # Onay => rim line break
    confirmed= False
    last_i= len(df)-1
    last_price= df[close_col].iloc[-1]
    if slope_rim is not None:
        rim_line_val= slope_rim* last_i + intercept
        if close_above_rim:
            if last_price> rim_line_val:
                confirmed= True
        else:
            high_col= f"High_{time_frame}"
            if high_col in df.columns:
                last_high= df[high_col].iloc[-1]
                if last_high> rim_line_val:
                    confirmed= True
    result["confirmed"]= confirmed
    if confirmed:
        result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))

    # Retest
    if check_retest and confirmed and result["rim_line"]:
        retest_info= _check_retest_cup_handle(
            df, time_frame,
            rim_line=result["rim_line"],
            break_bar= last_i,
            tolerance=retest_tolerance
        )
        result["retest_info"]= retest_info

    return result


#############################################
# 4) Örnek: Tüm Süreci Birleştiren Fonksiyon
#############################################

def detect_cup_handle_with_atrzigzag(
    df: pd.DataFrame,
    time_frame: str="1h",
    atr_period: int=14,
    atr_multiplier: float=2.0,
    min_bars_between_pivots: int=3,
    cup_args: dict=None
):
    """
    1) ATR tabanlı ZigZag pivot tespiti (find_pivots_atr_zigzag)
    2) Cup&Handle detect (detect_cup_and_handle_with_pivots)
    Parametreler:
      - cup_args => detect_cup_and_handle_with_pivots fonksiyonuna geçecek ek argümanlar
    Dönüş: cup_handle_result => {'pattern':'cup_handle', 'found':..., ...}
    """
    if cup_args is None:
        cup_args = {}  # default

    # 1) Pivots tespit et
    pivots = find_pivots_atr_zigzag(
        df, time_frame=time_frame,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        min_bars_between_pivots=min_bars_between_pivots
    )

    # 2) Cup & Handle
    res = detect_cup_and_handle_with_pivots(
        df=df,
        pivots=pivots,
        time_frame=time_frame,
        **cup_args
    )

    return res


#############################################
# 5) DEMO KULLANIM
#############################################
if __name__ == "__main__":
    # Örnek DataFrame: Kolonlar: [Open_1h, High_1h, Low_1h, Close_1h, Volume_1h]
    data = {
        'Open_1h':   [100,101,102,105,104,103,102,102,103,104,105,106,107,107,106],
        'High_1h':   [101,102,105,106,105,104,103,103,104,105,106,107,108,108,107],
        'Low_1h':    [99,100,101,102,102,102,101,101,102,102,103,105,106,106,105],
        'Close_1h':  [100,101,104,105,104,103,102,102,103,104,106,107,107,107,106],
        'Volume_1h': [500,600,650,800,700,600,550,600,580,590,600,650,700,800,750]
    }
    df_example = pd.DataFrame(data)
    # index bar sayısı kadar
    df_example['time'] = pd.date_range('2023-01-01', periods=len(df_example), freq='H')
    df_example.set_index('time', inplace=True)

    # Cup & Handle Tespiti
    cup_args = dict(
        tolerance=0.02,
        volume_drop_check=True,
        volume_drop_ratio=0.2,
        cup_min_bars=3,
        cup_max_bars=100,
        handle_ratio=0.3,
        handle_max_bars=10,
        handle_min_bars=2,
        close_above_rim=True,
        check_retest=True,
        retest_tolerance=0.01
    )
    res = detect_cup_handle_with_atrzigzag(
        df_example,
        time_frame="1h",
        atr_period=14,
        atr_multiplier=2.0,
        min_bars_between_pivots=3,
        cup_args=cup_args
    )

    print("Cup&Handle Detection =>", res)
    # pivotlar da df_example['ZigZag_Pivot'] kolonunda görebilirsiniz (None / +1 / -1).
import pandas as pd
import numpy as np

#############################################
# 1) ATR Hesaplama
#############################################
def compute_atr(df: pd.DataFrame, period: int=14, 
                high_col='High', low_col='Low', close_col='Close') -> pd.Series:
    """
    Basit ATR hesaplaması (Wilder ewm methodu). 
    Dönüş: ATR serisi
    """
    df = df.copy()
    df['H-L'] = df[high_col] - df[low_col]
    df['H-C'] = (df[high_col] - df[close_col].shift(1)).abs()
    df['L-C'] = (df[low_col]  - df[close_col].shift(1)).abs()
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)

    # Wilder ewm
    atr = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    return atr


#############################################
# 2) ATR Bazlı ZigZag Pivotlar
#############################################
def find_pivots_atr_zigzag(df: pd.DataFrame,
                           time_frame: str="1h",
                           atr_period: int=14,
                           atr_multiplier: float=2.0,
                           min_bars_between_pivots: int=3,
                           pivot_col_name: str="ZigZag_Pivot",
                           ) -> list:
    """
    ATR tabanlı dinamik ZigZag pivot tespiti.
    
    Parametreler:
      - atr_period: ATR hesaplamada kullanılacak periyot (örn.14)
      - atr_multiplier: pivot onayı için gereken hareket çarpanı
      - min_bars_between_pivots: iki pivot arasında en az kaç bar geçmeli
      - pivot_col_name: df'e yazılacak kolon ismi (opsiyonel)
    
    Dönüş => pivot_list: [ (bar_index, price, pivot_type), ... ]
               pivot_type => +1 pivot high, -1 pivot low
    """
    df = df.copy()
    close_col = f"Close_{time_frame}"
    high_col  = f"High_{time_frame}"
    low_col   = f"Low_{time_frame}"

    if close_col not in df.columns:
        raise ValueError(f"DataFrame does not have required column: {close_col}")

    # ATR hesapla
    df['ATR_TMP'] = compute_atr(df,
                                period=atr_period,
                                high_col=high_col,
                                low_col=low_col,
                                close_col=close_col)

    # ZigZag
    pivot_list = []
    current_leg = None  # 'up' or 'down'
    last_pivot_idx = 0
    last_pivot_price = df[close_col].iloc[0]

    # İlk pivot type: -1 (low) veya +1 (high)? 
    # Basitçe ilk bar'ı pivot low gibi alacağız, devreye girdikçe güncelleriz
    pivot_type = -1  
    pivot_list.append((0, last_pivot_price, pivot_type))
    current_leg = 'up'  # ilk bar'dan itibaren yukarı hareket arıyoruz

    for i in range(1, len(df)):
        c_price = df[close_col].iloc[i]
        c_atr   = df['ATR_TMP'].iloc[i]
        threshold = atr_multiplier * c_atr if c_atr is not None else 0

        # Yukarı hareket senaryosu => pivot low'dan threshold kadar yükseldi mi?
        if current_leg == 'up':
            # eğer c_price < (last_pivot_price - threshold) => pivot high onayla, leg'i down'a çevir
            if c_price < (last_pivot_price - threshold):
                # pivot high => bir önceki i-1, i-2 barların en yükseği nerede?
                # Basit yaklaşım: i-1 bar en yüksek miydi vs. 
                # Biraz geriye bakabiliriz. Ama en basit: pivot bar i-1 => close
                pivot_index = i-1
                pivot_price = df[close_col].iloc[pivot_index]

                # min_bars kontrol
                if (pivot_index - pivot_list[-1][0]) >= min_bars_between_pivots:
                    pivot_list.append((pivot_index, pivot_price, +1))  # pivot high
                    # yeni pivot referansı
                    last_pivot_idx = pivot_index
                    last_pivot_price = pivot_price
                    # leg'i down'a çevir
                    current_leg = 'down'
            else:
                # new high => last pivotı güncelle?
                if c_price > last_pivot_price:
                    last_pivot_price = c_price
                    last_pivot_idx = i

        # Aşağı hareket senaryosu => pivot high'dan threshold kadar düştü mü?
        elif current_leg == 'down':
            # eğer c_price > (last_pivot_price + threshold) => pivot low onayla, leg'i up'a çevir
            if c_price > (last_pivot_price + threshold):
                pivot_index = i-1
                pivot_price = df[close_col].iloc[pivot_index]
                # min_bars kontrol
                if (pivot_index - pivot_list[-1][0]) >= min_bars_between_pivots:
                    pivot_list.append((pivot_index, pivot_price, -1))  # pivot low
                    last_pivot_idx = pivot_index
                    last_pivot_price = pivot_price
                    current_leg = 'up'
            else:
                # new low => last pivotı güncelle
                if c_price < last_pivot_price:
                    last_pivot_price = c_price
                    last_pivot_idx = i

    # DataFrame'e opsiyonel pivot kolonunu yazmak istersek:
    df[pivot_col_name] = np.nan
    for idx, price, ptype in pivot_list:
        if idx < len(df):
            df.at[df.index[idx], pivot_col_name] = ptype

    return pivot_list


#############################################
# 3) Cup & Handle Gelişmiş Algılama (Daha Önceki)
#############################################
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

def _check_retest_cup_handle(df: pd.DataFrame, time_frame: str, rim_line: tuple,
                             break_bar: int, tolerance: float=0.01):
    (xL, pL), (xR, pR)= rim_line
    m, b = line_equation(xL, pL, xR, pR)
    if m is None:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    close_col = f"Close_{time_frame}"
    for i in range(break_bar+1, len(df)):
        c = df[close_col].iloc[i]
        line_y= m*i + b
        dist_ratio= abs(c - line_y)/(abs(line_y)+1e-9)
        if dist_ratio<= tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "distance_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None, "distance_ratio": None}


def detect_cup_and_handle_with_pivots(
    df: pd.DataFrame,
    pivots: list,
    time_frame: str="1h",
    tolerance: float=0.02,
    volume_drop_check: bool=True,
    volume_drop_ratio: float=0.2,
    cup_min_bars: int=20,
    cup_max_bars: int=300,
    handle_ratio: float=0.3,
    handle_max_bars: int=50,
    handle_min_bars: int=3,
    close_above_rim: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Pivots listesinden ( [ (bar_index, price, +1/-1), ... ] ) 
    Cup & Handle aranarak result sözlüğü döner.
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
    close_col= f"Close_{time_frame}"
    volume_col= f"Volume_{time_frame}"
    if close_col not in df.columns:
        result["msgs"].append(f"Missing col: {close_col}")
        return result

    top_pivots= [p for p in pivots if p[2] == +1]
    bot_pivots= [p for p in pivots if p[2] == -1]
    if len(top_pivots)<2 or len(bot_pivots)<1:
        result["msgs"].append("Not enough top/dip pivots for Cup&Handle.")
        return result

    sorted_p= sorted(pivots, key=lambda x: x[0])
    best_cup= None

    # Cup arayışı: pivot low ortada, solda pivot high, sağda pivot high
    for i in range(1, len(sorted_p)-1):
        if sorted_p[i][2]== -1:  # dip
            idxDip, pxDip= sorted_p[i][0], sorted_p[i][1]
            # solda pivot high var mı
            left_candidates= [tp for tp in sorted_p[:i] if tp[2]== +1]
            if not left_candidates:
                continue
            left_top= left_candidates[-1]
            # sağda pivot high
            right_candidates= [tp for tp in sorted_p[i+1:] if tp[2]== +1]
            if not right_candidates:
                continue
            right_top= right_candidates[0]

            bars_cup= right_top[0]- left_top[0]
            if bars_cup< cup_min_bars or bars_cup> cup_max_bars:
                continue

            avg_top= (left_top[1]+ right_top[1])/2
            top_diff= abs(left_top[1]- right_top[1])/(avg_top+1e-9)
            if top_diff> tolerance:
                continue
            if pxDip> avg_top:
                continue

            best_cup= (left_top, (idxDip, pxDip), right_top, bars_cup)
            break

    if not best_cup:
        result["msgs"].append("No valid cup found.")
        return result

    l_top, cup_dip, r_top, cup_bars= best_cup
    result["found"]= True
    result["cup_left_top"]= l_top
    result["cup_bottom"]= cup_dip
    result["cup_right_top"]= r_top
    result["cup_bars"]= cup_bars

    # Volume check
    if volume_drop_check and volume_col in df.columns:
        idxL, pxL = l_top[0], l_top[1]
        idxR, pxR = r_top[0], r_top[1]
        if 0 <= idxL < idxR < len(df):
            cup_vol_series= df[volume_col].iloc[idxL : idxR+1]
            if len(cup_vol_series)>5:
                start_vol= cup_vol_series.iloc[0]
                min_vol= cup_vol_series.min()
                drop_percent= (start_vol- min_vol)/(start_vol+1e-9)
                result["cup_volume_drop"]= drop_percent
                if drop_percent< volume_drop_ratio:
                    result["msgs"].append(f"Cup volume drop {drop_percent:.2f} < ratio({volume_drop_ratio:.2f}).")

    # Rim line
    rim_idxL, rim_pxL= l_top[0], l_top[1]
    rim_idxR, rim_pxR= r_top[0], r_top[1]
    slope_rim, intercept= line_equation(rim_idxL, rim_pxL, rim_idxR, rim_pxR)

    dip_price= cup_dip[1]
    cup_height= ((l_top[1] + r_top[1])/2) - dip_price
    if cup_height<=0:
        return result

    # Handle
    handle_found= False
    handle_top= None
    handle_bars= 0
    handle_start= r_top[0]
    handle_end= min(r_top[0]+ handle_max_bars, len(df)-1)
    if handle_start< handle_end:
        seg= df[close_col].iloc[handle_start: handle_end+1]
        loc_min_val= seg.min()
        loc_min_idx= seg.idxmin()
        handle_bars= (loc_min_idx - handle_start)
        handle_depth= ((r_top[1]+ l_top[1])/2) - loc_min_val
        ratio= handle_depth / (cup_height + 1e-9)
        if handle_bars>= handle_min_bars and handle_bars<= handle_max_bars and ratio<= handle_ratio and ratio>0:
            handle_found= True
            handle_top= (loc_min_idx, loc_min_val)

    result["handle_found"]= handle_found
    result["handle_top"]= handle_top
    result["handle_bars"]= handle_bars

    # Onay => rim line break
    confirmed= False
    last_i= len(df)-1
    last_price= df[close_col].iloc[-1]
    if slope_rim is not None:
        rim_line_val= slope_rim* last_i + intercept
        if close_above_rim:
            if last_price> rim_line_val:
                confirmed= True
        else:
            high_col= f"High_{time_frame}"
            if high_col in df.columns:
                last_high= df[high_col].iloc[-1]
                if last_high> rim_line_val:
                    confirmed= True
    result["confirmed"]= confirmed
    if confirmed:
        result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))

    # Retest
    if check_retest and confirmed and result["rim_line"]:
        retest_info= _check_retest_cup_handle(
            df, time_frame,
            rim_line=result["rim_line"],
            break_bar= last_i,
            tolerance=retest_tolerance
        )
        result["retest_info"]= retest_info

    return result


#############################################
# 4) Örnek: Tüm Süreci Birleştiren Fonksiyon
#############################################

def detect_cup_handle_with_atrzigzag(
    df: pd.DataFrame,
    time_frame: str="1h",
    atr_period: int=14,
    atr_multiplier: float=2.0,
    min_bars_between_pivots: int=3,
    cup_args: dict=None
):
    """
    1) ATR tabanlı ZigZag pivot tespiti (find_pivots_atr_zigzag)
    2) Cup&Handle detect (detect_cup_and_handle_with_pivots)
    Parametreler:
      - cup_args => detect_cup_and_handle_with_pivots fonksiyonuna geçecek ek argümanlar
    Dönüş: cup_handle_result => {'pattern':'cup_handle', 'found':..., ...}
    """
    if cup_args is None:
        cup_args = {}  # default

    # 1) Pivots tespit et
    pivots = find_pivots_atr_zigzag(
        df, time_frame=time_frame,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
        min_bars_between_pivots=min_bars_between_pivots
    )

    # 2) Cup & Handle
    res = detect_cup_and_handle_with_pivots(
        df=df,
        pivots=pivots,
        time_frame=time_frame,
        **cup_args
    )

    return res


#############################################
# 5) DEMO KULLANIM
#############################################
if __name__ == "__main__":
    # Örnek DataFrame: Kolonlar: [Open_1h, High_1h, Low_1h, Close_1h, Volume_1h]
    data = {
        'Open_1h':   [100,101,102,105,104,103,102,102,103,104,105,106,107,107,106],
        'High_1h':   [101,102,105,106,105,104,103,103,104,105,106,107,108,108,107],
        'Low_1h':    [99,100,101,102,102,102,101,101,102,102,103,105,106,106,105],
        'Close_1h':  [100,101,104,105,104,103,102,102,103,104,106,107,107,107,106],
        'Volume_1h': [500,600,650,800,700,600,550,600,580,590,600,650,700,800,750]
    }
    df_example = pd.DataFrame(data)
    # index bar sayısı kadar
    df_example['time'] = pd.date_range('2023-01-01', periods=len(df_example), freq='H')
    df_example.set_index('time', inplace=True)

    # Cup & Handle Tespiti
    cup_args = dict(
        tolerance=0.02,
        volume_drop_check=True,
        volume_drop_ratio=0.2,
        cup_min_bars=3,
        cup_max_bars=100,
        handle_ratio=0.3,
        handle_max_bars=10,
        handle_min_bars=2,
        close_above_rim=True,
        check_retest=True,
        retest_tolerance=0.01
    )
    res = detect_cup_handle_with_atrzigzag(
        df_example,
        time_frame="1h",
        atr_period=14,
        atr_multiplier=2.0,
        min_bars_between_pivots=3,
        cup_args=cup_args
    )

    print("Cup&Handle Detection =>", res)
    # pivotlar da df_example['ZigZag_Pivot'] kolonunda görebilirsiniz (None / +1 / -1).
