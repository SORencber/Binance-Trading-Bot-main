############################################################
# trading_system.py
# Tüm gelişmiş pattern dedektörleri + gerçekçi trade mantığı
############################################################

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

############################################################
# 0) LOGGING (Örnek)
############################################################
def log(msg, level="INFO"):
    print(f"[{level}] {msg}")

############################################################
# 1) CONFIG VE DB
############################################################
DB_PATH = "trades.db"  # SQLite için varsayılan

# Örnek DB sınıfı
class TradeDatabase:
    def __init__(self, db_path="trades.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            price REAL,
            qty REAL,
            pnl REAL,
            note TEXT
        )
        """)
        con.commit()
        con.close()

    def log_trade(self, timestamp, symbol, action, price, qty, pnl, note=""):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        INSERT INTO trade_logs (timestamp, symbol, action, price, qty, pnl, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, symbol, action, price, qty, pnl, note))
        con.commit()
        con.close()

trade_db = TradeDatabase(DB_PATH)


############################################################
# 2) GENEL YARDIMCI FONKSİYONLAR
############################################################

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def line_equation(x1, y1, x2, y2):
    if (x2 - x1) == 0:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return m, b

def line_intersection(m1, b1, m2, b2):
    if m1 == m2:
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y


############################################################
# 3) Gelişmiş Pivot & Zigzag
############################################################

class AdvancedPivotScanner:
    def __init__(self,
                 left_bars=5, 
                 right_bars=5,
                 volume_filter=False,
                 min_atr_factor=0.0,
                 df: pd.DataFrame = None,
                 time_frame: str = "1m"):  
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.volume_filter = volume_filter
        self.min_atr_factor = min_atr_factor
        self.df = df.copy() if df is not None else None
        self.time_frame = time_frame

        if self.df is not None:
            self._ensure_atr()

    def _ensure_atr(self):
        high_col  = get_col_name("High",  self.time_frame)
        low_col   = get_col_name("Low",   self.time_frame)
        close_col = get_col_name("Close", self.time_frame)
        atr_col   = get_col_name("ATR",   self.time_frame)

        if atr_col not in self.df.columns:
            hl_  = f"H-L_{self.time_frame}"
            hpc_ = f"H-PC_{self.time_frame}"
            lpc_ = f"L-PC_{self.time_frame}"
            tr_  = f"TR_{self.time_frame}"

            self.df[hl_]  = self.df[high_col] - self.df[low_col]
            self.df[hpc_] = (self.df[high_col] - self.df[close_col].shift(1)).abs()
            self.df[lpc_] = (self.df[low_col]  - self.df[close_col].shift(1)).abs()
            self.df[tr_]  = self.df[[hl_, hpc_, lpc_]].max(axis=1)
            self.df[atr_col] = self.df[tr_].rolling(14).mean()

    def find_pivots(self):
        price_col = get_col_name("Close", self.time_frame)
        if price_col not in self.df.columns:
            raise ValueError(f"DataFrame does not have column {price_col}")

        price_series = self.df[price_col]
        pivots = []
        n = len(price_series)

        for i in range(self.left_bars, n - self.right_bars):
            val = price_series.iloc[i]
            left_slice  = price_series.iloc[i - self.left_bars : i]
            right_slice = price_series.iloc[i+1 : i+1 + self.right_bars]

            # local max
            if all(val > x for x in left_slice) and all(val >= x for x in right_slice):
                if self._pivot_ok(i, val, +1):
                    pivots.append((i, val, +1))
            # local min
            elif all(val < x for x in left_slice) and all(val <= x for x in right_slice):
                if self._pivot_ok(i, val, -1):
                    pivots.append((i, val, -1))

        return pivots

    def _pivot_ok(self, idx, val, ptype):
        if self.df is None:
            return True

        # hacim filtresi
        if self.volume_filter:
            vol_col = get_col_name("Volume", self.time_frame)
            if vol_col in self.df.columns:
                vol_now = self.df[vol_col].iloc[idx]
                vol_mean = self.df[vol_col].iloc[max(0, idx-20): idx].mean()
                if vol_now < 1.2 * vol_mean:
                    return False

        # ATR faktörü
        if self.min_atr_factor > 0:
            atr_col = get_col_name("ATR", self.time_frame)
            if atr_col in self.df.columns:
                atr_now = self.df[atr_col].iloc[idx]
                if pd.isna(atr_now):
                    return True
                # Örnek bir kontrol
                prev_atr = self.df[atr_col].iloc[idx-1] if idx>0 else atr_now
                if abs(val - prev_atr) < (self.min_atr_factor * atr_now):
                    return False

        return True


def build_zigzag_wave(pivots):
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda x: x[0])
    wave = [sorted_p[0]]
    for i in range(1, len(sorted_p)):
        curr = sorted_p[i]
        prev = wave[-1]
        if curr[2] == prev[2]:
            if curr[2] == +1:
                if curr[1] > prev[1]:
                    wave[-1] = curr
            else:
                if curr[1] < prev[1]:
                    wave[-1] = curr
        else:
            wave.append(curr)
    return wave


############################################################
# 4) Pattern Dedektörleri (HeadShoulders, Double, Elliott vs.)
############################################################

# -- HEAD & SHOULDERS
def detect_head_and_shoulders_advanced(
    df: pd.DataFrame,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_distance_bars: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0
) -> list:
    high_col   = get_col_name("High",  time_frame)
    low_col    = get_col_name("Low",   time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume",time_frame)
    atr_col    = get_col_name("ATR",   time_frame)

    if any(col not in df.columns for col in [high_col,low_col,close_col]):
        return []

    # ATR (isteğe bağlı)
    if atr_filter>0:
        if atr_col not in df.columns:
            df[f"H-L_{time_frame}"]  = df[high_col] - df[low_col]
            df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
            df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()
            df[f"TR_{time_frame}"]   = df[[f"H-L_{time_frame}",
                                           f"H-PC_{time_frame}",
                                           f"L-PC_{time_frame}"]].max(axis=1)
            df[atr_col] = df[f"TR_{time_frame}"].rolling(14).mean()

    # local max pivot bul
    pivot_scanner= AdvancedPivotScanner(
        left_bars= left_bars,
        right_bars= right_bars,
        volume_filter= False,
        min_atr_factor= 0.0,
        df= df,
        time_frame= time_frame
    )
    pivot_list = pivot_scanner.find_pivots()
    top_pivots= [p for p in pivot_list if p[2]== +1]

    results=[]
    for i in range(len(top_pivots)-2):
        L= top_pivots[i]
        H= top_pivots[i+1]
        R= top_pivots[i+2]
        idxL, priceL,_= L
        idxH, priceH,_= H
        idxR, priceR,_= R

        if not (idxL< idxH< idxR):
            continue
        if not (priceH> priceL and priceH> priceR):
            continue

        bars_LH= idxH- idxL
        bars_HR= idxR- idxH
        if bars_LH< min_distance_bars or bars_HR< min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue

        diffShoulder= abs(priceL- priceR)/(priceH+ 1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        seg_LH= df[low_col].iloc[idxL: idxH+1]
        seg_HR= df[low_col].iloc[idxH: idxR+1]
        if len(seg_LH)<1 or len(seg_HR)<1:
            continue
        dip1_idx= seg_LH.idxmin()
        dip2_idx= seg_HR.idxmin()
        dip1_val= df[low_col].iloc[dip1_idx]
        dip2_val= df[low_col].iloc[dip2_idx]

        confirmed= False
        if neckline_break:
            if dip1_idx != dip2_idx:
                m_= (dip2_val - dip1_val)/(dip2_idx- dip1_idx)
                b_= dip1_val - m_* dip1_idx
                last_i= len(df)-1
                last_close= df[close_col].iloc[-1]
                line_y= m_* last_i+ b_
                if last_close< line_y:
                    confirmed= True

        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            if volH> (mean_shoulder_vol*0.8):
                vol_check= False

        res= {
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "shoulder_diff": diffShoulder,
          "neckline": ((dip1_idx, dip1_val),(dip2_idx, dip2_val)),
          "confirmed": confirmed,
          "volume_check": vol_check
        }
        results.append(res)
    return results


def detect_inverse_head_and_shoulders_advanced(
    df: pd.DataFrame,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_distance_bars: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0
) -> list:
    low_col   = get_col_name("Low", time_frame)
    close_col = get_col_name("Close", time_frame)
    volume_col= get_col_name("Volume", time_frame)
    atr_col   = get_col_name("ATR", time_frame)

    if any(col not in df.columns for col in [low_col,close_col]):
        return []

    if atr_filter>0:
        if atr_col not in df.columns:
            high_col= get_col_name("High", time_frame)
            df[f"H-L_{time_frame}"]  = df[high_col] - df[low_col]
            df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
            df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()
            df[f"TR_{time_frame}"]   = df[[f"H-L_{time_frame}",
                                           f"H-PC_{time_frame}",
                                           f"L-PC_{time_frame}"]].max(axis=1)
            df[atr_col]= df[f"TR_{time_frame}"].rolling(14).mean()

    # local min pivot
    pivot_scanner= AdvancedPivotScanner(
        left_bars= left_bars,
        right_bars= right_bars,
        volume_filter= False,
        min_atr_factor= 0.0,
        df= df,
        time_frame= time_frame
    )
    pivot_list= pivot_scanner.find_pivots()
    dip_pivots= [p for p in pivot_list if p[2]== -1]
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

        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            if volH> (mean_shoulder_vol*0.8):
                vol_check= False

        segment_LH= df[close_col].iloc[idxL: idxH+1]
        segment_HR= df[close_col].iloc[idxH: idxR+1]
        if len(segment_LH)<1 or len(segment_HR)<1:
            continue
        T1_idx= segment_LH.idxmax()
        T2_idx= segment_HR.idxmax()
        T1_val= df[close_col].iloc[T1_idx]
        T2_val= df[close_col].iloc[T2_idx]

        confirmed= False
        if neckline_break and close_col in df.columns:
            m_, b_= line_equation(T1_idx, T1_val, T2_idx, T2_val)
            if m_ is not None:
                last_close= df[close_col].iloc[-1]
                last_i= len(df)-1
                line_y= m_* last_i+ b_
                if last_close> line_y:
                    confirmed= True

        res= {
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "shoulder_diff": diffShoulder,
          "volume_check": vol_check,
          "confirmed": confirmed,
          "neckline": ((T1_idx,T1_val), (T2_idx,T2_val))
        }
        results.append(res)
    return results


# -- DOUBLE / TRIPLE TOP / BOTTOM
def detect_double_top(
    pivots,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    df: pd.DataFrame=None
):
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

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
              "tops": [(idx1,price1),(idx2,price2)],
              "neckline": None,
              "confirmed": False,
              "start_bar": idx1,
              "end_bar": idx2,
              "pattern": "double_top"
            }
            used_third= False
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

            if volume_check and df is not None and volume_col in df.columns:
                vol1= df[volume_col].iloc[idx1]
                vol2= df[volume_col].iloc[idx2]
                # 2.top hacmi, 1.top'tan %20 düşük olsun
                if vol2> (vol1*0.8):
                    i+= (2 if used_third else 1)
                    continue

            if neckline_break and df is not None and close_col in df.columns:
                seg_end= t2[0] if not used_third else top_pivots[i+2][0]
                dip_pivs= [pp for pp in pivots if pp[2]== -1 and pp[0]> idx1 and pp[0]< seg_end]
                if dip_pivs:
                    dip_sorted= sorted(dip_pivs, key=lambda x: x[1])
                    neck= dip_sorted[0]
                    found["neckline"]= (neck[0], neck[1])
                    last_close= df[close_col].iloc[-1]
                    if last_close< neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results


def detect_double_bottom(
    pivots,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    df: pd.DataFrame=None
):
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
              "pattern": "double_bottom"
            }
            used_third= False
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
                if vol2> (vol1*0.8):
                    i+= (2 if used_third else 1)
                    continue

            if neckline_break and df is not None and close_col in df.columns:
                seg_end= b2[0] if not used_third else bottom_pivots[i+2][0]
                top_pivs= [pp for pp in pivots if pp[2]== +1 and pp[0]> idx1 and pp[0]< seg_end]
                if top_pivs:
                    top_sorted= sorted(top_pivs, key=lambda x: x[1], reverse=True)
                    neck= top_sorted[0]
                    found["neckline"]= (neck[0], neck[1])
                    last_close= df[close_col].iloc[-1]
                    if last_close> neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results


# -- ELLIOTT
def detect_elliott_5wave_advanced(
    wave,
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
    df: pd.DataFrame = None
) -> dict:
    result = {
        "found": False,
        "trend": None,
        "pivots": [],
        "check_msgs": [],
        "abc": None,
        "extended_5th": False
    }
    if len(wave) < wave_min_bars:
        result["check_msgs"].append("Not enough pivots for Elliott 5-wave.")
        return result

    last5 = wave[-5:]
    types = [p[2] for p in last5]
    up_pattern   = [+1, -1, +1, -1, +1]
    down_pattern = [-1, +1, -1, +1, -1]

    trend = None
    if types == up_pattern:
        trend = "UP"
    elif check_alt_scenarios and (types == down_pattern):
        trend = "DOWN"
    else:
        result["check_msgs"].append("Pivot pattern not matching up or down Elliott.")
        return result

    result["trend"] = trend
    p0i,p0p,_= last5[0]
    p1i,p1p,_= last5[1]
    p2i,p2p,_= last5[2]
    p3i,p3p,_= last5[3]
    p4i,p4p,_= last5[4]
    result["pivots"] = [(p0i,p0p),(p1i,p1p),(p2i,p2p),(p3i,p3p),(p4i,p4p)]

    def wave_len(a,b): return abs(b-a)
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

    if w3< (rule_3rdwave_min_percent* w1):
        result["check_msgs"].append("3rd wave not long enough vs wave1.")
        return result

    if not allow_4th_overlap:
        if trend=="UP" and (p4p< p1p):
            result["check_msgs"].append("4th wave overlap in UP trend.")
            return result
        if trend=="DOWN" and (p4p> p1p):
            result["check_msgs"].append("4th wave overlap in DOWN trend.")
            return result

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

    if extended_waves and check_abc_correction and (len(wave)>=8):
        maybe_abc= wave[-3:]
        abc_types= [p[2] for p in maybe_abc]
        if trend=="UP":
            if abc_types== [-1,+1,-1]:
                result["abc"]= True
        else:
            if abc_types== [+1,-1,+1]:
                result["abc"]= True

    result["found"]= True
    return result


# -- WOLFE
def detect_wolfe_wave_advanced(
    wave,
    time_frame: str = "1m",
    price_tolerance: float = 0.03,
    strict_lines: bool = False,
    breakout_confirm: bool = True,
    line_projection_check: bool = True,
    check_2_4_slope: bool = True,
    check_1_4_intersection_time: bool = True,
    check_time_symmetry: bool = True,
    max_time_ratio: float = 0.3,
    df: pd.DataFrame = None
) -> dict:
    result= {
      "found": False,
      "msgs": [],
      "breakout": False,
      "intersection": None,
      "time_symmetry_ok": True,
      "sweet_zone": None
    }
    if len(wave)<5:
        result["msgs"].append("Not enough pivots (need 5).")
        return result

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

    m13,b13= line_equation(x1,y1, x3,y3)
    m35,b35= line_equation(x3,y3, x5,y5)
    if (m13 is None) or (m35 is None):
        result["msgs"].append("Line(1->3) or (3->5) vertical => fail.")
        return result

    diff_slope= abs(m35- m13)/(abs(m13)+ 1e-9)
    if diff_slope> price_tolerance:
        result["msgs"].append(f"Slope difference(1->3 vs 3->5) too big => {diff_slope:.3f}")

    if check_2_4_slope:
        m24,b24= line_equation(x2,y2, x4,y4)
        if strict_lines and (m24 is not None):
            slope_diff= abs(m24- m13)/(abs(m13)+1e-9)
            if slope_diff> 0.3:
                result["msgs"].append(f"Line(2->4) slope differs from line(1->3) => {slope_diff:.3f}")

    # sweet zone
    m24_,b24_= line_equation(x2,y2, x4,y4)
    if m24_ is not None:
        line13_y5= m13*x5+ b13
        line24_y5= m24_*x5+ b24_
        low_ = min(line13_y5, line24_y5)
        high_= max(line13_y5, line24_y5)
        result["sweet_zone"]= (low_, high_)
        if not (low_<= y5<= high_):
            result["msgs"].append("W5 not in sweet zone")

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

    if line_projection_check:
        m14,b14= line_equation(x1,y1, x4,y4)
        m23,b23= line_equation(x2,y2, x3,y3)
        if (m14 is not None) and (m23 is not None):
            ix,iy= line_intersection(m14,b14, m23,b23)
            if ix is not None:
                result["intersection"]= (ix, iy)
                if check_1_4_intersection_time and ix< x5:
                    result["msgs"].append("Intersection(1->4 & 2->3) < w5 => degrade")

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
                else:
                    result["msgs"].append("No breakout => last_close below line(1->4).")

    result["found"]= True
    return result


# -- Harmonic
def detect_harmonic_pattern_advanced(
    wave,
    time_frame: str = "1m",
    fib_tolerance: float=0.02,
    patterns: list = None,
    df: pd.DataFrame= None,
    left_bars: int=5,
    right_bars: int=5,
    check_volume: bool=False
) -> dict:
    if patterns is None:
        patterns= ["gartley","bat","crab","butterfly","shark","cipher"]
    result= {
      "found": False,
      "pattern_name": None,
      "xabc": [],
      "msgs": []
    }
    if len(wave)<5:
        result["msgs"].append("Not enough pivot for harmonic (need 5).")
        return result

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

    def length(a,b): return abs(b-a)
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
        if abs(mn-mx)< 1e-9:
            return abs(val- mn)<= (mn* tol)
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

        ratio1= AB_XA
        ratio2= BC_AB
        ratio3= CD_BC
        ok1= in_range(ratio1, rngAB_XA, fib_tolerance)
        ok2= in_range(ratio2, rngBC_AB, fib_tolerance)
        ok3= in_range(ratio3, rngCD_BC, fib_tolerance)

        if ok1 and ok2 and ok3:
            found_any= True
            matched_pattern= pat
            break

    if found_any:
        result["found"]= True
        result["pattern_name"]= matched_pattern
        if check_volume and df is not None:
            vol_col= get_col_name("Volume", time_frame)
            if vol_col in df.columns:
                idxD_int= idxD
                if idxD_int< len(df):
                    v_now= df[vol_col].iloc[idxD_int]
                    start_i= max(0, idxD_int- 20)
                    v_mean= df[vol_col].iloc[start_i: idxD_int].mean()
                    # Basit kontrol: D noktasında hacim artışı (opsiyonel)
                    if v_now> (1.3* v_mean):
                        pass
    else:
        result["msgs"].append("No harmonic pattern match.")

    return result


# -- Triangle
def detect_triangle_advanced(
    wave,
    time_frame: str="1m",
    triangle_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    triangle_types: list= None,
    df: pd.DataFrame= None
) -> dict:
    if triangle_types is None:
        triangle_types= ["ascending","descending","symmetrical"]
    result= {
      "found": False,
      "triangle_type": None,
      "breakout": False,
      "msgs": []
    }
    if len(wave)<4:
        result["msgs"].append("Not enough pivot for triangle (need >=4).")
        return result

    last4= wave[-4:]
    p1,p2,p3,p4= last4
    t_list= [p[2] for p in last4]
    up_zig=   [+1,-1,+1,-1]
    down_zig= [-1,+1,-1,+1]

    if t_list not in [up_zig, down_zig]:
        result["msgs"].append("Zigzag pattern not matching triangle requirement.")
        return result

    if t_list== up_zig:
        x1,y1= p1[0], p1[1]
        x3,y3= p3[0], p3[1]
        x2,y2= p2[0], p2[1]
        x4,y4= p4[0], p4[1]
    else:
        x1,y1= p2[0], p2[1]
        x3,y3= p4[0], p4[1]
        x2,y2= p1[0], p1[1]
        x4,y4= p3[0], p3[1]

    m_top,b_top= line_equation(x1,y1, x3,y3)
    m_bot,b_bot= line_equation(x2,y2, x4,y4)
    if m_top is None or m_bot is None:
        result["msgs"].append("Line top/bot eq fail => vertical slope.")
        return result

    def is_flat(m): return (abs(m)< triangle_tolerance)
    top_type= None
    bot_type= None

    if is_flat(m_top):
        top_type= "flat"
    elif m_top>0:
        top_type= "rising"
    else:
        top_type= "falling"

    if is_flat(m_bot):
        bot_type= "flat"
    elif m_bot>0:
        bot_type= "rising"
    else:
        bot_type= "falling"

    tri_type= None
    if top_type=="flat" and bot_type=="rising" and ("ascending" in triangle_types):
        tri_type= "ascending"
    elif top_type=="falling" and bot_type=="flat" and ("descending" in triangle_types):
        tri_type= "descending"
    elif top_type=="falling" and bot_type=="rising" and ("symmetrical" in triangle_types):
        tri_type= "symmetrical"

    if not tri_type:
        result["msgs"].append("No matching triangle type among ascending/descending/symmetrical.")
        return result

    # breakout
    brk= False
    if check_breakout and df is not None and len(df)>0:
        close_col= get_col_name("Close", time_frame)
        if close_col in df.columns:
            last_close= df[close_col].iloc[-1]
            last_i= len(df)-1
            line_y_top= m_top* last_i+ b_top
            line_y_bot= m_bot* last_i+ b_bot
            if tri_type=="ascending":
                if last_close> line_y_top:
                    brk= True
            elif tri_type=="descending":
                if last_close< line_y_bot:
                    brk= True
            else:
                if (last_close> line_y_top) or (last_close< line_y_bot):
                    brk= True

    result["found"]= True
    result["triangle_type"]= tri_type
    result["breakout"]= brk
    return result


# -- Wedge
def detect_wedge_advanced(
    wave,
    time_frame:str= "1m",
    wedge_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    df: pd.DataFrame= None
) -> dict:
    result= {
      "found": False,
      "wedge_type": None,
      "breakout": False,
      "msgs": []
    }
    if len(wave)<5:
        result["msgs"].append("Not enough pivot for wedge (need >=5).")
        return result

    last5= wave[-5:]
    types= [p[2] for p in last5]
    rising_pat=  [+1,-1,+1,-1,+1]
    falling_pat= [-1,+1,-1,+1,-1]

    wedge_type= None
    if types== rising_pat:
        wedge_type= "rising"
    elif types== falling_pat:
        wedge_type= "falling"
    else:
        result["msgs"].append("Pivot pattern not matching rising/falling wedge.")
        return result

    x1,y1= last5[0][0], last5[0][1]
    x3,y3= last5[2][0], last5[2][1]
    x5,y5= last5[4][0], last5[4][1]
    slope_top= (y5- y1)/ ((x5- x1)+1e-9)

    x2,y2= last5[1][0], last5[1][1]
    x4,y4= last5[3][0], last5[3][1]
    slope_bot= (y4- y2)/ ((x4- x2)+1e-9)

    if wedge_type=="rising":
        if (slope_top<0) or (slope_bot<0):
            result["msgs"].append("Expected positive slopes for rising wedge.")
            return result
        if not (slope_bot> slope_top):
            result["msgs"].append("slope(2->4) <= slope(1->3) => not wedge shape.")
            return result
    else: # falling
        if (slope_top>0) or (slope_bot>0):
            result["msgs"].append("Expected negative slopes for falling wedge.")
            return result
        if not (slope_bot> slope_top):
            result["msgs"].append("Dip slope <= top slope => not wedge shape.")
            return result

    ratio= abs(slope_bot- slope_top)/ (abs(slope_top)+1e-9)
    if ratio< wedge_tolerance:
        result["msgs"].append(f"Wedge slope difference ratio {ratio:.3f} < tolerance => might be channel.")
        return result

    brk= False
    if check_breakout and df is not None:
        close_col= get_col_name("Close", time_frame)
        if close_col in df.columns:
            last_close= df[close_col].iloc[-1]
            m_,b_= line_equation(x2,y2, x4,y4)
            if m_ is not None:
                last_i= len(df)-1
                line_y= m_* last_i+ b_
                if wedge_type=="rising":
                    if last_close< line_y:
                        brk= True
                else:
                    if last_close> line_y:
                        brk= True

    result["found"]= True
    result["wedge_type"]= wedge_type
    result["breakout"]= brk
    return result


############################################################
# 5) Hepsini Tek Fonksiyonda Çağırma
############################################################
def detect_all_patterns(
    pivots, 
    wave, 
    df: pd.DataFrame = None, 
    time_frame: str = "1m", 
    config: dict = None
) -> dict:
    if config is None:
        config= {}
    elliott_cfg   = config.get("elliott", {})
    wolfe_cfg     = config.get("wolfe",   {})
    harmonic_cfg  = config.get("harmonic",{})
    hs_cfg        = config.get("headshoulders", {})
    invhs_cfg     = config.get("inverse_headshoulders", {})
    dt_cfg        = config.get("doubletriple", {})
    tri_cfg       = config.get("triangle_wedge", {})
    wedge_cfg     = config.get("wedge_params", {})

    hs_res = detect_head_and_shoulders_advanced(
        df=df, 
        time_frame=time_frame,
        left_bars= hs_cfg.get("left_bars",10),
        right_bars= hs_cfg.get("right_bars",10),
        min_distance_bars= hs_cfg.get("min_distance_bars",10),
        shoulder_tolerance= hs_cfg.get("shoulder_tolerance",0.03),
        volume_decline= hs_cfg.get("volume_decline",True),
        neckline_break= hs_cfg.get("neckline_break",True),
        max_shoulder_width_bars= hs_cfg.get("max_shoulder_width_bars",50),
        atr_filter= hs_cfg.get("atr_filter",0.0)
    )
    inv_hs_res = detect_inverse_head_and_shoulders_advanced(
        df=df,
        time_frame= time_frame,
        left_bars= invhs_cfg.get("left_bars",10),
        right_bars= invhs_cfg.get("right_bars",10),
        min_distance_bars= invhs_cfg.get("min_distance_bars",10),
        shoulder_tolerance= invhs_cfg.get("shoulder_tolerance",0.03),
        volume_decline= invhs_cfg.get("volume_decline",True),
        neckline_break= invhs_cfg.get("neckline_break",True),
        max_shoulder_width_bars= invhs_cfg.get("max_shoulder_width_bars",50),
        atr_filter= invhs_cfg.get("atr_filter",0.0)
    )
    dtops= detect_double_top(
        pivots= pivots,
        time_frame= time_frame,
        tolerance= dt_cfg.get("tolerance",0.01),
        min_distance_bars= dt_cfg.get("min_distance_bars",20),
        triple_variation= dt_cfg.get("triple_variation",True),
        volume_check= dt_cfg.get("volume_check",False),
        neckline_break= dt_cfg.get("neckline_break",False),
        df= df
    )
    dbots= detect_double_bottom(
        pivots= pivots,
        time_frame= time_frame,
        tolerance= dt_cfg.get("tolerance",0.01),
        min_distance_bars= dt_cfg.get("min_distance_bars",20),
        triple_variation= dt_cfg.get("triple_variation",True),
        volume_check= dt_cfg.get("volume_check",False),
        neckline_break= dt_cfg.get("neckline_break",False),
        df= df
    )
    ell_res= detect_elliott_5wave_advanced(
        wave= wave,
        time_frame= time_frame,
        fib_tolerance= elliott_cfg.get("fib_tolerance",0.1),
        wave_min_bars= elliott_cfg.get("wave_min_bars",5),
        extended_waves= elliott_cfg.get("extended_waves",True),
        rule_3rdwave_min_percent= elliott_cfg.get("rule_3rdwave_min_percent",1.618),
        rule_5thwave_ext_range= elliott_cfg.get("rule_5thwave_ext_range",(1.0,1.618)),
        check_alt_scenarios= elliott_cfg.get("check_alt_scenarios",True),
        check_abc_correction= elliott_cfg.get("check_abc_correction",True),
        allow_4th_overlap= elliott_cfg.get("allow_4th_overlap",False),
        min_bar_distance= elliott_cfg.get("min_bar_distance",3),
        check_fib_retracements= elliott_cfg.get("check_fib_retracements",True),
        df= df
    )
    wolfe_res= detect_wolfe_wave_advanced(
        wave= wave,
        time_frame= time_frame,
        price_tolerance= wolfe_cfg.get("price_tolerance",0.03),
        strict_lines= wolfe_cfg.get("strict_lines",False),
        breakout_confirm= wolfe_cfg.get("breakout_confirm",True),
        line_projection_check= wolfe_cfg.get("line_projection_check",True),
        check_2_4_slope= wolfe_cfg.get("check_2_4_slope",True),
        check_1_4_intersection_time= wolfe_cfg.get("check_1_4_intersection_time",True),
        check_time_symmetry= wolfe_cfg.get("check_time_symmetry",True),
        max_time_ratio= wolfe_cfg.get("max_time_ratio",0.3),
        df= df
    )
    harm_res= detect_harmonic_pattern_advanced(
        wave= wave,
        time_frame= time_frame,
        fib_tolerance= harmonic_cfg.get("fib_tolerance",0.02),
        patterns= harmonic_cfg.get("patterns", ["gartley","bat","crab","butterfly","shark","cipher"]),
        df= df,
        left_bars= harmonic_cfg.get("left_bars",5),
        right_bars= harmonic_cfg.get("right_bars",5),
        check_volume= harmonic_cfg.get("check_volume",False)
    )
    tri_res= detect_triangle_advanced(
        wave= wave,
        time_frame= time_frame,
        triangle_tolerance= tri_cfg.get("triangle_tolerance",0.02),
        check_breakout= tri_cfg.get("check_breakout",True),
        check_retest= tri_cfg.get("check_retest",False),
        triangle_types= tri_cfg.get("triangle_types",["ascending","descending","symmetrical"]),
        df= df
    )
    wedge_res= detect_wedge_advanced(
        wave= wave,
        time_frame= time_frame,
        wedge_tolerance= wedge_cfg.get("wedge_tolerance",0.02),
        check_breakout= wedge_cfg.get("check_breakout",True),
        check_retest= wedge_cfg.get("check_retest",False),
        df= df
    )

    return {
        "elliott": ell_res,
        "wolfe": wolfe_res,
        "harmonic": harm_res,
        "headshoulders": hs_res,
        "inverse_headshoulders": inv_hs_res,
        "double_top": dtops,
        "double_bottom": dbots,
        "triangle": tri_res,
        "wedge": wedge_res
    }


############################################################
# 6) Basit ML Örneği (Opsiyonel)
############################################################
class PatternEnsembleModel:
    def __init__(self, model_path:str=None):
        self.model_path = model_path
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier())
        ])
        self.is_fitted = False

    def fit(self, X, y):
        self.pipe.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.pipe.predict(X)

    def extract_features(self, wave):
        n= len(wave)
        if n<2:
            return np.zeros((1,5))
        last= wave[-1]
        second= wave[-2]
        maxi= max([w[1] for w in wave])
        mini= min([w[1] for w in wave])
        amp= maxi- mini
        arr= [n, last[2], last[1], second[2], amp]
        return np.array([arr])

    def save(self):
        if self.model_path:
            joblib.dump(self.pipe, self.model_path)
            log(f"Model saved to {self.model_path}","info")

    def load(self):
        if self.model_path and os.path.exists(self.model_path):
            self.pipe= joblib.load(self.model_path)
            self.is_fitted= True
            log(f"Model loaded from {self.model_path}","info")


############################################################
# 7) Breakout + Hacim Kontrolü (Opsiyonel)
############################################################
def check_breakout_volume(
    df: pd.DataFrame, 
    time_frame: str = "1m",
    atr_window: int = 14,
    vol_window: int = 20
) -> tuple:
    high_col   = get_col_name("High", time_frame)
    low_col    = get_col_name("Low",  time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume", time_frame)
    atr_col    = get_col_name("ATR", time_frame)

    if atr_col not in df.columns:
        hl_  = f"H-L_{time_frame}"
        hpc_ = f"H-PC_{time_frame}"
        lpc_ = f"L-PC_{time_frame}"
        tr_  = f"TR_{time_frame}"
        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
            return (False,False,False)
        df[hl_]  = df[high_col] - df[low_col]
        df[hpc_] = (df[high_col] - df[close_col].shift(1)).abs()
        df[lpc_] = (df[low_col]  - df[close_col].shift(1)).abs()
        df[tr_]  = df[[hl_, hpc_, lpc_]].max(axis=1)
        df[atr_col] = df[tr_].rolling(atr_window).mean()

    if len(df)<2:
        return (False,False,False)

    last_close = df[close_col].iloc[-1]
    prev_close = df[close_col].iloc[-2]
    last_atr   = df[atr_col].iloc[-1]
    if pd.isna(last_atr):
        last_atr=0

    breakout_up   = (last_close - prev_close)> last_atr
    breakout_down = (prev_close - last_close)> last_atr

    volume_spike= False
    if volume_col in df.columns and len(df)> vol_window:
        v_now  = df[volume_col].iloc[-1]
        v_mean = df[volume_col].rolling(vol_window).mean().iloc[-2]
        volume_spike= (v_now> 1.5* v_mean)

    return (breakout_up, breakout_down, volume_spike)


############################################################
# 8) Sinyal Oluşturma + Trade Yönetimi
############################################################

TIMEFRAME_CONFIGS = {
    # Yukarıdaki parametreleri buraya ekledik. (Kısaltılmış; uzun config yukarıda)
    "1m": {
        "system_params": {
            "pivot_left_bars": 5,
            "pivot_right_bars": 5,
            "volume_filter": True,
            "min_atr_factor": 0.3
        },
        "pattern_config": {
            # ...
            "elliott": {
                "fib_tolerance": 0.08,
                "wave_min_bars": 20,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.5,
                "rule_5thwave_ext_range": (1.0, 1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            # ... (Diğerleri)
            "doubletriple": {
                "tolerance": 0.015,
                "min_distance_bars": 20,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            # vb.
        }
    }
    # Diğer timeframe'ler benzer şekilde eklenebilir...
}


def generate_signals(df: pd.DataFrame, time_frame: str = "1m", ml_model=None) -> dict:
    """
    Pattern'leri tarar, basit bir skor tabanlı sinyal döndürür.
    """
    if time_frame not in TIMEFRAME_CONFIGS:
        raise ValueError(f"Invalid time_frame='{time_frame}'")

    tf_settings = TIMEFRAME_CONFIGS[time_frame]
    system_params= tf_settings["system_params"]
    pattern_conf = tf_settings["pattern_config"]

    # Pivots + wave
    scanner= AdvancedPivotScanner(
        left_bars= system_params["pivot_left_bars"],
        right_bars= system_params["pivot_right_bars"],
        volume_filter= system_params["volume_filter"],
        min_atr_factor= system_params["min_atr_factor"],
        df= df,
        time_frame= time_frame
    )
    pivots= scanner.find_pivots()
    wave= build_zigzag_wave(pivots)

    # Pattern tespiti
    patterns= detect_all_patterns(pivots, wave, df=df, time_frame=time_frame, config=pattern_conf)

    # ML tahmini (isteğe bağlı)
    ml_label= None  # 0=HOLD,1=BUY,2=SELL
    if ml_model is not None:
        feats= ml_model.extract_features(wave)
        ml_label= ml_model.predict(feats)[0]

    # Breakout + hacim
    b_up, b_down, v_spike = check_breakout_volume(df, time_frame=time_frame)

    # Skor tabanlı örnek:
    score_map = {
        "headshoulders": -3,
        "inverse_headshoulders": +3,
        "double_top": -2,
        "double_bottom": +2,
        "triangle_BUY": +1,
        "triangle_SELL": -1,
        "wedge_BUY": +1,
        "wedge_SELL": -1,
        "elliott_UP": +2,
        "elliott_DOWN": -2,
        "harmonic": -1,
        "wolfe": +1,
        "ml_buy": +3,
        "ml_sell": -3
    }
    total_score = 0
    reasons = []

    # HeadShoulders
    if len(patterns["headshoulders"])>0:
        total_score += score_map["headshoulders"]
        reasons.append("headshoulders")
    if len(patterns["inverse_headshoulders"])>0:
        total_score += score_map["inverse_headshoulders"]
        reasons.append("inverse_headshoulders")

    if len(patterns["double_top"])>0:
        total_score += score_map["double_top"]
        reasons.append("double_top")
    if len(patterns["double_bottom"])>0:
        total_score += score_map["double_bottom"]
        reasons.append("double_bottom")

    if patterns["triangle"]["found"]:
        tri_dir = None
        if b_up:
            tri_dir = "BUY"
        elif b_down:
            tri_dir = "SELL"
        if tri_dir:
            key_ = f"triangle_{tri_dir}"
            if key_ in score_map:
                total_score += score_map[key_]
                reasons.append(f"triangle_{tri_dir}")

    if patterns["wedge"]["found"]:
        wed_dir = None
        if b_up:
            wed_dir = "BUY"
        elif b_down:
            wed_dir = "SELL"
        if wed_dir:
            key_ = f"wedge_{wed_dir}"
            if key_ in score_map:
                total_score += score_map[key_]
                reasons.append(f"wedge_{wed_dir}")

    ell = patterns["elliott"]
    if ell["found"]:
        if ell["trend"] == "UP":
            total_score += score_map["elliott_UP"]
            reasons.append("elliott_UP")
        elif ell["trend"] == "DOWN":
            total_score += score_map["elliott_DOWN"]
            reasons.append("elliott_DOWN")

    if patterns["harmonic"]["found"]:
        total_score += score_map["harmonic"]
        reasons.append("harmonic")

    if patterns["wolfe"]["found"]:
        total_score += score_map["wolfe"]
        reasons.append("wolfe")

    if ml_label == 1:
        total_score += score_map["ml_buy"]
        reasons.append("ml_buy")
    elif ml_label == 2:
        total_score += score_map["ml_sell"]
        reasons.append("ml_sell")

    # Breakout ek puan
    if total_score > 0:  # BUY yönlü
        if b_up:
            total_score += 1
            reasons.append("breakout_up")
            if v_spike:
                total_score += 1
                reasons.append("vol_spike_up")
    elif total_score < 0:  # SELL yönlü
        if b_down:
            total_score -= 1
            reasons.append("breakout_down")
            if v_spike:
                total_score -= 1
                reasons.append("vol_spike_down")

    # Final
    final_signal = "HOLD"
    if total_score >= 2:
        final_signal = "BUY"
    elif total_score <= -2:
        final_signal = "SELL"

    reason_str = ",".join(reasons) if reasons else "NONE"
    return {
        "signal": final_signal,
        "score": total_score,
        "reason": reason_str,
        "patterns": patterns,
        "ml_label": ml_label,
        "breakout_up": b_up,
        "breakout_down": b_down,
        "volume_spike": v_spike,
        "time_frame": time_frame
    }


############################################################
# 9) Örnek: Trade Manager (Long / Short / Exit)
############################################################

class TradeManager:
    """
    Örnek bir trade yöneticisi.
    - Tek seferde sadece 1 pozisyon (ya long ya short) tutar.
    - Giriş sinyali => (BUY => Long aç) veya (SELL => Short aç)
    - StopLoss / TakeProfit / TrailingStop vs. 
    """
    def __init__(self, initial_balance=10000.0, risk_per_trade=0.01):
        self.position = None  # { 'side': 'LONG'/'SHORT', 'entry_price': float, 'stop_loss': float, 'size': float, ...}
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade

    def on_signal(self, signal_dict, df:pd.DataFrame, symbol="BTCUSDT"):
        """
        signal_dict => generate_signals sonucu
        df => son bar(lar) => price
        symbol => trade edilecek sembol
        """
        close_col = get_col_name("Close", signal_dict["time_frame"])
        last_close= df[close_col].iloc[-1]

        # Eğer pozisyon yoksa => sinyale göre pozisyon aç
        if self.position is None:
            if signal_dict["signal"]=="BUY":
                # Long aç
                self.open_position("LONG", last_close, df.index[-1], symbol)
            elif signal_dict["signal"]=="SELL":
                # Short aç
                self.open_position("SHORT", last_close, df.index[-1], symbol)
        else:
            # Pozisyon varsa => exit veya stop vb. kontrol
            side= self.position["side"]
            entry= self.position["entry_price"]
            size= self.position["size"]

            # Basit kural: Tersi sinyal gelirse direkt kapat
            if side=="LONG" and signal_dict["signal"]=="SELL":
                self.close_position(last_close, df.index[-1], symbol, reason="OppositeSignal")
            elif side=="SHORT" and signal_dict["signal"]=="BUY":
                self.close_position(last_close, df.index[-1], symbol, reason="OppositeSignal")

            # StopLoss ya da trailing
            # Örnek: Basit stop => Long için entry'nin %2 altına inerse kapat
            # (Gerçek hayatta ATR-based stop vs. daha mantıklı)
            if side=="LONG":
                if last_close< entry*0.98:
                    self.close_position(last_close, df.index[-1], symbol, reason="StopLossHit")
            elif side=="SHORT":
                # Short'ta tam tersi => last_close entry'nin %2 üstüne çıkarsa stop
                if last_close> entry*1.02:
                    self.close_position(last_close, df.index[-1], symbol, reason="StopLossHit")

    def open_position(self, side:str, price: float, bar_time, symbol):
        # Risk miktarı
        risk_amount = self.balance * self.risk_per_trade
        # Basitçe "hesap bakiyesinin %1'i kadar" riske girelim. 
        # 10000 balance => risk_amount=100 => position size approx?

        # StopLoss varsayımı => %2 
        # Örnek: LONG'ta stop => entry*(1-0.02)= entry*0.98
        # Pozisyon büyüklüğü => risk_amount / (2% of entry_price)
        # => risk_amount / (price*0.02)
        # => bu tamamen örnek bir hesap
        stop_loss= None
        if side=="LONG":
            stop_price_diff= price*0.02
            size= risk_amount/ stop_price_diff
            stop_loss= price*0.98
        else: # SHORT
            stop_price_diff= price*0.02
            size= risk_amount/ stop_price_diff
            stop_loss= price*1.02

        self.position = {
            "side": side,
            "entry_price": price,
            "stop_loss": stop_loss,
            "size": size,
            "entry_time": bar_time
        }
        log(f"Open {side} position at {price:.2f}, SL={stop_loss:.2f}, size={size:.4f}", "INFO")
        # trade_db.log_trade(...) => DB'ye kaydedebilirsiniz
        trade_db.log_trade(str(bar_time), symbol, f"OPEN_{side}", price, size, 0.0, "PositionOpened")

    def close_position(self, price: float, bar_time, symbol, reason=""):
        if self.position is None:
            return
        side= self.position["side"]
        entry= self.position["entry_price"]
        size= self.position["size"]

        if side=="LONG":
            pnl= (price- entry)* size
        else:
            pnl= (entry- price)* size

        self.balance+= pnl
        log(f"Close {side} position at {price:.2f}, PnL={pnl:.2f}, reason={reason}", "INFO")
        trade_db.log_trade(str(bar_time), symbol, f"CLOSE_{side}", price, size, pnl, reason)

        self.position = None


############################################################
# 10) Örnek "main" Uygulaması
############################################################
def main_trading_loop(df, symbol="BTCUSDT", time_frame="1m"):
    """
    df => Güncel/Geçmiş bar datası (pd.DataFrame)
    symbol => işlem yapacağınız sembol
    time_frame => 1m/5m/15m vb.
    """
    tm= TradeManager(initial_balance=10000.0, risk_per_trade=0.01)
    # Örnek: Her bar kapandığında sinyal üret => trade manager'e gönder
    for i in range(100, len(df)): 
        # i'ye kadar olan veriyi al
        df_slice= df.iloc[:i].copy()
        sig= generate_signals(df_slice, time_frame=time_frame, ml_model=None)
        tm.on_signal(sig, df_slice, symbol=symbol)

    log(f"Final Balance= {tm.balance:.2f}", "INFO")


############################################################
# Eğer direkt çalıştırmak isterseniz:
############################################################
if __name__=="__main__":
    # Örnek DataFrame yaratma (normalde siz CSV veya API'den okuyacaksınız)
    dates = pd.date_range("2023-01-01", periods=300, freq="1min")
    data = {
        "Open_1m": np.random.randn(len(dates)).cumsum() + 20000,
        "High_1m": np.nan,
        "Low_1m":  np.nan,
        "Close_1m": np.nan,
        "Volume_1m": np.random.randint(100,1000, size=len(dates))
    }
    df_mock= pd.DataFrame(data, index=dates)
    df_mock["High_1m"] = df_mock["Open_1m"] + np.random.rand(len(dates))*10
    df_mock["Low_1m"]  = df_mock["Open_1m"] - np.random.rand(len(dates))*10
    df_mock["Close_1m"]= df_mock["Open_1m"] + np.random.randn(len(dates))*2

    # main loop
    main_trading_loop(df_mock, symbol="BTCUSDT", time_frame="1m")
