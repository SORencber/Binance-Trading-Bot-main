"""
advanced_patterns.py

"""


import pandas as pd
import math


##############################################################################
# 0) GENEL YARDIMCI FONKSİYONLAR
##############################################################################

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    'High' + '5m' -> 'High_5m'
    """
    return f"{base_col}_{time_frame}"


def line_equation(x1, y1, x2, y2):
    """
    İki nokta üzerinden (m, b) formunda doğru denklem katsayılarını döndürür.
    Dikey ise (None, None).
    """
    if (x2 - x1) == 0:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return m, b


def line_intersection(m1, b1, m2, b2):
    """
    İki doğrunun kesişim noktası (x, y).
    Paralel ise (None, None).
    """
    if m1 == m2:
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y


##############################################################################
# 1) GELİŞMİŞ PIVOT SCANNER (Local Max/Min + Hacim + ATR)
##############################################################################

class PivotScanner:
    """
    Local max/min bulmak için gelişmiş bir tarayıcı.
    left_bars, right_bars => local tepe/dip aralığı
    volume_filter => True ise pivot anındaki hacim, ortalama hacmi geçmeli
    atr_filter    => 0'dan büyük ise, pivotın ATR'e kıyasla anlamlı olması istenir

    ATR ve Volume hesabı, time_frame'e göre 'High_{tf}, Low_{tf}, Close_{tf}, Volume_{tf}' 
    kolonlarını kullanır.
    """

    def __init__(self,
                 left_bars: int = 5,
                 right_bars: int = 5,
                 volume_filter: bool = False,
                 atr_filter: float = 0.0,
                 df: pd.DataFrame = None,
                 time_frame: str = "1m"):
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.volume_filter = volume_filter
        self.atr_filter = atr_filter
        self.df = df
        self.time_frame = time_frame

        if self.df is not None:
            self._prepare_atr()

    def _prepare_atr(self):
        """
        ATR kolonu yoksa ekler. 
        """
        if self.atr_filter <= 0:
            return
        high_col = get_col_name("High", self.time_frame)
        low_col  = get_col_name("Low",  self.time_frame)
        close_col= get_col_name("Close",self.time_frame)
        atr_col  = get_col_name("ATR", self.time_frame)

        if atr_col not in self.df.columns:
            self.df[f"H-L_{self.time_frame}"]  = self.df[high_col] - self.df[low_col]
            self.df[f"H-PC_{self.time_frame}"] = (self.df[high_col] - self.df[close_col].shift(1)).abs()
            self.df[f"L-PC_{self.time_frame}"] = (self.df[low_col]  - self.df[close_col].shift(1)).abs()
            self.df[f"TR_{self.time_frame}"]   = self.df[
                [f"H-L_{self.time_frame}", f"H-PC_{self.time_frame}", f"L-PC_{self.time_frame}"]
            ].max(axis=1)
            self.df[atr_col] = self.df[f"TR_{self.time_frame}"].rolling(14).mean()

    def find_pivots(self):
        """
        pivot_type= +1 => local max, -1 => local min
        """
        if self.df is None:
            return []
        close_col = get_col_name("Close", self.time_frame)
        if close_col not in self.df.columns:
            return []

        price_series = self.df[close_col]
        n = len(price_series)
        pivots = []

        for i in range(self.left_bars, n - self.right_bars):
            val = price_series.iloc[i]
            left_slice  = price_series.iloc[i - self.left_bars : i]
            right_slice = price_series.iloc[i+1 : i+1 + self.right_bars]

            is_local_max = (all(val > l for l in left_slice) and 
                            all(val >= r for r in right_slice))
            is_local_min = (all(val < l for l in left_slice) and
                            all(val <= r for r in right_slice))

            if is_local_max:
                if self._pivot_ok(i, val, +1):
                    pivots.append((i, val, +1))
            elif is_local_min:
                if self._pivot_ok(i, val, -1):
                    pivots.append((i, val, -1))

        return pivots

    def _pivot_ok(self, idx, val, ptype):
        """
        volume_filter => pivot anında hacim ortalamanın üstünde mi?
        atr_filter    => pivot, ATR'e göre anlamlı mı?
        """
        volume_col = get_col_name("Volume", self.time_frame)
        atr_col    = get_col_name("ATR",    self.time_frame)

        # Hacim filtresi
        if self.volume_filter and volume_col in self.df.columns:
            vol_now = self.df[volume_col].iloc[idx]
            window_size = 20
            if idx-window_size < 0:
                window_start=0
            else:
                window_start= idx-window_size
            vol_mean = self.df[volume_col].iloc[window_start: idx].mean()
            if vol_now < 1.2 * vol_mean:
                return False

        # ATR filtresi
        if self.atr_filter>0 and atr_col in self.df.columns:
            pivot_atr= self.df[atr_col].iloc[idx]
            if pd.isna(pivot_atr):
                return True
            left_val = self.df[atr_col].iloc[idx-1] if idx>0 else pivot_atr
            diff_left= abs(val - (self.df[get_col_name("Close", self.time_frame)].iloc[idx-1] if idx>0 else val))
            # Mantıken pivotun (val) ile yakın barların farkı ATR'e göre 
            # en az self.atr_filter * pivot_atr olmalı:
            if (diff_left < (self.atr_filter * pivot_atr)):
                return False

        return True


##############################################################################
# 2) HEAD & SHOULDERS (Advanced) - Tam sürüm
##############################################################################

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
    """
    Gelişmiş Head & Shoulders. 
    Tamamen gerçekçi: Head hacmi, omuzların hacminden farklı olabilir. 
    Boyun çizgisi altına inince 'confirmed' True.
    """
    high_col   = get_col_name("High",  time_frame)
    low_col    = get_col_name("Low",   time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume",time_frame)
    atr_col    = get_col_name("ATR",   time_frame)

    if any(col not in df.columns for col in [high_col,low_col,close_col]):
        return []

    # ATR
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
    pivot_scanner= PivotScanner(
        left_bars= left_bars,
        right_bars= right_bars,
        volume_filter= False,  # H&S'de volume_filter pivot bulmada opsiyonel
        atr_filter= 0.0,
        df= df,
        time_frame= time_frame
    )
    pivot_list = pivot_scanner.find_pivots()
    # sadece +1 tip => top
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
            # Burada gerçekçi: Head hacmi, sol ve sağ omuzdan ortalama %20 düşük mü?
            mean_shoulder_vol= (volL+ volR)/2
            if volH> (mean_shoulder_vol*0.8):
                vol_check= False

        res= {
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "bars_L-H": bars_LH,
          "bars_H-R": bars_HR,
          "shoulder_diff": diffShoulder,
          "neckline": ((dip1_idx, dip1_val),(dip2_idx, dip2_val)),
          "confirmed": confirmed,
          "volume_check": vol_check
        }
        results.append(res)
    return results


##############################################################################
# 3) INVERSE HEAD & SHOULDERS (Advanced) - Tam sürüm
##############################################################################

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
    """
    Gelişmiş Inverse Head & Shoulders dedektörü.
    local min pivot => HEAD (en düşük), Sol/ Sağ omuz mesafesi, 
    boyun çizgisi => local max bulma, volume => Head hacmi, omuzlardan ortalama 
    %20 az vs. 
    """
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
    pivot_scanner= PivotScanner(
        left_bars= left_bars,
        right_bars= right_bars,
        volume_filter= False,
        atr_filter= 0.0,
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

        # boyun => local max L->H ve H->R
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
          "bars_L-H": bars_LH,
          "bars_H-R": bars_HR,
          "shoulder_diff": diffShoulder,
          "volume_check": vol_check,
          "confirmed": confirmed,
          "neckline": ((T1_idx,T1_val), (T2_idx,T2_val))
        }
        results.append(res)
    return results


##############################################################################
# 4) DOUBLE / TRIPLE TOP - BOTTOM (Advanced)
##############################################################################

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
    """
    Gelişmiş Double/Triple Top. 
    Pivots => (idx, price, +1) tepe pivotları. 
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
                # Gerçekçi => 2.top hacmi, 1.top'tan %20 düşük olsun
                if vol2> (vol1*0.8):
                    i+= (2 if used_third else 1)
                    continue

            if neckline_break and df is not None and close_col in df.columns:
                # Neckline => en düşük dip pivot idx1< dip< idx2
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
    """
    Gelişmiş Double/Triple Bottom.
    Pivots => (idx, price, -1) dip pivotları. 
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
                # 2.dip hacmi, 1.dipe göre %20 düşük
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


##############################################################################
# 5) WOLFE WAVE (Advanced)
##############################################################################

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
    """
    Gelişmiş Wolfe Wave dedektörü. 
    Son 5 pivot => w1,w2,w3,w4,w5 
    sweet zone, epa line, time symmetry vb.
    """
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

    # sweet zone => w5 => between line(1->3) & line(2->4)
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


##############################################################################
# DEVAM: BUNDAN SONRAKİ PATTERNLER (Elliott, Harmonic, Triangle, Wedge, vb.)
##############################################################################

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
    """
    Gelişmiş Elliott 5 Dalga Dedektörü:
     - wave: [(bar_index, price, pivot_type), ...] en az 5 pivot
     - pivot_type: +1 => tepe, -1 => dip
     - check_alt_scenarios => DOWN trend dalga sayımına da izin
     - check_fib_retracements => wave2, wave4 retracement oranları
     - rule_3rdwave_min_percent => wave3, wave1'in bu katsayısı kadar min
     - wave_min_bars => en az 5 pivot
     - min_bar_distance => dalgalar arası min bar
     - extended_waves => wave5 extension, ABC correction vb.
    Dönüş => {
      "found": bool,
      "trend": "UP" / "DOWN" / None,
      "pivots": [...],
      "check_msgs": [...],
      "abc": bool veya None,
      "extended_5th": bool
    }
    """
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

    # min_bar_distance
    d1= p1i- p0i
    d2= p2i- p1i
    d3= p3i- p2i
    d4= p4i- p3i
    if any(d< min_bar_distance for d in [d1,d2,d3,d4]):
        result["check_msgs"].append("Bar distance too small between waves.")
        return result

    # wave3 => en az rule_3rdwave_min_percent * wave1
    if w3< (rule_3rdwave_min_percent* w1):
        result["check_msgs"].append("3rd wave not long enough vs wave1.")
        return result

    # 4th overlap => allow_4th_overlap => false => 4th dalga, 1. dalga fiyat bölgesine girmesin
    if not allow_4th_overlap:
        if trend=="UP" and (p4p< p1p):
            result["check_msgs"].append("4th wave overlap in UP trend.")
            return result
        if trend=="DOWN" and (p4p> p1p):
            result["check_msgs"].append("4th wave overlap in DOWN trend.")
            return result

    # fib retracements => wave2 / wave1, wave4 / wave3
    if check_fib_retracements:
        w2r= w2/(w1+1e-9)
        w4r= w4/(w3+1e-9)
        typical_min= 0.382- fib_tolerance
        typical_max= 0.618+ fib_tolerance
        if not (typical_min<= w2r<= typical_max):
            result["check_msgs"].append("Wave2 retracement ratio not in typical range.")
        if not (typical_min<= w4r<= typical_max):
            result["check_msgs"].append("Wave4 retracement ratio not in typical range.")

    # 5th wave => p3->p4 => length= w4, compare wave1 => rule_5thwave_ext_range
    wave5_ratio= w4/ (w1+1e-9)
    if (wave5_ratio>= rule_5thwave_ext_range[0]) and (wave5_ratio<= rule_5thwave_ext_range[1]):
        result["extended_5th"]= True

    # ABC => extended_waves => wave[-3:] => A,B,C
    if extended_waves and check_abc_correction and (len(wave)>=8):
        # p5 => wave[-5], belki wave[-3], wave[-2], wave[-1] => A,B,C
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
    """
    Gelişmiş Harmonic Pattern dedektörü. 
    X->A->B->C->D pivotları. 
    patterns => ["gartley","bat","crab","butterfly","shark","cipher", ...]
    fib_tolerance => her orana ±% 
    check_volume => D noktasında hacim artışı vs. opsiyonel
    wave => zigzag pivot (en az 5 pivot)

    Dönüş => {
      "found": bool,
      "pattern_name": "gartley"/"bat"/... veya None,
      "xabc": [... pivotlar],
      "msgs": [...]
    }
    """
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

    # son 5 pivot => X, A, B, C, D 
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

    # harmonic spec
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
        # eğer rng sabit => mn==mx
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
        # volume check?
        if check_volume and df is not None:
            vol_col= get_col_name("Volume", time_frame)
            if vol_col in df.columns:
                idxD_int= idxD
                if idxD_int< len(df):
                    v_now= df[vol_col].iloc[idxD_int]
                    start_i= max(0, idxD_int- 20)
                    v_mean= df[vol_col].iloc[start_i: idxD_int].mean()
                    if v_now> (1.3* v_mean):
                        pass  # opsiyonel => "Vol at D is high => stronger"
    else:
        result["msgs"].append("No harmonic pattern match in given list.")

    return result


def detect_triangle_advanced(
    wave,
    time_frame: str="1m",
    triangle_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    triangle_types: list= None,
    df: pd.DataFrame= None
) -> dict:
    """
    Gelişmiş Üçgen Formasyonu Dedektörü: 
      ascending, descending, symmetrical
    wave => en az 4 pivot: p1, p2, p3, p4
    pivot_type: +1, -1, ...
    check_breakout => son bar (df) 3gen çizgileri kırdı mı
    triangle_tolerance => slope flat/rising/falling
    Dönüş => {
      "found": bool,
      "triangle_type": "ascending"/"descending"/"symmetrical"/None,
      "breakout": bool,
      "msgs": [...]
    }
    """
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
            else: # symmetrical
                if (last_close> line_y_top) or (last_close< line_y_bot):
                    brk= True
    result["found"]= True
    result["triangle_type"]= tri_type
    result["breakout"]= brk
    return result


def detect_wedge_advanced(
    wave,
    time_frame:str= "1m",
    wedge_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    df: pd.DataFrame= None
) -> dict:
    """
    Gelişmiş Wedge Formasyonu:
    - En az 5 pivot => p1,p2,p3,p4,p5
    - rising wedge => +1,-1,+1,-1,+1
    - falling wedge => -1,+1,-1,+1,-1
    wedge_tolerance => slope fark toleransı
    check_breakout => son fiyata bak
    Dönüş => {
       "found": bool,
       "wedge_type": "rising" / "falling" / None,
       "breakout": bool,
       "msgs": []
    }
    """
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
            # line(2->4) => eğer rising => breakout alt => last_close< line(2->4) => SELL
            # ama wedge genelde "rising wedge" => Aşağı breakout
            # "falling wedge" => yukarı
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


##############################################################################
# 10) TÜM PATTERNLERİ TEK FONKSİYONLA ÇAĞIRMA
##############################################################################
def detect_all_patterns(
    pivots, 
    wave, 
    df: pd.DataFrame = None, 
    time_frame: str = "1m", 
    config: dict = None
) -> dict:
    """
    Gelişmiş detect_all_patterns:
    config => {
       "elliott": { ... },
       "wolfe": { ... },
       "harmonic": { ... },
       "headshoulders": { ... },
       "inverse_headshoulders": {...},
       "doubletriple": { ... },
       "triangle_wedge": { ... },
       "wedge_params": { ... }
    }
    pivots => local max/min => (idx, price, ptype)
    wave   => zigzag (tepe/dip sıralı) => Elliott/Wolfe/Harmonic ...
    Dönüş => {
       "elliott": {...},
       "wolfe": {...},
       "harmonic": {...},
       "headshoulders": [...],
       "inverse_headshoulders": [...],
       "double_top": [...],
       "double_bottom": [...],
       "triangle": {...},
       "wedge": {...}
    }
    """
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

    # Head & Shoulders
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
    #print("detect_head_and_shoulders_advanced",hs_res ) 
    # Inverse H&S
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
    #print("detect_inverse_head_and_shoulders_advanced",inv_hs_res ) 

    # Double/Triple top
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
    #print("detect_double_top",dtops ) 

    # Double/Triple bottom
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
    #print("detect_double_bottom",dbots ) 

    # Elliott
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
    #print("detect_elliott_5wave_advanced",ell_res ) 

    # Wolfe
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
    #print("detect_wolfe_wave_advanced",wolfe_res ) 

    # Harmonic
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
    #print("detect_harmonic_pattern_advanced",harm_res ) 

    # Triangle
    tri_res= detect_triangle_advanced(
        wave= wave,
        time_frame= time_frame,
        triangle_tolerance= tri_cfg.get("triangle_tolerance",0.02),
        check_breakout= tri_cfg.get("check_breakout",True),
        check_retest= tri_cfg.get("check_retest",False),
        triangle_types= tri_cfg.get("triangle_types",["ascending","descending","symmetrical"]),
        df= df
    )
    #print("detect_triangle_advanced",tri_res ) 

    # Wedge
    wedge_res= detect_wedge_advanced(
        wave= wave,
        time_frame= time_frame,
        wedge_tolerance= wedge_cfg.get("wedge_tolerance",0.02),
        check_breakout= wedge_cfg.get("check_breakout",True),
        check_retest= wedge_cfg.get("check_retest",False),
        df= df
    )
    #print("detect_wedge_advanced",wedge_res ) 

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
