# patterns/elliott.py
import pandas as pd
from core.logging_setup import log

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
    return f"{base_col}_{time_frame}"
def detect_elliott_5wave_advanced(
    wave,
    time_frame: str = "1h",   # opsiyonel, eğer df ile ATR vb. teyit isterseniz
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
    Gelişmiş Elliott 5 Dalga Dedektörü.

    Parametreler:
    ------------
    wave : list[ (bar_index, price, pivot_type), ... ]
       => ZigZag şeklinde dizilmiş en az 5 pivot (tepe/dip).
          pivot_type: +1 => tepe, -1 => dip
    time_frame : str
       => "1h","1m", vb. (df ile ek teyit kullanacak iseniz).
    fib_tolerance : float
       => Fibonacci oranlarındaki tolerans (ör. 0.1 => %10).
    wave_min_bars : int
       => En az 5 pivot lazım (1–2–3–4–5).
    extended_waves : bool
       => 5 dalga sonrasında ABC düzeltme, wave5 ext. vs. ek kontroller.
    rule_3rdwave_min_percent : float
       => 3. dalga, 1. dalganın en az “bu katsayı”sı kadar uzun olsun (1.618).
    rule_5thwave_ext_range : tuple
       => (min_ext, max_ext) => 5. dalga, 1. dalga uzunluğunun bu aralığında kalabilir.
    check_alt_scenarios : bool
       => Downtrend dalga sayımı (pivot tipi -1,+1,-1,+1,-1).
    check_abc_correction : bool
       => 5 dalga sonrası ABC (3 pivot) ek kontrol.
    allow_4th_overlap : bool
       => True ise, 4. dalga 1. dalga fiyat bölgesine girebilir (klasik Elliott yasaklar).
    min_bar_distance : int
       => Dalgalar arası min. bar farkı.
    check_fib_retracements : bool
       => Wave2, Wave4'ün retracement kontrolü (0.382..0.618 ± fib_tolerance).
    df : pd.DataFrame
       => ATR, kapanış vb. ek teyit için (opsiyonel).

    Dönüş:
    ------
    dict => {
       "found": bool, 
       "trend": "UP"/"DOWN"/None,
       "pivots": [(idx0, p0), (idx1,p1), (idx2,p2), (idx3,p3), (idx4,p4)], 
       "check_msgs": list,
       "abc": bool or None,
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

    # 1) En az 5 pivot?
    if len(wave) < wave_min_bars:
        result["check_msgs"].append("Not enough pivots")
        return result

    # 2) Son 5 pivot (basit yaklaşım)
    last5 = wave[-5:]
    types = [p[2] for p in last5]

    up_pattern   = [+1, -1, +1, -1, +1]
    down_pattern = [-1, +1, -1, +1, -1]

    if types == up_pattern:
        trend = "UP"
    elif check_alt_scenarios and types == down_pattern:
        trend = "DOWN"
    else:
        result["check_msgs"].append("Pivot types pattern fail")
        return result

    result["trend"] = trend

    # 3) Pivot fiyatlarını al
    p0i, p0p, _ = last5[0]
    p1i, p1p, _ = last5[1]
    p2i, p2p, _ = last5[2]
    p3i, p3p, _ = last5[3]
    p4i, p4p, _ = last5[4]
    result["pivots"] = [(p0i,p0p),(p1i,p1p),(p2i,p2p),(p3i,p3p),(p4i,p4p)]

    def wave_len(a,b):
        return abs(b-a)

    w1 = wave_len(p0p, p1p)
    w2 = wave_len(p1p, p2p)
    w3 = wave_len(p2p, p3p)
    w4 = wave_len(p3p, p4p)

    # 4) Bar mesafesi
    if min_bar_distance>0:
        d1 = p1i - p0i
        d2 = p2i - p1i
        d3 = p3i - p2i
        d4 = p4i - p3i
        if any(d< min_bar_distance for d in [d1,d2,d3,d4]):
            result["check_msgs"].append("Bar distance fail")
            return result

    # 5) 3. dalga >= rule_3rdwave_min_percent * 1. dalga
    if w3 < (rule_3rdwave_min_percent * w1):
        result["check_msgs"].append("3rd wave not long enough")
        return result

    # 6) 4. dalga overlap
    if not allow_4th_overlap:
        if trend=="UP" and p4p< p1p:
            result["check_msgs"].append("4th wave overlap fail (UP)")
            return result
        if trend=="DOWN" and p4p> p1p:
            result["check_msgs"].append("4th wave overlap fail (DOWN)")
            return result

    # 7) Fib retracements (wave2, wave4)
    if check_fib_retracements:
        # wave2 genellikle wave1'in 0.382..0.618'i
        # wave4 => wave3'ün 0.382..0.618'i
        # Tolerans => fib_tolerance
        w2r = w2/(w1+1e-9)
        w4r = w4/(w3+1e-9)
        typical_w2_min= 0.382 - fib_tolerance
        typical_w2_max= 0.618 + fib_tolerance
        if not (typical_w2_min<= w2r<= typical_w2_max):
            result["check_msgs"].append("Wave2 fib retrace fail")
        typical_w4_min= 0.382 - fib_tolerance
        typical_w4_max= 0.618 + fib_tolerance
        if not (typical_w4_min<= w4r<= typical_w4_max):
            result["check_msgs"].append("Wave4 fib retrace fail")

    # 8) 5. dalga extension => wave5 => p3->p4
    wave5_len = w4
    wave5_ratio= wave5_len/(w1+ 1e-9)
    if wave5_ratio>= rule_5thwave_ext_range[0] and wave5_ratio<= rule_5thwave_ext_range[1]:
        result["extended_5th"] = True

    # 9) ABC correction
    if extended_waves and check_abc_correction and len(wave)>=8:
        # p5 => wave[-5], A => wave[-3], B => wave[-2], C => wave[-1]
        maybe_abc= wave[-3:]
        abc_types= [p[2] for p in maybe_abc]
        if trend=="UP":
            # A=-1, B=+1, C=-1
            if abc_types== [-1,+1,-1]:
                result["abc"]= True
        else:
            # down => +1,-1,+1
            if abc_types== [+1,-1,+1]:
                result["abc"]= True

    # 10) Sonuç => found=True
    result["found"]= True
    return result

