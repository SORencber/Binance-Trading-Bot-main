# patterns/harmonics.py
import pandas as pd
from core.logging_setup import log

############################
# HARMONIC PATTERNS
############################

def detect_harmonic_pattern_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    fib_tolerance: float=0.02,
    patterns: list = None,
    check_volume: bool=False,
    volume_factor: float=1.3,
    check_retest: bool= False,
    retest_tolerance: float=0.01
)-> dict:
    """
    Harmonic Pattern (X,A,B,C,D).
    """
    if patterns is None:
        patterns= ["gartley","bat","crab","butterfly","shark","cipher"]
    result= {
      "pattern": "harmonic",
      "found": False,
      "pattern_name": None,
      "xabc": [],
      "msgs": [],
      "retest_info": None
    }
    wave= build_zigzag_wave(pivots)
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
    result["xabc"]=[X,A,B,C,D]

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
        mn,mx= rng
        if abs(mn-mx)<1e-9:
            return abs(val- mn)<= abs(mn)*tol
        else:
            low_= mn- abs(mn)* tol
            high_= mx+ abs(mx)* tol
            return low_<= val<= high_

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

        volume_col= get_col_name("Volume", time_frame)
        if check_volume and volume_col in df.columns and idxD<len(df):
            vol_now= df[volume_col].iloc[idxD]
            prepare_volume_ma(df, time_frame, period=20)
            ma_col= f"Volume_MA_20_{time_frame}"
            if ma_col in df.columns:
                v_mean= df[ma_col].iloc[idxD]
                if (v_mean>0) and (vol_now> volume_factor*v_mean):
                    # "Güçlü hacim"
                    pass

        if check_retest:
            close_col= get_col_name("Close", time_frame)
            if close_col in df.columns:
                retest_done= False
                retest_bar = None
                for i in range(idxD+1, len(df)):
                    c= df[close_col].iloc[i]
                    dist_ratio = abs(c - pxD)/(abs(pxD)+1e-9)
                    if dist_ratio <= retest_tolerance:
                        retest_done= True
                        retest_bar= i
                        break
                if retest_done:
                    result["retest_info"] = {
                        "retest_done": True,
                        "retest_bar": retest_bar,
                        "retest_price": df[close_col].iloc[retest_bar]
                    }
                else:
                    result["retest_info"] = {"retest_done": False}
    else:
        result["msgs"].append("No harmonic pattern matched.")
    return result

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def prepare_volume_ma(df: pd.DataFrame, time_frame: str="1m", period: int=20):
    """
    Volume_MA_20_{time_frame} hesaplayıp ekler.
    """
    vol_col = get_col_name("Volume", time_frame)
    ma_col  = f"Volume_MA_{period}_{time_frame}"
    if (vol_col in df.columns) and (ma_col not in df.columns):
        df[ma_col] = df[vol_col].rolling(period).mean()

############################
# ZIGZAG HELPER
############################

def build_zigzag_wave(pivots):
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda x: x[0])
    wave = [sorted_p[0]]
    for i in range(1, len(sorted_p)):
        curr = sorted_p[i]
        prev = wave[-1]
        if curr[2] == prev[2]:
            # aynı tip => pivotu güncelle
            if curr[2] == +1:
                if curr[1] > prev[1]:
                    wave[-1] = curr
            else:
                if curr[1] < prev[1]:
                    wave[-1] = curr
        else:
            wave.append(curr)
    return wave

