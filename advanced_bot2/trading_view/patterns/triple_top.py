############################
# TRIPLE TOP 
############################
import pandas as pd
def detect_triple_top_advanced(
    df: pd.DataFrame,
    pivots,
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
    Triple Top => 3 tepe pivot, birbirine yakın (tolerance).
    """
    top_pivots = [p for p in pivots if p[2] == +1]
    if len(top_pivots) < 3:
        return []

    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume", time_frame)
    results = []
    i = 0
    while i < len(top_pivots) - 2:
        t1 = top_pivots[i]
        t2 = top_pivots[i+1]
        t3 = top_pivots[i+2]

        idx1, price1 = t1[0], t1[1]
        idx2, price2 = t2[0], t2[1]
        idx3, price3 = t3[0], t3[1]

        bar_diff_12 = idx2 - idx1
        bar_diff_23 = idx3 - idx2
        if bar_diff_12 < min_distance_bars or bar_diff_23 < min_distance_bars:
            i+=1
            continue

        avgp = (price1 + price2 + price3)/3
        pdiff_1 = abs(price1 - avgp)/(avgp+1e-9)
        pdiff_2 = abs(price2 - avgp)/(avgp+1e-9)
        pdiff_3 = abs(price3 - avgp)/(avgp+1e-9)
        if any(p > tolerance for p in [pdiff_1, pdiff_2, pdiff_3]):
            i+=1
            continue

        vol_ok = True
        msgs = []
        if volume_check and volume_col in df.columns:
            vol1 = df[volume_col].iloc[idx1]
            vol2 = df[volume_col].iloc[idx2]
            vol3 = df[volume_col].iloc[idx3]
            mean_top_vol = (vol1 + vol2)/2
            if vol3 > (mean_top_vol* volume_col_factor):
                vol_ok = False
                msgs.append(f"3rd top volume not lower => vol3={vol3:.2f}, mean12={mean_top_vol:.2f}")

        # Neckline
        seg_min_pivots = [p for p in pivots if p[2] == -1 and p[0]> idx1 and p[0]< idx3]
        neckline = None
        if seg_min_pivots:
            sorted_dips = sorted(seg_min_pivots, key=lambda x: x[1])
            neckline = (sorted_dips[0][0], sorted_dips[0][1])
        else:
            msgs.append("No local dip pivot found for neckline.")

        conf=False
        retest_data=None
        if neckline_break and neckline is not None and close_col in df.columns:
            neck_idx, neck_prc= neckline
            last_close = df[close_col].iloc[-1]
            if last_close< neck_prc:
                conf = True
                if check_retest:
                    retest_data= _check_retest_triple_top(
                        df, time_frame,
                        neckline_price= neck_prc,
                        confirm_bar= len(df)-1,
                        retest_tolerance= retest_tolerance
                    )
            else:
                msgs.append("Neckline not broken => not confirmed")

        pattern_info = {
            "pattern": "triple_top",
            "tops": [(idx1,price1),(idx2,price2),(idx3,price3)],
            "neckline": neckline,
            "confirmed": conf,
            "volume_check": vol_ok,
            "msgs": msgs,
            "retest_info": retest_data
        }
        if vol_ok:
            results.append(pattern_info)

        i+=1

    return results

def _check_retest_triple_top(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float = 0.02
):
    close_col = get_col_name("Close", time_frame)
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


def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

