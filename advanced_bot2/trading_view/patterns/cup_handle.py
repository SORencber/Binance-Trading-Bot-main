
############################
# CUP & HANDLE (ADV)
############################
import pandas as pd
def detect_cup_and_handle_advanced(
    df: pd.DataFrame,
    pivots=None,
    time_frame: str="1m",
    tolerance: float=0.02,
    volume_drop_check: bool=True,
    volume_drop_ratio: float=0.2,
    cup_min_bars: int=20,
    cup_max_bars: int=300,
    handle_ratio: float=0.3,
    handle_max_bars: int=50,
    close_above_rim: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
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
    close_col= get_col_name("Close", time_frame)
    volume_col= get_col_name("Volume", time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"Missing col: {close_col}")
        return result

    if pivots is None:
        # PivotScanner vs. => opsiyonel
        pass

    top_pivots= [p for p in pivots if p[2]== +1]
    bot_pivots= [p for p in pivots if p[2]== -1]
    if len(top_pivots)<2 or len(bot_pivots)<1:
        result["msgs"].append("Not enough top/dip pivots for Cup&Handle.")
        return result

    sorted_p= sorted(pivots, key=lambda x: x[0])
    best_cup= None
    for i in range(1, len(sorted_p)-1):
        if sorted_p[i][2]== -1:  # dip
            idxDip, pxDip= sorted_p[i][0], sorted_p[i][1]
            left_candidates= [tp for tp in sorted_p[:i] if tp[2]== +1]
            right_candidates= [tp for tp in sorted_p[i+1:] if tp[2]== +1]
            if (not left_candidates) or (not right_candidates):
                continue
            left_top= left_candidates[-1]
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
            best_cup= (left_top, (idxDip,pxDip), right_top, bars_cup)
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

    if volume_drop_check and volume_col in df.columns:
        idxL, pxL= l_top[0], l_top[1]
        idxR, pxR= r_top[0], r_top[1]
        cup_vol_series= df[volume_col].iloc[idxL : idxR+1]
        if len(cup_vol_series)>5:
            start_vol= cup_vol_series.iloc[0]
            min_vol= cup_vol_series.min()
            drop_percent= (start_vol- min_vol)/(start_vol+1e-9)
            result["cup_volume_drop"]= drop_percent
            if drop_percent< volume_drop_ratio:
                result["msgs"].append(f"Cup volume drop {drop_percent:.2f} < {volume_drop_ratio:.2f}")

    rim_idxL, rim_pxL= l_top[0], l_top[1]
    rim_idxR, rim_pxR= r_top[0], r_top[1]
    slope_rim= (rim_pxR- rim_pxL)/(rim_idxR- rim_idxL+1e-9)
    intercept= rim_pxL - slope_rim* rim_idxL

    dip_price= cup_dip[1]
    cup_height= ((l_top[1] + r_top[1])/2) - dip_price
    if cup_height<=0:
        return result

    handle_start= rim_idxR
    handle_end= min(rim_idxR+ handle_max_bars, len(df)-1)
    handle_found= False
    handle_top= None
    handle_bars= 0

    if handle_start< handle_end:
        seg= df[close_col].iloc[handle_start: handle_end+1]
        loc_max_val= seg.max()
        loc_max_idx= seg.idxmax()
        handle_bars= handle_end- handle_start
        handle_depth= ((r_top[1]+ l_top[1])/2)- loc_max_val
        if handle_depth>0:
            ratio= handle_depth/cup_height
            if ratio<= handle_ratio:
                handle_found= True
                handle_top= (loc_max_idx, loc_max_val)

    result["handle_found"]= handle_found
    result["handle_top"]= handle_top
    result["handle_bars"]= handle_bars

    # Cup&Handle onayı => rim break
    last_price= df[close_col].iloc[-1]
    last_i= len(df)-1
    rim_line_val= slope_rim* last_i + intercept
    if close_above_rim:
        if last_price> rim_line_val:
            result["confirmed"]= True
            result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))
    else:
        high_col= get_col_name("High", time_frame)
        if high_col in df.columns:
            last_high= df[high_col].iloc[-1]
            if last_high> rim_line_val:
                result["confirmed"]= True
                result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))

    # Retest
    if check_retest and result["confirmed"] and result["rim_line"]:
        retest_info= _check_retest_cup_handle(
            df, time_frame,
            rim_line= result["rim_line"],
            break_bar= last_i,
            tolerance= retest_tolerance
        )
        result["retest_info"]= retest_info

    return result

def _check_retest_cup_handle(
    df: pd.DataFrame,
    time_frame: str,
    rim_line: tuple,
    break_bar: int,
    tolerance: float=0.01
):
    (xL, pL), (xR, pR)= rim_line
    m,b= line_equation(xL, pL, xR, pR)
    if m is None:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    close_col= get_col_name("Close", time_frame)
    for i in range(break_bar+1, len(df)):
        c= df[close_col].iloc[i]
        line_y= m*i + b
        dist_ratio= abs(c- line_y)/(abs(line_y)+1e-9)
        if dist_ratio<= tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "distance_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None, "distance_ratio": None}
def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"
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
