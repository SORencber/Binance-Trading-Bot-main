############################
# FLAG / PENNANT
############################
import pandas as pd 
def detect_flag_pennant_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str="1m",
    min_flagpole_bars: int=15,
    impulse_pct: float=0.05,
    max_cons_bars: int=40,
    pivot_channel_tolerance: float=0.02,
    pivot_triangle_tolerance: float=0.02,
    require_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
) -> dict:
    result={
        "pattern":"flag_pennant",
        "found":False,
        "direction":None,
        "pattern_type":None,
        "consolidation_pivots":[],
        "upper_line":None,
        "lower_line":None,
        "confirmed":False,
        "breakout_bar":None,
        "breakout_line":None,
        "retest_info":None,
        "msgs":[]
    }
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"Missing {close_col}")
        return result
    n=len(df)
    if n< min_flagpole_bars:
        result["msgs"].append("Not enough bars for flagpole check.")
        return result

    start_i= n- min_flagpole_bars
    price_start= df[close_col].iloc[start_i]
    price_end= df[close_col].iloc[-1]
    pct_chg= (price_end- price_start)/(price_start+1e-9)
    if abs(pct_chg)< impulse_pct:
        result["msgs"].append(f"No strong impulse (< {impulse_pct*100}%).")
        return result

    direction= "bull" if (pct_chg>0) else "bear"
    result["direction"]= direction

    cons_start= n - min_flagpole_bars
    cons_end= min(n-1, cons_start+ max_cons_bars)
    if cons_end<= cons_start:
        result["msgs"].append("Consolidation not enough bars.")
        return result

    cons_piv= [p for p in pivots if (p[0]>= cons_start and p[0]<= cons_end)]
    result["consolidation_pivots"]= cons_piv

    top_pivs= [p for p in cons_piv if p[2]==+1]
    bot_pivs= [p for p in cons_piv if p[2]==-1]
    if len(top_pivs)<2 or len(bot_pivs)<2:
        result["msgs"].append("Not enough top/bottom pivots => can't form mini-channel or triangle.")
        return result

    top_sorted= sorted(top_pivs, key=lambda x: x[0])
    bot_sorted= sorted(bot_pivs, key=lambda x: x[0])
    up1,up2= top_sorted[0], top_sorted[1]
    dn1,dn2= bot_sorted[0], bot_sorted[1]

    def slope(x1,y1,x2,y2):
        if (x2-x1)==0: return None
        return (y2-y1)/(x2-x1)
    s_up= slope(up1[0], up1[1], up2[0], up2[1])
    s_dn= slope(dn1[0], dn1[1], dn2[0], dn2[1])
    if (s_up is None) or (s_dn is None):
        result["msgs"].append("Channel lines vertical => cannot form slope.")
        return result

    slope_diff= abs(s_up- s_dn)/(abs(s_up)+1e-9)
    is_parallel= (slope_diff< pivot_channel_tolerance)
    is_opposite_sign= (s_up* s_dn< 0)

    upper_line= ((up1[0], up1[1]),(up2[0], up2[1]))
    lower_line= ((dn1[0], dn1[1]),(dn2[0], dn2[1]))
    result["upper_line"]= upper_line
    result["lower_line"]= lower_line

    pattern_type=None
    if is_parallel:
        pattern_type= "flag"
    elif is_opposite_sign and slope_diff> pivot_triangle_tolerance:
        pattern_type= "pennant"

    if not pattern_type:
        result["msgs"].append("No definitive mini-flag or mini-pennant.")
        return result

    result["pattern_type"]= pattern_type
    result["found"]= True

    if not require_breakout:
        return result

    last_i= n-1
    last_close= df[close_col].iloc[-1]
    def line_val(p1,p2,x):
        if (p2[0]- p1[0])==0:
            return p1[1]
        m= (p2[1]- p1[1])/(p2[0]- p1[0])
        b= p1[1] - m*p1[0]
        return m*x+ b

    up_line_last= line_val(up1, up2, last_i)
    dn_line_last= line_val(dn1, dn2, last_i)
    conf= False
    brk_bar= None
    if direction=="bull":
        if last_close> up_line_last:
            conf= True
            brk_bar= last_i
    else:
        if last_close< dn_line_last:
            conf= True
            brk_bar= last_i

    result["confirmed"]= conf
    result["breakout_bar"]= brk_bar
    if conf:
        if direction=="bull":
            result["breakout_line"]= upper_line
        else:
            result["breakout_line"]= lower_line
        if check_retest and result["breakout_line"]:
            (ixA,pxA),(ixB,pxB)= result["breakout_line"]
            mF,bF= line_equation(ixA,pxA,ixB,pxB)
            if mF is not None:
                retest_done=False
                retest_bar=None
                for i in range(ixB+1, n):
                    c= df[close_col].iloc[i]
                    line_y= mF*i+ bF
                    diff_perc= abs(c- line_y)/(abs(line_y)+1e-9)
                    if diff_perc<= retest_tolerance:
                        retest_done=True
                        retest_bar= i
                        break
                result["retest_info"]={
                    "retest_done": retest_done,
                    "retest_bar": retest_bar
                }
    return result

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
