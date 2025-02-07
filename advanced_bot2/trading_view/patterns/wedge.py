# patterns/triangle_wedge.py
# patterns/harmonics.py
import pandas as pd
from core.logging_setup import log


############################
# WEDGE (Rising, Falling)
############################

def detect_wedge_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame:str="1m",
    wedge_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
)-> dict:
    result={
      "pattern":"wedge",
      "found":False,
      "wedge_type":None,
      "breakout":False,
      "breakout_line":None,
      "retest_info":None,
      "msgs":[]
    }
    wave= build_zigzag_wave(pivots)
    if len(wave)<5:
        result["msgs"].append("Not enough pivot for wedge (need>=5).")
        return result

    last5= wave[-5:]
    types=[p[2] for p in last5]
    rising_pat= [+1,-1,+1,-1,+1]
    falling_pat=[-1,+1,-1,+1,-1]
    if types==rising_pat:
        wedge_type="rising"
    elif types==falling_pat:
        wedge_type="falling"
    else:
        result["msgs"].append("Pivot pattern not matching rising/falling wedge.")
        return result

    x1,y1= last5[0][0], last5[0][1]
    x3,y3= last5[2][0], last5[2][1]
    x5,y5= last5[4][0], last5[4][1]
    slope_top= (y5-y1)/((x5-x1)+1e-9)

    x2,y2= last5[1][0], last5[1][1]
    x4,y4= last5[3][0], last5[3][1]
    slope_bot= (y4-y2)/((x4-x2)+1e-9)

    if wedge_type=="rising":
        if (slope_top<0) or (slope_bot<0):
            result["msgs"].append("Expected positive slopes for rising wedge.")
            return result
        if not (slope_bot> slope_top):
            result["msgs"].append("slope(2->4)<= slope(1->3)? => not wedge shape.")
            return result
    else:
        if (slope_top>0) or (slope_bot>0):
            result["msgs"].append("Expected negative slopes for falling wedge.")
            return result
        if not (slope_bot> slope_top):
            result["msgs"].append("Dip slope <= top slope => not wedge shape.")
            return result

    ratio= abs(slope_bot- slope_top)/(abs(slope_top)+1e-9)
    if ratio< wedge_tolerance:
        result["msgs"].append(f"Wedge slope difference ratio {ratio:.3f}< tolerance => might be channel.")

    df_len= len(df)
    brk=False
    close_col= get_col_name("Close", time_frame)
    if check_breakout and close_col in df.columns and df_len>0:
        last_close= df[close_col].iloc[-1]
        m_,b_= line_equation(x2,y2,x4,y4)  # alt çizgi
        if wedge_type=="rising":
            if m_ is not None:
                last_i= df_len-1
                line_y= m_* last_i + b_
                if last_close< line_y:
                    brk= True
        else:
            # falling => üst çizgi
            m2,b2= line_equation(x1,y1,x3,y3)
            if m2 is not None:
                last_i= df_len-1
                line_y2= m2* last_i+ b2
                if last_close> line_y2:
                    brk= True

    if brk:
        result["breakout"]=True
        if wedge_type=="rising":
            result["breakout_line"]= ((x2,y2),(x4,y4))
        else:
            result["breakout_line"]= ((x1,y1),(x3,y3))

    result["found"]= True
    result["wedge_type"]= wedge_type

    if check_retest and brk and result["breakout_line"]:
        (ixA,pxA),(ixB,pxB)= result["breakout_line"]
        mW,bW= line_equation(ixA,pxA,ixB,pxB)
        if mW is not None:
            retest_done=False
            retest_bar=None
            for i in range(ixB+1, df_len):
                c= df[close_col].iloc[i]
                line_y= mW*i + bW
                diff_perc= abs(c-line_y)/(abs(line_y)+1e-9)
                if diff_perc<= retest_tolerance:
                    retest_done= True
                    retest_bar= i
                    break
            result["retest_info"]= {
                "retest_done": retest_done,
                "retest_bar": retest_bar
            }

    return result


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
def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"
