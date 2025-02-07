############################
# WOLFE WAVE ADV
############################
import pandas as pd
def detect_wolfe_wave_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    price_tolerance: float = 0.03,
    strict_lines: bool = False,
    breakout_confirm: bool = True,
    line_projection_check: bool = True,
    check_2_4_slope: bool = True,
    check_1_4_intersection_time: bool = True,
    check_time_symmetry: bool = True,
    max_time_ratio: float = 0.3,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
):
    result= {
      "pattern": "wolfe",
      "found": False,
      "msgs": [],
      "breakout": False,
      "intersection": None,
      "time_symmetry_ok": True,
      "sweet_zone": None,
      "wolfe_line": None,
      "retest_info": None
    }
    wave = build_zigzag_wave(pivots)
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
    diff_slope= abs(m35- m13)/(abs(m13)+1e-9)
    if diff_slope> price_tolerance:
        result["msgs"].append(f"Slope difference too big => {diff_slope:.3f}")
    if check_2_4_slope:
        m24,b24= line_equation(x2,y2, x4,y4)
        if strict_lines and (m24 is not None):
            slope_diff= abs(m24- m13)/(abs(m13)+1e-9)
            if slope_diff>0.3:
                result["msgs"].append("Line(2->4) slope differs from line(1->3).")

    # sweet zone
    m24_,b24_= line_equation(x2,y2, x4,y4)
    if m24_ is not None:
        line13_y5= m13*x5+ b13
        line24_y5= m24_*x5+ b24_
        low_  = min(line13_y5, line24_y5)
        high_ = max(line13_y5, line24_y5)
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
        m14,b14= line_equation(x1,y1,x4,y4)
        m23,b23= line_equation(x2,y2,x3,y3)
        if (m14 is not None) and (m23 is not None):
            ix,iy= line_intersection(m14,b14, m23,b23)
            if ix is not None:
                result["intersection"]= (ix, iy)
                if check_1_4_intersection_time and ix< x5:
                    result["msgs"].append("Intersection(1->4 & 2->3) < w5 => degrade")

    if breakout_confirm:
        close_col= get_col_name("Close", time_frame)
        if close_col in df.columns:
            last_close= df[close_col].iloc[-1]
            m14,b14= line_equation(x1,y1, x4,y4)  # (1->4) hattı
           
            if m14 is not None:
                last_i= len(df)-1
                line_y= m14* last_i + b14
                #print(f"Bearish check => last_close={last_close:.4f}, line_y={line_y:.4f}, slope={m14:.4f}, x1={x1}, x4={x4}")

                if m14 > 0:
                    # Bullish Wolfe => Üst tarafa kırılırsa breakout
                    if last_close > line_y:
                        result["breakout"] = True
                        result["wolfe_line"] = ((x1,y1),(x4,y4))
                        result["w5"] = (x5,y5)
                        result["direction"] = "LONG"  # <-- Ekleyebilirsiniz
                else:
                    # Bearish Wolfe => Alt tarafa kırılırsa breakout
                    if last_close < line_y:

                        result["breakout"] = True
                        result["wolfe_line"] = ((x1,y1),(x4,y4))
                        result["w5"] = (x5,y5)
                        result["direction"] = "SHORT"  # <-- Ekleyebilirsiniz

    result["found"]= True


    if check_retest and result["breakout"] and result["wolfe_line"]:
        (ixA,pxA),(ixB,pxB)= result["wolfe_line"]
        m_,b_= line_equation(ixA, pxA, ixB, pxB)
        if m_ is not None and df is not None:
            close_col= get_col_name("Close", time_frame)
            last_i= len(df)-1
            retest_done= False
            retest_bar= None
            for i in range(last_i+1, len(df)):
                c= df[close_col].iloc[i]
                line_val= m_* i + b_
                dist_perc= abs(c - line_val)/(abs(line_val)+1e-9)
                if dist_perc<= retest_tolerance:
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

def line_intersection(m1, b1, m2, b2):
    """
    Intersection (x, y) of lines y=m1*x+b1 and y=m2*x+b2.
    If parallel or invalid => returns (None, None).
    """
    if (m1 is None) or (m2 is None):
        return None, None
    if abs(m1 - m2) < 1e-15:  # parallel
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y

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

