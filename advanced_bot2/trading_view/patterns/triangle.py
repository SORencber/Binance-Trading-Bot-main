
############################
# TRIANGLE (Asc, Desc, Sym)
############################
import pandas as pd
def detect_triangle_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str="1m",
    triangle_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01,
    triangle_types: list=None
):
    result={
      "pattern": "triangle",
      "found": False,
      "triangle_type": None,
      "breakout": False,
      "breakout_line": None,
      "retest_info": None,
      "msgs": []
    }
    wave= build_zigzag_wave(pivots)
    if triangle_types is None:
        triangle_types=["ascending","descending","symmetrical"]
    if len(wave)<4:
        result["msgs"].append("Not enough pivots for triangle (need >=4).")
        return result

    last4= wave[-4:]
    p1,p2,p3,p4= last4
    t_list=[p[2] for p in last4]
    up_zig= [+1,-1,+1,-1]
    down_zig= [-1,+1,-1,+1]
    if t_list not in [up_zig, down_zig]:
        result["msgs"].append("Zigzag pattern not matching triangle requirement.")
        return result

    if t_list==up_zig:
        x1,y1=p1[0],p1[1]
        x3,y3=p3[0],p3[1]
        x2,y2=p2[0],p2[1]
        x4,y4=p4[0],p4[1]
    else:
        # Ters
        x1,y1=p2[0],p2[1]
        x3,y3=p4[0],p4[1]
        x2,y2=p1[0],p1[1]
        x4,y4=p3[0],p3[1]

    m_top,b_top=line_equation(x1,y1,x3,y3)
    m_bot,b_bot=line_equation(x2,y2,x4,y4)
    if m_top is None or m_bot is None:
        result["msgs"].append("Line top/bot eq fail => vertical slope.")
        return result

    def is_flat(m):
        return abs(m)< triangle_tolerance

    top_type= None
    bot_type= None
    if is_flat(m_top):
        top_type="flat"
    elif m_top>0:
        top_type="rising"
    else:
        top_type="falling"

    if is_flat(m_bot):
        bot_type="flat"
    elif m_bot>0:
        bot_type="rising"
    else:
        bot_type="falling"

    tri_type=None
    if top_type=="flat" and bot_type=="rising" and ("ascending" in triangle_types):
        tri_type="ascending"
    elif top_type=="falling" and bot_type=="flat" and ("descending" in triangle_types):
        tri_type="descending"
    elif top_type=="falling" and bot_type=="rising" and ("symmetrical" in triangle_types):
        tri_type="symmetrical"

    if not tri_type:
        result["msgs"].append("No matching triangle type.")
        return result

    result["found"]=True
    result["triangle_type"]= tri_type

    breakout=False
    close_col= get_col_name("Close", time_frame)
    if check_breakout and close_col in df.columns:
        last_close= df[close_col].iloc[-1]
        last_i= len(df)-1
        line_y_top= m_top* last_i + b_top
        line_y_bot= m_bot* last_i + b_bot
        if tri_type=="ascending":
            if last_close> line_y_top:
                breakout= True
                result["breakout_line"]= ((x1,y1),(x3,y3))
        elif tri_type=="descending":
            if last_close< line_y_bot:
                breakout= True
                result["breakout_line"]= ((x2,y2),(x4,y4))
        else:
            if (last_close> line_y_top) or (last_close< line_y_bot):
                breakout= True
                result["breakout_line"]= ((x1,y1),(x3,y3))
    result["breakout"]= breakout

    if check_retest and breakout and result["breakout_line"]:
        (xA,pA),(xB,pB)= result["breakout_line"]
        m_,b_= line_equation(xA,pA,xB,pB)
        if m_ is not None:
            retest_done=False
            retest_bar=None
            for i in range(xB+1, len(df)):
                c= df[close_col].iloc[i]
                line_y= m_*i + b_
                diff_perc= abs(c - line_y)/(abs(line_y)+1e-9)
                if diff_perc<= retest_tolerance:
                    retest_done=True
                    retest_bar=i
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

