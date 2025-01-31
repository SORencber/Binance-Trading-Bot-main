# patterns/triangle_wedge.py
# patterns/harmonics.py
import pandas as pd
from core.logging_setup import log

def line_eq(x1, y1, x2, y2):
    if (x2 - x1) == 0:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return m, b

def line_intersection(m1,b1, m2,b2):
    if m1 == m2:  # parallel
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
def detect_triangle(
    wave,
    time_frame="1m",
    triangle_tolerance=0.02,
    check_breakout=True,
    check_retest=False,
    triangle_types=["ascending","descending","symmetrical"],
    df=None
) -> bool:
    """
    Üçgen formasyonu dedektörü.
    """
    if len(wave)<4:
        return False

    last4 = wave[-4:]
    p1,p2,p3,p4 = last4
    types= [p[2] for p in last4]

    up_zigzag   = [+1,-1,+1,-1]
    down_zigzag = [-1,+1,-1,+1]

    if types not in [up_zigzag, down_zigzag]:
        return False

    if types== up_zigzag:
        # tepe => p1,p3, dip=>p2,p4
        x1,y1= p1[0], p1[1]
        x3,y3= p3[0], p3[1]
        x2,y2= p2[0], p2[1]
        x4,y4= p4[0], p4[1]
    else:
        # tepe => p2,p4, dip=>p1,p3
        x1,y1= p2[0], p2[1]
        x3,y3= p4[0], p4[1]
        x2,y2= p1[0], p1[1]
        x4,y4= p3[0], p3[1]

    def line_eq(x1,y1,x2,y2):
        if (x2-x1)==0:
            return None,None
        m= (y2-y1)/(x2-x1)
        b= y1 - m*x1
        return m,b

    m_top,b_top= line_eq(x1,y1, x3,y3)
    m_bot,b_bot= line_eq(x2,y2, x4,y4)
    if m_top is None or m_bot is None:
        return False

    # slope check ...
    def is_flat(m):
        return abs(m)< triangle_tolerance

    top_type, bot_type= None,None
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

    form=None
    if top_type=="flat" and bot_type=="rising" and "ascending" in triangle_types:
        form="ascending"
    elif top_type=="falling" and bot_type=="flat" and "descending" in triangle_types:
        form="descending"
    elif top_type=="falling" and bot_type=="rising" and "symmetrical" in triangle_types:
        form="symmetrical"
    else:
        return False

    if check_breakout and df is not None and len(df)>0:
        close_col= get_col_name("Close", time_frame)
        if close_col not in df.columns:
            return False
        last_price= df[close_col].iloc[-1]
        last_i= len(df)-1

        # ...
        if form=="ascending":
            # check top line
            line_y_top= m_top*last_i + b_top
            if last_price<= line_y_top:
                return False
        elif form=="descending":
            line_y_bot= m_bot*last_i + b_bot
            if last_price>= line_y_bot:
                return False
        else: # symmetrical
            line_y_top= m_top*last_i + b_top
            line_y_bot= m_bot*last_i + b_bot
            if line_y_bot<= last_price<= line_y_top:
                return False

    return True


def detect_wedge(
    wave,
    time_frame="1m",
    wedge_tolerance=0.02,
    check_breakout=True,
    check_retest=False,
    df=None
) -> bool:
    """
    Wedge formasyonu dedektörü.
    """
    if len(wave)<5:
        return False

    last5= wave[-5:]
    ptypes= [p[2] for p in last5]
    rising_pat  = [+1,-1,+1,-1,+1]
    falling_pat = [-1,+1,-1,+1,-1]

    if ptypes== rising_pat:
        # ... rising wedge
        x1,y1= last5[0][0], last5[0][1]
        x3,y3= last5[2][0], last5[2][1]
        x5,y5= last5[4][0], last5[4][1]
        slope_top= (y5-y1)/((x5-x1)+1e-9)

        x2,y2= last5[1][0], last5[1][1]
        x4,y4= last5[3][0], last5[3][1]
        slope_bot= (y4-y2)/((x4-x2)+1e-9)

        if slope_top<0 or slope_bot<0:
            return False
        if not (slope_bot> slope_top):
            return False

        ratio= abs(slope_bot- slope_top)/(abs(slope_top)+1e-9)
        if ratio< wedge_tolerance:
            return False

        # breakout => asagı
        if check_breakout and df is not None:
            close_col= get_col_name("Close", time_frame)
            if close_col not in df.columns:
                return False
            last_close= df[close_col].iloc[-1]
            m_bot,b_bot= line_eq(x2,y2, x4,y4)
            if m_bot is not None:
                line_val= m_bot*(len(df)-1) + b_bot
                if last_close>= line_val:
                    return False

        return True

    elif ptypes== falling_pat:
        # falling wedge
        x1,y1= last5[0][0], last5[0][1]
        x3,y3= last5[2][0], last5[2][1]
        x5,y5= last5[4][0], last5[4][1]
        slope_bot= (y5-y1)/((x5-x1)+1e-9)

        x2,y2= last5[1][0], last5[1][1]
        x4,y4= last5[3][0], last5[3][1]
        slope_top= (y4-y2)/((x4-x2)+1e-9)

        if slope_top>0 or slope_bot>0:
            return False
        if not (slope_bot> slope_top):
            return False

        ratio= abs(slope_bot- slope_top)/(abs(slope_top)+1e-9)
        if ratio< wedge_tolerance:
            return False

        if check_breakout and df is not None:
            close_col= get_col_name("Close", time_frame)
            if close_col not in df.columns:
                return False
            last_close= df[close_col].iloc[-1]
            m_top,b_top= line_eq(x2,y2, x4,y4)
            if m_top is not None:
                line_val= m_top*(len(df)-1)+ b_top
                if last_close<= line_val:
                    return False

        return True

    else:
        return False
