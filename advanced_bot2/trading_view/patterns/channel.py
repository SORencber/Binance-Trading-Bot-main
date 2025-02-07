
import pandas as pd
############################
# CHANNEL (Advanced)
############################

def detect_channel_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str="1m",
    parallel_thresh: float=0.02,
    min_top_pivots: int=3,
    min_bot_pivots: int=3,
    max_iter: int=10,
    check_retest: bool=False,
    retest_tolerance: float=0.01
)-> dict:
    import numpy as np
    result={
        "pattern":"channel",
        "found":False,
        "channel_type":None,
        "upper_line_points":[],
        "lower_line_points":[],
        "upper_line_eq":None,
        "lower_line_eq":None,
        "breakout":False,
        "breakout_line":None,
        "retest_info":None,
        "msgs":[]
    }
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        result["msgs"].append("No close col found.")
        return result
    if not pivots or len(pivots)==0:
        result["msgs"].append("No pivots given.")
        return result

    top_piv= [p for p in pivots if p[2]== +1]
    bot_piv= [p for p in pivots if p[2]== -1]
    if len(top_piv)< min_top_pivots or len(bot_piv)< min_bot_pivots:
        result["msgs"].append("Not enough top/bottom pivots.")
        return result

    def best_fit_line(pivot_list):
        xs= np.array([p[0] for p in pivot_list], dtype=float)
        ys= np.array([p[1] for p in pivot_list], dtype=float)
        if len(xs)<2:
            return (0.0, float(ys.mean()))
        m= (np.mean(xs*ys)- np.mean(xs)* np.mean(ys)) / \
           (np.mean(xs**2)- (np.mean(xs))**2+1e-9)
        b= np.mean(ys)- m*np.mean(xs)
        return (m,b)

    m_top,b_top= best_fit_line(top_piv)
    m_bot,b_bot= best_fit_line(bot_piv)
    slope_diff= abs(m_top- m_bot)/(abs(m_top)+1e-9)
    if slope_diff> parallel_thresh:
        msg= f"Slope diff {slope_diff:.3f}>threshold => not channel."
        result["msgs"].append(msg)
        return result

    result["found"]= True
    result["upper_line_points"]= top_piv
    result["lower_line_points"]= bot_piv
    result["upper_line_eq"]= (m_top,b_top)
    result["lower_line_eq"]= (m_bot,b_bot)

    avg_slope= (m_top+ m_bot)/2
    if abs(avg_slope)<0.01:
        result["channel_type"]="horizontal"
    elif avg_slope>0:
        result["channel_type"]="ascending"
    else:
        result["channel_type"]="descending"

    last_i= len(df)-1
    last_close= df[close_col].iloc[-1]
    top_line_val= m_top* last_i+ b_top
    bot_line_val= m_bot* last_i+ b_bot
    breakout_up= (last_close> top_line_val)
    breakout_down= (last_close< bot_line_val)
    if breakout_up or breakout_down:
        result["breakout"]= True

        def line_points_from_regression(m,b, pivot_list):
            xvals=[p[0] for p in pivot_list]
            x_min,x_max= min(xvals), max(xvals)
            y_min= m*x_min+ b
            y_max= m*x_max+ b
            return ((x_min,y_min),(x_max,y_max))

        if breakout_up:
            line2d= line_points_from_regression(m_top,b_top, top_piv)
            result["breakout_line"]= line2d
        else:
            line2d= line_points_from_regression(m_bot,b_bot, bot_piv)
            result["breakout_line"]= line2d

        if check_retest and result["breakout_line"]:
            (ixA,pxA),(ixB,pxB)= result["breakout_line"]
            mC,bC= line_equation(ixA,pxA,ixB,pxB)
            if mC is not None:
                retest_done=False
                retest_bar=None
                for i in range(ixB+1, len(df)):
                    c= df[close_col].iloc[i]
                    line_y= mC*i+ bC
                    diff_perc= abs(c-line_y)/(abs(line_y)+1e-9)
                    if diff_perc<= retest_tolerance:
                        retest_done= True
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
