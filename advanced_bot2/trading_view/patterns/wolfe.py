# patterns/wolfe.py
import pandas as pd
from core.logging_setup import log

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
    return f"{base_col}_{time_frame}"
def detect_wolfe_wave_advanced(
    wave,
    time_frame: str = "1m",
    price_tolerance: float = 0.03,
    strict_lines: bool = False,
    breakout_confirm: bool = True,
    line_projection_check: bool = True,
    check_2_4_slope: bool = True,
    check_1_4_intersection_time: bool = True,
    check_time_symmetry: bool = True,
        max_time_ratio: float = 0.3,  # <-- Yeni parametre

    df=None
) -> dict:
    """
    Wolfe Wave dedektörü (EPA line, sweet zone, time symmetry, vb.) 
    Kuralları 'harfiyen' check etse de, bir kural fail ettiğinde 
    'return' etmeyip msgs'e ekliyor. 
    Sonuçta 'found=True' olsa bile msgs içinde hangi kuralların 
    kısmen ihlal edildiğini görebilirsiniz.

    wave: list of (bar_index, price, pivot_type) => en az 5 pivot.
    Dönen dict:
      {
        "found": bool,
        "msgs": list[str],
        "breakout": bool,
        "intersection": (None or (ix, iy)),
        "epa_line": { ... },
        "sweet_zone": (low,high),
        "time_symmetry_ok": bool
      }
    """

    result = {
        "found": False,
        "msgs": [],
        "breakout": False,
        "intersection": None,
        "epa_line": None,
        "sweet_zone": None,
        "time_symmetry_ok": True
    }

    # 1) En az 5 pivot
    if len(wave) < 5:
        result["msgs"].append("Not enough pivots (need >=5).")
        return result

    # w1..w5 => son 5 pivot
    w1 = wave[-5]
    w2 = wave[-4]
    w3 = wave[-3]
    w4 = wave[-2]
    w5 = wave[-1]

    x1, y1, t1 = w1
    x2, y2, t2 = w2
    x3, y3, t3 = w3
    x4, y4, t4 = w4
    x5, y5, t5 = w5

    def line_eq(xA,yA, xB,yB):
        if (xB-xA) == 0:
            return None, None
        m = (yB-yA)/(xB-xA)
        b = yA - m*xA
        return m,b

    m13,b13 = line_eq(x1,y1, x3,y3)
    m35,b35 = line_eq(x3,y3, x5,y5)

    if (m13 is None) or (m35 is None):
        result["msgs"].append("Vertical slope on (1->3) or (3->5)? fails.")
        # Yine de wave = degrade => found=False, return
        return result

    # slope fark
    diff_slope = abs(m35 - m13)/(abs(m13)+1e-9)
    if diff_slope > price_tolerance:
        result["msgs"].append(f"Slope(1->3 vs 3->5) difference too big: {diff_slope:.3f}")

    # check_2_4_slope => line(2->4)
    if check_2_4_slope:
        m24,b24= line_eq(x2,y2, x4,y4)
        if (m24 is not None) and (m13 is not None) and strict_lines:
            slope_diff= abs(m24 - m13)/(abs(m13)+1e-9)
            if slope_diff> 0.3:
                result["msgs"].append(f"Line(2->4) slope differs from (1->3) by {slope_diff:.3f} (strict fail)")

    # sweet zone => w5 => line(1->3)(x5) & line(2->4)(x5)
    m24_, b24_ = line_eq(x2,y2, x4,y4)
    if (m24_ is not None) and (m13 is not None):
        line13_y5 = m13*x5 + b13
        line24_y5 = m24_*x5 + b24_
        low_  = min(line13_y5, line24_y5)
        high_ = max(line13_y5, line24_y5)
        result["sweet_zone"] = (low_, high_)
        if not (low_ <= y5 <= high_):
            result["msgs"].append("W5 not in sweet zone (between (1->3) & (2->4)).")

    # Time symmetry => bar fark
    if check_time_symmetry:
        bars_23= x3- x2
        bars_34= x4- x3
        bars_45= x5- x4
        def ratio(a,b):
            return abs(a-b)/(abs(b)+1e-9)
        r1= ratio(bars_23, bars_34)
        r2= ratio(bars_34, bars_45)
        # tolerance => 0.3 => %30
        if (r1> 0.3) or (r2>0.3):
            result["time_symmetry_ok"] = False
            result["msgs"].append(f"Time symmetry fail => bars(2->3,3->4,4->5) differ ratio>0.3 => r1={r1:.2f}, r2={r2:.2f}")

    # line_projection => intersection(1->4)&(2->3) => epa
    def line_intersect(m1,b1, m2,b2):
        if abs(m1- m2)< 1e-9:
            return None,None
        ix= (b2- b1)/(m1- m2)
        iy= m1*ix + b1
        return ix,iy

    if line_projection_check:
        m14,b14 = line_eq(x1,y1, x4,y4)
        m23,b23 = line_eq(x2,y2, x3,y3)
        if (m14 is not None) and (m23 is not None):
            ix, iy = line_intersect(m14,b14, m23,b23)
            if (ix is not None) and (iy is not None):
                result["intersection"] = (ix, iy)
                # check_1_4_intersection_time => ix> x5 ?
                if check_1_4_intersection_time:
                    if ix< x5:
                        result["msgs"].append("Line(1->4)&(2->3) intersection < w5 => might degrade.")
                # epa_line
                result["epa_line"] = {
                    "m": m14,
                    "b": b14,
                    "ix": ix,
                    "iy": iy
                }

    # breakout confirm => w5, line(1->4) => df last close
    # ama "degrade" => eger fail => msgs'e ekle, found i kesme
    if breakout_confirm and df is not None:
        close_col = f"Close_{time_frame}"
        if close_col not in df.columns:
            result["msgs"].append(f"Missing col {close_col} in df => can't confirm breakout.")
        else:
            last_close= df[close_col].iloc[-1]
            m14,b14= line_eq(x1,y1, x4,y4)
            if m14 is None:
                result["msgs"].append("Line(1->4) slope vertical => can't confirm breakout.")
            else:
                last_i= len(df)-1
                line_y= m14* last_i + b14
                if last_close> line_y:
                    result["breakout"]= True
                else:
                    result["msgs"].append("No breakout => last_close below line(1->4).")

    # => "found" => True. Eger tumu fail olsa bile degrade cıkabilir.
    # Yine de "true" derseniz "tam wolfe" gibi. 
    # Isterseniz "found" = (len(result["msgs"])< X) 
    # vs. parametre ile karar verebilirsiniz.
    result["found"] = True

    # Or "found"u su sekilde degrade edebilirsiniz:
    # if len(result["msgs"])> 3:
    #    result["found"]= False
    #    result["msgs"].append("Too many fails => not wolfe wave.")
    # 
    # su an icin "her ne olursa" found=True => degrade form.

    return result
