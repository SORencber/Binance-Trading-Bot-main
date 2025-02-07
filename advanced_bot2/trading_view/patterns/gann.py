
############################
# GANN ULTRA FINAL
############################
import pandas as pd 
import swisseph as swe
from datetime import date, datetime
import math


def get_planet_angle(dt, planet_name="SUN"):
    if swe is None:
        return 0
    if not isinstance(dt, (datetime, date)):
        dt= pd.to_datetime(dt)
    jd= swe.julday(dt.year, dt.month, dt.day)
    planet_codes= {
        "SUN": swe.SUN, "MOON": swe.MOON, "MERCURY": swe.MERCURY, "VENUS": swe.VENUS,
        "MARS": swe.MARS, "JUPITER": swe.JUPITER, "SATURN": swe.SATURN,
        "URANUS": swe.URANUS, "NEPTUNE": swe.NEPTUNE, "PLUTO": swe.PLUTO
    }
    pcode= planet_codes.get(planet_name.upper(), swe.SUN)
    flag= swe.FLG_SWIEPH | swe.FLG_SPEED
    pos, ret= swe.calc(jd, pcode, flag)
    return pos[0]  # 0-360

def advanced_wheel_of_24_variants(anchor_price: float, variant: str = "typeA", steps: int = 5):
    """
    Basit Wheel-of-24 türetmesi. 
    """
    levels = []
    if anchor_price <= 0:
        return levels
    if variant == "typeA":
        for n in range(1, steps+1):
            uv = anchor_price*(1+ n*(24/100))
            dv = anchor_price*(1- n*(24/100))
            if uv>0: levels.append(uv)
            if dv>0: levels.append(dv)
    elif variant == "typeB":
        base= math.sqrt(24)
        anc_sqrt= math.sqrt(anchor_price)
        for n in range(1, steps+1):
            upv= (anc_sqrt+ n*base)**2
            dnv= None
            if anc_sqrt> n*base:
                dnv= (anc_sqrt- n*base)**2
            if upv>0: levels.append(upv)
            if dnv and dnv>0: levels.append(dnv)
    else:
        for n in range(1, steps+1):
            uv= anchor_price+ n*15
            dv= anchor_price- n*15
            if uv>0: levels.append(uv)
            if dv>0: levels.append(dv)
    return sorted(list(set(levels)))

def detect_gann_pattern_ultra_final(
    df: pd.DataFrame,
    pivots,
    use_ultra: bool=True,
    time_frame: str = "1m",
    # Gann Ratios veya Angle
    use_gann_ratios: bool = True,
    gann_ratios = [1.0, 2.0, 0.5],
    angles = [45.0, 22.5, 67.5, 90.0, 135.0, 180.0],
    additional_angle_shift: float = 180.0,

    # Anchor / Pivot param
    pivot_window: int = 200,
    anchor_count: int = 3,
    pivot_select_mode: str = "extremes_vol",

    # Fan param
    line_tolerance: float = 0.005,
    min_line_respects: int = 3,

    # ATR/Volume filtreleri
    atr_filter: bool = True,
    atr_period: int = 14,
    atr_factor: float = 0.5,
    volume_filter: bool = False,
    volume_ratio: float = 1.3,

    # Square of 9 & Wheel of 24
    sq9_variant: str = "typeA",
    sq9_steps: int = 5,
    sq9_tolerance: float = 0.01,
    w24_variant: str = "typeB",
    w24_steps: int = 5,
    w24_tolerance: float = 0.01,

    # Time / Astro cycles
    cycles = None,
    astro_cycles = None,
    cycle_pivot_tolerance: int = 2,
    pivot_left_bars: int = 3,
    pivot_right_bars: int = 3,

    debug: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
)-> dict:
    """
    Gann Patterns hepsi bir arada. 
    """
    import numpy as np

    result={
        "pattern": "gann",
        "found": False,
        "best_anchor": None,
        "anchors": [],
        "gann_line": None,
        "retest_info": None,
        "msgs": []
    }

    close_col= f"Close_{time_frame}"
    if cycles is None:
        cycles= [30,90,180]
    if astro_cycles is None:
        astro_cycles= [90,180,360]

    if close_col not in df.columns or len(df)< pivot_window:
        result["msgs"].append("Not enough data or missing close_col.")
        return result

    if atr_filter:
        prepare_atr(df, time_frame, period=atr_period)

    # 1) Anchor pivot seçimi
    anchor_pivots= []
    seg= df[close_col].iloc[-pivot_window:]
    smin= seg.min()
    smax= seg.max()
    i_min= seg.idxmin()
    i_max= seg.idxmax()
    anchor_pivots.append((i_min, smin))
    anchor_pivots.append((i_max, smax))

    if pivot_select_mode=="extremes_vol":
        vol_col= get_col_name("Volume", time_frame)
        if vol_col in df.columns:
            vseg= df[vol_col].iloc[-pivot_window:]
            iv= vseg.idxmax()
            if iv not in [i_min, i_max]:
                pv= df[close_col].loc[iv]
                anchor_pivots.append((iv, pv))

    anchor_pivots= list(dict.fromkeys(anchor_pivots))
    if len(anchor_pivots)> anchor_count:
        anchor_pivots= anchor_pivots[:anchor_count]

    def slope_from_gann_ratio(ratio: float)-> float:
        return ratio

    def slope_from_angle(deg: float)-> float:
        return math.tan(math.radians(deg))

    def build_fan_lines(anc_idx, anc_val):
        fan=[]
        if use_gann_ratios:
            for r in gann_ratios:
                m_pos= slope_from_gann_ratio(r)
                m_neg= -m_pos
                fan.append({
                    "label": f"{r}x1(+)",
                    "ratio": r,
                    "angle_deg": None,
                    "slope": m_pos,
                    "respects": 0,
                    "confidence": 0.0,
                    "points":[]
                })
                fan.append({
                    "label": f"{r}x1(-)",
                    "ratio": -r,
                    "angle_deg": None,
                    "slope": m_neg,
                    "respects": 0,
                    "confidence": 0.0,
                    "points":[]
                })
        else:
            expanded_angles= angles[:]
            if additional_angle_shift>0:
                for ag in angles:
                    shifted= ag+ additional_angle_shift
                    if shifted not in expanded_angles:
                        expanded_angles.append(shifted)
            expanded_angles= sorted(list(set(expanded_angles)))
            for ag in expanded_angles:
                m= slope_from_angle(ag)
                fan.append({
                    "label": f"{ag}°",
                    "ratio": None,
                    "angle_deg": ag,
                    "slope": m,
                    "respects": 0,
                    "confidence":0.0,
                    "points":[]
                })
        return fan

    def check_fan_respects(fan_lines, anc_idx, anc_val):
        for b_i in range(len(df)):
            px= df[close_col].iloc[b_i]
            # ATR Filter
            if atr_filter:
                atr_col= get_col_name("ATR", time_frame)
                if atr_col in df.columns:
                    av= df[atr_col].iloc[b_i]
                    if not math.isnan(av):
                        rng= df[get_col_name("High", time_frame)].iloc[b_i]- df[get_col_name("Low", time_frame)].iloc[b_i]
                        if rng< (av* atr_factor):
                            continue
            xdiff= b_i- anc_idx
            for fl in fan_lines:
                line_y= fl["slope"]* xdiff+ anc_val
                dist_rel= abs(px- line_y)/(abs(line_y)+1e-9)
                if dist_rel< line_tolerance:
                    # pivot?
                    ptype= None
                    # local min check
                    # buraya pivot_left_bars vs. entegre edebilirsiniz
                    fl["respects"]+=1
                    fl["points"].append((b_i, line_y, px, ptype))
        for fl in fan_lines:
            c=0.0
            pivot_count= sum(1 for pt in fl["points"] if pt[3] is not None)
            if fl["respects"]>= min_line_respects:
                c= 0.5+ min(0.5, 0.05* (fl["respects"]- min_line_respects))
            c+= pivot_count*0.1
            if c>1.0: c=1.0
            fl["confidence"]= round(c,2)

    def compute_sq9_levels(anchor_price: float):
        return advanced_wheel_of_24_variants(anchor_price, variant=sq9_variant, steps=sq9_steps)

    def check_levels_respects(level_list, tolerance):
        out=[]
        for lv in level_list:
            res_count=0
            plist=[]
            for b_i in range(len(df)):
                px_b= df[close_col].iloc[b_i]
                dist= abs(px_b- lv)/(abs(lv)+1e-9)
                if dist< tolerance:
                    res_count+=1
                    plist.append((b_i, px_b))
            conf_= min(1.0, res_count/10)
            out.append((lv, res_count, conf_, plist))
        return out

    def compute_wheel24(anchor_price: float):
        return advanced_wheel_of_24_variants(anchor_price, variant=w24_variant, steps=w24_steps)

    def build_time_cycles(anchor_idx):
        cyc_data=[]
        # sabit bar cycles
        for cyc in cycles:
            tbar= anchor_idx+ cyc
            cyc_date= df.index[tbar] if (0<= tbar< len(df)) else None
            cyc_data.append({
                "bars": cyc,
                "astro": None,
                "target_bar": tbar,
                "approx_date": str(cyc_date) if cyc_date else None,
                "cycle_confidence": 0.0,
                "pivot_detected": None
            })
        # astro
        anchor_date= df.index[anchor_idx] if (0<= anchor_idx< len(df)) else None
        if anchor_date:
            anchor_astro_angle= get_planet_angle(anchor_date, "SUN")
        else:
            anchor_astro_angle= 0
        for deg in astro_cycles:
            target_bar= anchor_idx+ deg
            cyc_date= df.index[target_bar] if (0<= target_bar< len(df)) else None
            cyc_data.append({
                "bars": None,
                "astro": (anchor_astro_angle+ deg)%360,
                "target_bar": target_bar,
                "approx_date": str(cyc_date) if cyc_date else None,
                "cycle_confidence":0.0,
                "pivot_detected": None
            })
        return cyc_data

    def is_local_min(df, bar_i: int, close_col: str, left_bars: int, right_bars: int)-> bool:
        if bar_i< left_bars or bar_i> (len(df)- right_bars-1):
            return False
        val= df[close_col].iloc[bar_i]
        left_slice= df[close_col].iloc[bar_i- left_bars: bar_i]
        right_slice= df[close_col].iloc[bar_i+1: bar_i+1+ right_bars]
        return (all(val< x for x in left_slice) and all(val<= x for x in right_slice))

    def is_local_max(df, bar_i: int, close_col: str, left_bars: int, right_bars: int)-> bool:
        if bar_i< left_bars or bar_i> (len(df)- right_bars-1):
            return False
        val= df[close_col].iloc[bar_i]
        left_slice= df[close_col].iloc[bar_i- left_bars: bar_i]
        right_slice= df[close_col].iloc[bar_i+1: bar_i+1+ right_bars]
        return (all(val> x for x in left_slice) and all(val>= x for x in right_slice))

    def check_cycle_pivots(cyc_data):
        for ci in cyc_data:
            tb= ci["target_bar"]
            if (tb is not None) and (0<= tb< len(df)):
                lb= max(0, tb- cycle_pivot_tolerance)
                rb= min(len(df)-1, tb+ cycle_pivot_tolerance)
                found_piv= None
                for b_ in range(lb, rb+1):
                    if is_local_min(df, b_, close_col, pivot_left_bars, pivot_right_bars):
                        found_piv= (b_, df[close_col].iloc[b_], "min")
                        break
                    elif is_local_max(df, b_, close_col, pivot_left_bars, pivot_right_bars):
                        found_piv= (b_, df[close_col].iloc[b_], "max")
                        break
                if found_piv:
                    ci["pivot_detected"]= found_piv
                    ci["cycle_confidence"]= 1.0
                else:
                    ci["cycle_confidence"]= 0.0

    def build_confluence_points(fan_lines, sq9_data, w24_data, cyc_data, anc_idx, anc_val):
        conf=[]
        for fl in fan_lines:
            if fl["confidence"]<=0:
                continue
            for (b_i, line_y, px, ptype) in fl["points"]:
                sq9_match= None
                for (lvl, rescount, conf_, plist) in sq9_data:
                    dist= abs(px- lvl)/(abs(lvl)+1e-9)
                    if dist< (sq9_tolerance*2):
                        sq9_match= lvl
                        break
                w24_match= None
                for (wl, wres, wconf, wpl) in w24_data:
                    dist= abs(px- wl)/(abs(wl)+1e-9)
                    if dist< (w24_tolerance*2):
                        w24_match= wl
                        break
                cyc_found= None
                for ci in cyc_data:
                    if ci["cycle_confidence"]>0 and ci["pivot_detected"]:
                        if abs(ci["pivot_detected"][0]- b_i)<= cycle_pivot_tolerance:
                            cyc_found= ci
                            break
                if (sq9_match or w24_match or cyc_found):
                    cboost= fl["confidence"]
                    if sq9_match: cboost+= 0.3
                    if w24_match: cboost+= 0.2
                    if cyc_found: cboost+= 0.4
                    if cboost> 2.0: cboost= 2.0
                    conf.append({
                        "bar_index": b_i,
                        "price": px,
                        "fan_line_label": fl["label"],
                        "ptype": ptype,
                        "sq9_level": sq9_match,
                        "w24_level": w24_match,
                        "cycle_bar": cyc_found["target_bar"] if cyc_found else None,
                        "confidence_boost": round(cboost,2)
                    })
        return conf

    anchor_list=[]
    for (anc_idx, anc_val) in anchor_pivots:
        item={
            "anchor_idx": anc_idx,
            "anchor_price": anc_val,
            "fan_lines":[],
            "sq9_levels":[],
            "wheel24_levels":[],
            "time_cycles":[],
            "confluence_points":[],
            "score":0.0
        }
        fl= build_fan_lines(anc_idx, anc_val)
        check_fan_respects(fl, anc_idx, anc_val)
        item["fan_lines"]= fl

        sq9_lvls= compute_sq9_levels(anc_val)
        sq9_data= check_levels_respects(sq9_lvls, sq9_tolerance)
        item["sq9_levels"]= sq9_data

        w24_lvls= compute_wheel24(anc_val)
        w24_data= check_levels_respects(w24_lvls, w24_tolerance)
        item["wheel24_levels"]= w24_data

        cyc_data= build_time_cycles(anc_idx)
        check_cycle_pivots(cyc_data)
        item["time_cycles"]= cyc_data

        conf_pts= build_confluence_points(fl, sq9_data, w24_data, cyc_data, anc_idx, anc_val)
        item["confluence_points"]= conf_pts

        best_fan_conf= max([f["confidence"] for f in fl]) if fl else 0
        ccount= len(conf_pts)
        item["score"]= round(best_fan_conf+ ccount*0.2,2)
        anchor_list.append(item)

    if not anchor_list:
        return result
    best_anchor= max(anchor_list, key=lambda x: x["score"])
    result["best_anchor"]= best_anchor
    result["anchors"]= anchor_list
    if best_anchor["score"]<= 0:
        return result

    result["found"]= True
    best_fan_line= None
    best_conf= -999
    for fl in best_anchor["fan_lines"]:
        if fl["confidence"]> best_conf:
            best_conf= fl["confidence"]
            best_fan_line= fl
    if best_fan_line and best_fan_line["respects"]>=3:
        anc_idx= best_anchor["anchor_idx"]
        anc_val= best_anchor["anchor_price"]
        pivot_points= [pt for pt in best_fan_line["points"] if pt[3] is not None]
        if len(pivot_points)>=2:
            x_vals= [pp[0] for pp in pivot_points]
            y_vals= [pp[2] for pp in pivot_points]
            m_ref,b_ref= np.polyfit(x_vals, y_vals,1)
            x2= anc_idx+ 100
            y2= m_ref*x2+ b_ref
            result["gann_line"]= ((anc_idx, anc_val),(x2,y2))
        else:
            x2= anc_idx+ 100
            y2= anc_val+ best_fan_line["slope"]*100
            result["gann_line"]= ((anc_idx,anc_val),(x2,y2))

    if check_retest and result["gann_line"]:
        (ixA, pxA),(ixB, pxB)= result["gann_line"]
        m_line, b_line= line_equation(ixA, pxA, ixB, pxB)
        if m_line is not None:
            retest_done=False
            retest_bar=None
            start_bar= int(max(ixA, ixB))
            for i in range(start_bar, len(df)):
                c= df[close_col].iloc[i]
                line_y= m_line* i+ b_line
                dist_perc= abs(c- line_y)/(abs(line_y)+1e-9)
                if dist_perc<= retest_tolerance:
                    retest_done= True
                    retest_bar= i
                    break
            result["retest_info"]= {
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
def prepare_atr(df: pd.DataFrame, time_frame: str = "1m", period: int = 14):
    """
    ATR hesaplama ve df'ye ekleme
    """
    high_col  = get_col_name("High",  time_frame)
    low_col   = get_col_name("Low",   time_frame)
    close_col = get_col_name("Close", time_frame)
    atr_col   = get_col_name("ATR",   time_frame)

    if atr_col in df.columns:
        return
    df[f"H-L_{time_frame}"] = df[high_col] - df[low_col]
    df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
    df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()

    df[f"TR_{time_frame}"] = df[[f"H-L_{time_frame}",
                                 f"H-PC_{time_frame}",
                                 f"L-PC_{time_frame}"]].max(axis=1)
    df[atr_col] = df[f"TR_{time_frame}"].rolling(period).mean()
