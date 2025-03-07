import pandas as pd
import numpy as np
from math import isnan
import talib

##################################################################
# 1) Pattern Skor Fonksiyonu (final_score)
##################################################################
# def final_score(df, patterns, time_frame, check_rsi_macd,
#                 v_spike, rsi_macd_signal, b_up, b_down,
#                 ml_label, max_bars_ago, wave, require_confirmed, trade_levels):
#     """
#     Verilen pattern dictionary'si üzerinden bir skor ve
#     sinyal (BUY/SELL/HOLD) oluşturur.
#     trade_levels: pattern bazında hesaplanmış entry/stop/tp gibi seviye bilgileri
#     """
#     pattern_score = 0
#     reasons = []

#     def log(msg, level="info"):
#         print(f"[{level.upper()}] {msg}")
     
#     def filter_patterns(pat_list):
#         #print(pat_list)
#         if isinstance(pat_list, dict):
#             pat_list = [pat_list]
#         elif not isinstance(pat_list, list):
#             log(f"Pattern type mismatch => {pat_list}", "error")
#             return []
        
#         filtered = []
#         cutoff = len(df) - max_bars_ago
#         for p in pat_list:
#             end_bar = p.get("end_bar", None)
#             cbar = p.get("confirmed_bar", None)
#             if end_bar is None:
#                 end_bar = cbar if cbar else 0
#             if end_bar >= cutoff:
#                 if require_confirmed:
#                     if p.get("confirmed", False):
#                         filtered.append(p)
#                 else:
#                     filtered.append(p)
#         #print("FILITRE------------------",filtered)
#         return filtered

#     # ----------------------------------------------------------------
#     # HEAD & SHOULDERS
#     # ----------------------------------------------------------------
#     hs_list = filter_patterns(patterns.get("head_and_shoulders", []))
#     for hs in hs_list:
#         val = -4
#         if hs.get("confirmed") and hs.get("volume_check", True):
#             val = -5
#         pattern_score += val
#         reasons.append(f"headshoulders({val})")

#     inv_hs_list = filter_patterns(patterns.get("inverse_head_and_shoulders", []))
#     for inv in inv_hs_list:
#         val = +3
#         if inv.get("confirmed") and inv.get("volume_check", True):
#             val = +4
#         pattern_score += val
#         reasons.append(f"inverseHS({val})")

#     # ----------------------------------------------------------------
#     # DOUBLE / TRIPLE TOP-BOTTOM (Klasik)
#     # ----------------------------------------------------------------
#     dtops = filter_patterns(patterns.get("double_top", []))
#     for dt in dtops:
#         val = -2
#         if dt.get("confirmed"):
#             val -= 3  # toplam -5
#         pattern_score += val
#         reasons.append(f"double_top({val})")

#     dbots = filter_patterns(patterns.get("double_bottom", []))
#     for db in dbots:
#         val = +2
#         if db.get("confirmed"):
#             val += 1  # toplam +3
#         pattern_score += val
#         reasons.append(f"double_bottom({val})")

#     # ----------------------------------------------------------------
#     # TRIPLE TOP / BOTTOM ADVANCED
#     # ----------------------------------------------------------------
#     triple_tops = filter_patterns(patterns.get("triple_top_advanced", []))
#     for ttop in triple_tops:
#         val = -3
#         if ttop.get("confirmed"):
#             val -= 2  # -5
#         pattern_score += val
#         reasons.append(f"triple_top_adv({val})")

#     triple_bots = filter_patterns(patterns.get("triple_bottom_advanced", []))
#     for tbot in triple_bots:
#         val = +3
#         if tbot.get("confirmed"):
#             val += 2  # +5
#         pattern_score += val
#         reasons.append(f"triple_bottom_adv({val})")

#     # ----------------------------------------------------------------
#     # ELLIOTT
#     # ----------------------------------------------------------------
#     ell = patterns.get("elliott", {})
#     if ell.get("found") and wave:
#         if wave[-1][0] >= (len(df) - max_bars_ago):
#             if ell.get("trend")=="UP":
#                 pattern_score += 3
#                 reasons.append("elliott_up")
#             else:
#                 pattern_score -= 3
#                 reasons.append("elliott_down")

#     # ----------------------------------------------------------------
#     # WOLFE
#     # ----------------------------------------------------------------
#     wol = patterns.get("wolfe", {})
#     if wol.get("found") and wave:
#         if wave[-1][0] >= (len(df) - max_bars_ago):
#             wol_val = +2
#             if wol.get("breakout"):
#                 wol_val += 1
#             pattern_score += wol_val
#             reasons.append(f"wolfe({wol_val})")

#     # ----------------------------------------------------------------
#     # HARMONIC
#     # ----------------------------------------------------------------
#     harm = patterns.get("harmonic", {})
#     if harm.get("found") and wave:
#         if wave[-1][0] >= (len(df)-max_bars_ago):
#             pattern_score -= 1
#             reasons.append("harmonic(-1)")

#     # ----------------------------------------------------------------
#     # TRIANGLE
#     # ----------------------------------------------------------------
#     tri = patterns.get("triangle", {})
#     if tri.get("found") and tri.get("breakout", False) and wave:
#         if wave[-1][0] >= (len(df)-max_bars_ago):
#             if tri.get("triangle_type")=="ascending":
#                 pattern_score += 1
#                 reasons.append("triangle_asc(+1)")
#             elif tri.get("triangle_type")=="descending":
#                 pattern_score -= 1
#                 reasons.append("triangle_desc(-1)")
#             else:
#                 pattern_score += 1
#                 reasons.append("triangle_sym(+1)")

#     # ----------------------------------------------------------------
#     # WEDGE
#     # ----------------------------------------------------------------
#     wd = patterns.get("wedge", {})
#     if wd.get("found") and wd.get("breakout", False) and wave:
#         if wave[-1][0] >= (len(df)-max_bars_ago):
#             if wd.get("wedge_type")=="rising":
#                 pattern_score -= 1
#                 reasons.append("wedge_rising(-1)")
#             else:
#                 pattern_score += 1
#                 reasons.append("wedge_falling(+1)")

#     # ----------------------------------------------------------------
#     # CUP & HANDLE
#     # ----------------------------------------------------------------
#     cup_list = filter_patterns(patterns.get("cup_handle", []))
#     for ch in cup_list:
#         val = +2
#         if ch.get("confirmed"):
#             val += 1
#         pattern_score += val
#         reasons.append(f"cup_handle({val})")

#     # ----------------------------------------------------------------
#     # FLAG / PENNANT
#     # ----------------------------------------------------------------
#     flag_list = filter_patterns(patterns.get("flag_pennant", []))
#     for fp in flag_list:
#         val = +2
#         if fp.get("confirmed"):
#             val += 1
#         pattern_score += val
#         reasons.append(f"flag_pennant({val})")

#     # ----------------------------------------------------------------
#     # CHANNEL
#     # ----------------------------------------------------------------
#     chn_list = filter_patterns(patterns.get("channel", []))
#     for cn in chn_list:
#         val = +1
#         ctype = cn.get("channel_type", "")
#         if ctype == "descending":
#             val = -1
#         elif ctype == "horizontal":
#             val = 0
#         pattern_score += val
#         reasons.append(f"channel({val})")

#     # ----------------------------------------------------------------
#     # GANN
#     # ----------------------------------------------------------------
#     gann_info = filter_patterns(patterns.get("gann", {}))
#     #print(gann_info)
#     if isinstance(gann_info, dict):
#         if gann_info.get("found", False):
#             pattern_score += 1
#             reasons.append("gann(+1)")

#     # ----------------------------------------------------------------
#     # ML LABEL => 0=HOLD, 1=BUY, 2=SELL
#     # ----------------------------------------------------------------
#     if ml_label == 1:
#         pattern_score += 3
#         reasons.append("ml_buy")
#     elif ml_label == 2:
#         pattern_score -= 3
#         reasons.append("ml_sell")

#     # ----------------------------------------------------------------
#     # Breakout / Hacim => final_score >0 => potansiyel long => breakout_up => +1
#     # ----------------------------------------------------------------
#     final_score_ = pattern_score
#     if final_score_ > 0:
#         if b_up:
#             final_score_ += 1
#             reasons.append("breakout_up")
#             if v_spike:
#                 final_score_ += 1
#                 reasons.append("vol_spike_up")
#     elif final_score_ < 0:
#         if b_down:
#             final_score_ -= 1
#             reasons.append("breakout_down")
#             if v_spike:
#                 final_score_ -= 1
#                 reasons.append("vol_spike_down")

#     # ----------------------------------------------------------------
#     # RSI/MACD => eğer false ise skoru 1 puan düşür
#     # (Bu basit bir skor düşürme mekanizması; 
#     #  asıl filtreleme "filter_trades_with_indicators" ile daha detaylı yapılabilir)
#     # ----------------------------------------------------------------
#     if check_rsi_macd:
#         if not rsi_macd_signal:
#             final_score_ -= 1
#             reasons.append("rsi_macd_fail")

#     # Son Karar
#     final_signal = "HOLD"
#     if final_score_ >= 2:
#         final_signal = "BUY"
#     elif final_score_ <= -2:
#         final_signal = "SELL"

#     reason_str = ",".join(reasons) if reasons else "NONE"

#     return {
#         "signal": final_signal,
#         "score": final_score_,
#         "reason": reason_str,
#         "patterns": patterns,
#         "ml_label": ml_label,
#         "breakout_up": b_up,
#         "breakout_down": b_down,
#         "volume_spike": v_spike,
#         "time_frame": time_frame,
#         "pattern_trade_levels": trade_levels
#     }


##################################################################
# 2) Pattern kritikleri için parse_line_or_point
##################################################################
def parse_line_or_point(val):
    """
    Pattern sonuçlarında kritik seviye (ör. neckline, rim_line)
    şu formatlardan biri olabilir:
      - float/int => direkt seviye (örnek: 5.23)
      - (idx, price) => price döndürür
      - ((ix1, px1),(ix2, px2)) => iki pivot noktası => ortalama fiyat
    """
    if val is None:
        return None

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, tuple):
        # Tek pivot: (barIndex, price)
        if len(val) == 2 and all(isinstance(x, (int,float)) for x in val):
            return float(val[1])

        # İki pivot: ((ix1, px1),(ix2, px2))
        if (len(val) == 2
            and isinstance(val[0], tuple)
            and isinstance(val[1], tuple)):
            (i1, p1), (i2, p2) = val
            return (p1 + p2) / 2.0

    return None


##################################################################
# 3) Pattern Mesafe Ölçümü (measure_pattern_distances)
##################################################################
def measure_pattern_distances(
    patterns_dict: dict,
    current_price: float,
    tolerance: float = 0.01
) -> list:
    """
    'patterns_dict': detect_all_patterns_v2(...) çıktısı
    'current_price': anlık fiyat
    'tolerance':     oransal yakınlık eşiği (örn. %1)

    Pattern’deki kritik çizgiler (ör. neckline, breakout_line, rim_line vb.)
    float değere dönüştürülür. Sonra (current_price - line_value)/line_value
    oranını bulur. 'within_tolerance' => True/False.
    """
    results = []

    def _add_result(pname, sub_idx, line_label, line_val, confirmed, raw_pattern):
        if line_val is None:
            return
        dist_ratio = abs(current_price - line_val) / (abs(line_val)+1e-9)
        info = {
            "pattern"         : pname,
            "sub_index"       : sub_idx,
            "line_label"      : line_label,
            "line_value"      : line_val,
            "confirmed"       : confirmed,
            "distance_ratio"  : dist_ratio,
            "within_tolerance": (dist_ratio <= tolerance),
            "pattern_raw"     : raw_pattern
        }
        #print(pname,confirmed, dist_ratio,tolerance)
        results.append(info)

    def parse_line_or_point_local(val):
        """
        Lokal parse, yukarıdaki parse_line_or_point fonksiyonu ile aynı mantık.
        """
        if val is None:
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, tuple):
            if len(val) == 2 and all(isinstance(x, (int,float)) for x in val):
                return float(val[1])
            if len(val) == 2 and isinstance(val[0], tuple) and isinstance(val[1], tuple):
                (i1, p1), (i2, p2) = val
                return (p1 + p2) / 2.0
        return None

    # ----------------- HEAD & SHOULDERS -----------------
    hs_list = patterns_dict.get("head_and_shoulders", [])
    #print(hs_list)

    if isinstance(hs_list, dict):
        hs_list = [hs_list]
    for i, hs in enumerate(hs_list):
        lvl = parse_line_or_point_local(hs.get("neckline"))
        conf = hs.get("confirmed", False)
        _add_result("head_and_shoulders", i, "neckline", lvl, conf, hs)

    # ----------------- INVERSE HS -----------------
    inv_hs_list = patterns_dict.get("inverse_head_and_shoulders", [])
    #print(inv_hs_list)

    if isinstance(inv_hs_list, dict):
        inv_hs_list = [inv_hs_list]
    for i, invh in enumerate(inv_hs_list):
        lvl = parse_line_or_point_local(invh.get("neckline"))
        conf = invh.get("confirmed", False)
        _add_result("inverse_head_and_shoulders", i, "neckline", lvl, conf, invh)

    # ----------------- DOUBLE TOP -----------------
    dt_list = patterns_dict.get("double_top", [])
    if isinstance(dt_list, dict):
        dt_list = [dt_list]
    for i, dt in enumerate(dt_list):
        lvl = parse_line_or_point_local(dt.get("neckline"))
        conf = dt.get("confirmed", False)
        _add_result("double_top", i, "neckline", lvl, conf, dt)

    # ----------------- DOUBLE BOTTOM -----------------
    db_list = patterns_dict.get("double_bottom", [])
    if isinstance(db_list, dict):
        db_list = [db_list]
    for i, db in enumerate(db_list):
        lvl = parse_line_or_point_local(db.get("neckline"))
        conf = db.get("confirmed", False)
        _add_result("double_bottom", i, "neckline", lvl, conf, db)

    # ----------------- TRIPLE TOP ADVANCED -----------------
    tta_list = patterns_dict.get("triple_top_advanced", [])
    if isinstance(tta_list, dict):
        tta_list = [tta_list]
    for i, tta in enumerate(tta_list):
        lvl = parse_line_or_point_local(tta.get("neckline"))
        conf = tta.get("confirmed", False)
        _add_result("triple_top_advanced", i, "neckline", lvl, conf, tta)

    # ----------------- TRIPLE BOTTOM ADVANCED -----------------
    tba_list = patterns_dict.get("triple_bottom_advanced", [])
    if isinstance(tba_list, dict):
        tba_list = [tba_list]
    for i, tba in enumerate(tba_list):
        lvl = parse_line_or_point_local(tba.get("neckline"))
        conf = tba.get("confirmed", False)
        _add_result("triple_bottom_advanced", i, "neckline", lvl, conf, tba)

    # ----------------- WEDGE -----------------
    wedge_data = patterns_dict.get("wedge", {})
    #print(wedge_data)

    if isinstance(wedge_data, dict):
        if wedge_data.get("found", False):
            wline = wedge_data.get("breakout_line")
            lvl = parse_line_or_point_local(wline)
            confirmed = wedge_data.get("breakout", False)
            _add_result("wedge", 0, "breakout_line", lvl, confirmed, wedge_data)

    # ----------------- TRIANGLE -----------------
    tri_data = patterns_dict.get("triangle", {})
    #print(tri_data)
    if isinstance(tri_data, dict):
        if tri_data.get("found", False):
            tline = tri_data.get("breakout_line")
            lvl = parse_line_or_point_local(tline)
            confirmed = tri_data.get("breakout", False)
            _add_result("triangle", 0, "breakout_line", lvl, confirmed, tri_data)

    # ----------------- CUP & HANDLE -----------------
    ch_data = patterns_dict.get("cup_handle", {})
    #print(ch_data)

    if isinstance(ch_data, dict):
        if ch_data.get("found", False):
            rim_line = ch_data.get("rim_line")
            lvl = parse_line_or_point_local(rim_line)
            confirmed = ch_data.get("confirmed", False)
            _add_result("cup_handle", 0, "rim_line", lvl, confirmed, ch_data)

    # ----------------- FLAG/PENNANT -----------------
    fp_data = patterns_dict.get("flag_pennant", {})
   # print(fp_data)

    if isinstance(fp_data, dict):
        if fp_data.get("found", False):
            br_line = fp_data.get("breakout_line")
            lvl = parse_line_or_point_local(br_line)
            confirmed = fp_data.get("confirmed", False)
            _add_result("flag_pennant", 0, "breakout_line", lvl, confirmed, fp_data)

    # ----------------- CHANNEL -----------------
    chan_data = patterns_dict.get("channel", {})
    #print(chan_data)
    if isinstance(chan_data, dict):
        if chan_data.get("found", False):
            br_line = chan_data.get("breakout_line")
            lvl = parse_line_or_point_local(br_line)
            confirmed = chan_data.get("breakout", False)
            _add_result("channel", 0, "breakout_line", lvl, confirmed, chan_data)

    # ----------------- ELLIOTT -----------------
    ell_data = patterns_dict.get("elliott", {})
    #print(ell_data)
    if isinstance(ell_data, dict):
        if ell_data.get("found", False):
            w4_val = ell_data.get("wave4_level")
            lvl = parse_line_or_point_local(w4_val)
            confirmed = True
            _add_result("elliott", 0, "wave4_level", lvl, confirmed, ell_data)

    # ----------------- WOLFE -----------------
    wolfe_data = patterns_dict.get("wolfe", {})
    if isinstance(wolfe_data, dict):
        if wolfe_data.get("found", False):
            wline = wolfe_data.get("wolfe_line")
            lvl = parse_line_or_point_local(wline)
            confirmed = wolfe_data.get("breakout", False)
            _add_result("wolfe", 0, "wolfe_line", lvl, confirmed, wolfe_data)

    # ----------------- HARMONIC -----------------
    har_data = patterns_dict.get("harmonic", {})
    if isinstance(har_data, dict):
        if har_data.get("found", False):
            xabc = har_data.get("xabc", [])
            lvl = None
            confirmed = True
            if xabc and len(xabc)>=5:
                d_pivot = xabc[-1]
                if isinstance(d_pivot, (list, tuple)) and len(d_pivot)==3:
                    lvl = parse_line_or_point_local((d_pivot[0], d_pivot[1]))
                elif isinstance(d_pivot, (list, tuple)) and len(d_pivot)==2:
                    lvl = parse_line_or_point_local(d_pivot)
            _add_result("harmonic", 0, "pivot_D", lvl, confirmed, har_data)

    # ----------------- GANN -----------------
    gann_data = patterns_dict.get("gann", {})
    #print(gann_data)
    if isinstance(gann_data, dict):
        if gann_data.get("found", False):
            gline = gann_data.get("gann_line")
            lvl = parse_line_or_point_local(gline)
            _add_result("gann", 0, "gann_line", lvl, True, gann_data)
        # ----------------- RECTANGLE (Range) -----------------
        rect_data = patterns_dict.get("rectangle", {})
        if isinstance(rect_data, dict):
            if rect_data.get("found", False):
                # top_line ve bot_line değerlerini parse edip ekleyelim
                top_line_val = parse_line_or_point_local(rect_data.get("top_line"))
                bot_line_val = parse_line_or_point_local(rect_data.get("bot_line"))
                confirmed = rect_data.get("confirmed", False)

                # measure_pattern_distances fonksiyonundaki `_add_result` çağrısıyla raporlayabilirsiniz
                _add_result("rectangle", 0, "top_line", top_line_val, confirmed, rect_data)
                _add_result("rectangle", 0, "bot_line", bot_line_val, confirmed, rect_data)

    return results
 

##################################################################
# 4) filter_confirmed_within_tolerance
##################################################################
def filter_confirmed_within_tolerance(distances_list: list) -> dict:
    """
    measure_pattern_distances(...) çıktısından, confirmed=True ve
    within_tolerance=True olanları pattern bazında gruplayıp döndürür.
    
    Dönüş: {
      "double_bottom": [0, 1, ...],  # sub_index listesi
      "headshoulders": [0, ...],
      ...
    }
    """
    filtered_map = {}

    for d in distances_list:
        pname  = d["pattern"]
        sindex = d.get("sub_index", 0)
        conf   = d["confirmed"]
        within = d["within_tolerance"]
        if conf and within:
            if pname not in filtered_map:
                filtered_map[pname] = []
            filtered_map[pname].append(sindex)
    print(filtered_map)
    return filtered_map


##################################################################
# 5) Pattern bazlı trade seviyeleri çıkarma (extract_pattern_trade_levels_filtered)
##################################################################
def extract_pattern_trade_levels_filtered(
    patterns_dict: dict,
    confirmed_map: dict,
    df: pd.DataFrame,
    time_frame: str = "1m",
    atr_period: int = 14,
    atr_sl_multiplier: float = 1.0,
    atr_tp_multiplier: float = 1.0,
    default_break_offset: float = 0.001
) -> dict:
    """
    Pattern bazında entry/SL/TP hesaplayıp döndürür.
    'confirmed_map', filter_confirmed_within_tolerance(...) sonucudur.
    """
    results = {
        "head_and_shoulders": [],
        "inverse_head_and_shoulders": [],
        "double_top": [],
        "double_bottom": [],
        "triple_top_advanced": [],
        "triple_bottom_advanced": [],
        "elliott": [],
        "wolfe": [],
        "harmonic": [],
        "triangle": [],
        "wedge": [],
        "cup_handle": [],
        "flag_pennant": [],
        "channel": [],
        "gann": []
    }

    pattern_score = 0

    def get_col_name(base_col: str, tf: str) -> str:
        return f"{base_col}_{tf}"

   
    def prepare_atr(df: pd.DataFrame, tf: str = "1m", period: int = 14):
        high_col  = get_col_name("High",  tf)
        low_col   = get_col_name("Low",   tf)
        close_col = get_col_name("Close", tf)
        atr_col   = get_col_name("ATR",   tf)

        if atr_col in df.columns:
            return

        df[f"H-L_{tf}"]  = df[high_col] - df[low_col]
        df[f"H-PC_{tf}"] = (df[high_col] - df[close_col].shift(1)).abs()
        df[f"L-PC_{tf}"] = (df[low_col]  - df[close_col].shift(1)).abs()

        df[f"TR_{tf}"] = df[[f"H-L_{tf}",
                             f"H-PC_{tf}",
                             f"L-PC_{tf}"]].max(axis=1)
        df[atr_col] = df[f"TR_{tf}"].rolling(period).mean()
     # ATR hazırlığı
    atr_col = f"ATR_{time_frame}"
    if atr_col not in df.columns:
        prepare_atr(df, time_frame, period=atr_period)
    last_atr = df[atr_col].iloc[-1] if (atr_col in df.columns) else None

    # Son bar index ve fiyat
    last_i = len(df) - 1
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        # Veri yok => boş sonuç döndür
        return results

    current_price = df[close_col].iloc[-1]

    def get_close(bar_index: int):
        if bar_index < len(df):
            return df[close_col].iloc[bar_index]
        else:
            return current_price

    def add_atr_above(val: float) -> float:
        if (last_atr is not None) and (not isnan(last_atr)):
            return val + last_atr * atr_sl_multiplier
        else:
            return val * (1.0 + default_break_offset)

    def add_atr_below(val: float) -> float:
        if (last_atr is not None) and (not isnan(last_atr)):
            return val - last_atr * atr_sl_multiplier
        else:
            return val * (1.0 - default_break_offset)

    def add_atr_tp_up(val: float) -> float:
        if (last_atr is not None) and (not isnan(last_atr)):
            return val + last_atr * atr_tp_multiplier
        else:
            return val * (1.0 + 0.02)

    def add_atr_tp_down(val: float) -> float:
        if (last_atr is not None) and (not isnan(last_atr)):
            return val - last_atr * atr_tp_multiplier
        else:
            return val * (1.0 - 0.02)

    def get_entry_on_breakout_or_retest(pattern_dict, default_line_val, break_bar, side=None):
        """
        1) retest_info varsa ve retest_done => entry = retest_bar close
        2) yoksa => breakout bar close
        3) fallback => default_line_val ± offset
        4) side="long"/"short" => isterseniz ufak offset ekleyebilirsiniz
        """
        rinfo = pattern_dict.get("retest_info", {})
        if rinfo and rinfo.get("retest_done", False):
            rb = rinfo.get("retest_bar", break_bar)
            price = get_close(rb)
        else:
            if break_bar < len(df):
                price = get_close(break_bar)
            else:
                # fallback
                if side == "long":
                    price = default_line_val * (1.0 + default_break_offset)
                elif side == "short":
                    price = default_line_val * (1.0 - default_break_offset)
                else:
                    price = default_line_val

        if side == "long":
            price = price * 1.0005  # küçük bir offset örneği
        elif side == "short":
            price = price * 0.9995

        return price

    # -----------------------------------------------------
    # HEAD & SHOULDERS
    # -----------------------------------------------------
    hs_list = patterns_dict.get("head_and_shoulders",[] )
    needed_idxs = confirmed_map.get("head_and_shoulders", [])
    #print(hs_list, needed_idxs)
    if isinstance(hs_list, dict):
        hs_list = [hs_list]
    if isinstance(hs_list, list):
        for idx_, hs_data in enumerate(hs_list):
            if idx_ not in needed_idxs:
                continue
            if not hs_data.get("confirmed", False):
                continue
            H_idx, H_prc = hs_data["H"]
            (nx1, px1), (nx2, px2) = hs_data["neckline"]
            neck_avg = (px1 + px2) / 2
            cbar = hs_data.get("confirmed_bar", last_i)

            # SHORT
            entry_price = get_entry_on_breakout_or_retest(hs_data, neck_avg, cbar, side="short")
            stop_loss   = add_atr_above(H_prc)
            distance    = (H_prc - neck_avg)
            take_profit = neck_avg - distance
            pattern_score-=3
            results["head_and_shoulders"].append({
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": "SHORT",
                "pattern_raw": hs_data
            })

    # -----------------------------------------------------
    # INVERSE HEAD & SHOULDERS
    # -----------------------------------------------------
    inv_list = patterns_dict.get("inverse_head_and_shoulders", [])
    inv_need = confirmed_map.get("inverse_head_and_shoulders", [])
    if isinstance(inv_list, dict):
        inv_list = [inv_list]

    for i, inv in enumerate(inv_list):
        if i not in inv_need:
            continue
        if not inv.get("confirmed", False):
            continue
        H_idx, H_prc = inv["H"]
        (nx1, px1), (nx2, px2) = inv["neckline"]
        neck_avg = (px1 + px2)/2
        cbar = inv.get("confirmed_bar", last_i)

        # LONG
        entry_price = get_entry_on_breakout_or_retest(inv, neck_avg, cbar, side="long")
        stop_loss   = add_atr_below(H_prc)
        dist        = neck_avg - H_prc
        take_profit = neck_avg + dist
        pattern_score+=3

        results["inverse_head_and_shoulders"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": "LONG",
            "pattern_raw": inv
        })

    # -----------------------------------------------------
    # DOUBLE BOTTOM
    # -----------------------------------------------------
    dblbot_list = patterns_dict.get("double_bottom", [])
    dblbot_need = confirmed_map.get("double_bottom", [])
    if isinstance(dblbot_list, dict):
        dblbot_list = [dblbot_list]

    for i, db in enumerate(dblbot_list):
        if i not in dblbot_need:
            continue
        if not db.get("confirmed", False):
            continue
        bottoms = db.get("bottoms", [])
        min_bot = min(b[1] for b in bottoms)
        neck    = db.get("neckline",(None,None))
        cbar    = db.get("end_bar", last_i)

        if isinstance(neck, tuple) and len(neck)==2:
            neck_val = neck[1]
        else:
            neck_val = current_price

        entry_price = get_entry_on_breakout_or_retest(db, neck_val, cbar, side="long")
        stop_loss   = add_atr_below(min_bot)
        dist        = neck_val - min_bot
        take_profit = neck_val + dist
        pattern_score+=1

        results["double_bottom"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": "LONG",
            "pattern_raw": db
        })

    # -----------------------------------------------------
    # DOUBLE TOP
    # -----------------------------------------------------
    dbltop_list = patterns_dict.get("double_top", [])
    dbltop_need = confirmed_map.get("double_top", [])
    if isinstance(dbltop_list, dict):
        dbltop_list = [dbltop_list]

    for i, dt in enumerate(dbltop_list):
        if i not in dbltop_need:
            continue
        if not dt.get("confirmed", False):
            continue
        tops = dt.get("tops", [])
        max_top = max(t[1] for t in tops)
        neck    = dt.get("neckline", (None,None))
        cbar    = dt.get("end_bar", last_i)

        if isinstance(neck, tuple) and len(neck)==2:
            neck_val = neck[1]
        else:
            neck_val = current_price

        entry_price = get_entry_on_breakout_or_retest(dt, neck_val, cbar, side="short")
        stop_loss   = add_atr_above(max_top)
        dist        = max_top - neck_val
        take_profit = neck_val - dist
        pattern_score-=1

        results["double_top"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": "SHORT",
            "pattern_raw": dt
        })

    # -----------------------------------------------------
    # TRIPLE TOP ADVANCED
    # -----------------------------------------------------
    tta_list = patterns_dict.get("triple_top_advanced", [])
    tta_need = confirmed_map.get("triple_top_advanced", [])
    if isinstance(tta_list, dict):
        tta_list = [tta_list]

    for i, tta in enumerate(tta_list):
        if i not in tta_need:
            continue
        if not tta.get("confirmed", False):
            continue

        triple_tops = tta.get("tops", [])
        max_top = max(x[1] for x in triple_tops)
        neck    = tta.get("neckline",(None,None))
        cbar    = last_i

        if isinstance(neck, tuple) and len(neck)==2:
            neck_val= neck[1]
        else:
            neck_val= current_price

        entry_price= get_entry_on_breakout_or_retest(tta, neck_val, cbar, side="short")
        stop_loss= add_atr_above(max_top)
        dist= max_top- neck_val
        take_profit= neck_val- dist
        pattern_score-=1

        results["triple_top_advanced"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": "SHORT",
            "pattern_raw": tta
        })

    # -----------------------------------------------------
    # TRIPLE BOTTOM ADVANCED
    # -----------------------------------------------------
   # triple_bottom_advanced
    tba_list = patterns_dict.get("triple_bottom_advanced", [])
    tba_need = confirmed_map.get("triple_bottom_advanced", [])
    for i, tba in enumerate(tba_list):
        if i not in tba_need:
            continue
        if not tba.get("confirmed", False):
            continue

        triple_bots = tba.get("bottoms", [])
        min_bot = min(x[1] for x in triple_bots)
        neck = tba.get("neckline", (None,None))
        cbar = last_i

        if isinstance(neck, tuple) and len(neck)==2:
            neck_val = neck[1]
        else:
            neck_val = current_price

        # LONG mantık
        entry_price = get_entry_on_breakout_or_retest(tba, neck_val, cbar, side="long")
        stop_loss = add_atr_below(min_bot)
        dist = neck_val - min_bot
        take_profit = neck_val + dist
        pattern_score += 1

        results["triple_bottom_advanced"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": "LONG",
            "pattern_raw": tba
        })

    # -----------------------------------------------------
    # ELLIOTT
    # -----------------------------------------------------
    ell_data = patterns_dict.get("elliott", {})
    # confirmed_map içinden de bakabiliriz ama ell_data["found"] == True ise en azından trade oluşturmayı deneyebiliriz.
    # "0 in confirmed_map['elliott']" gibi bir kontrol de eklenebilir.
    if ell_data.get("found"):
        pivs = ell_data.get("pivots", [])
        trend = ell_data.get("trend", None)
        if len(pivs) == 5 and trend in ("UP","DOWN"):
            p0i, p0p = pivs[0]
            p1i, p1p = pivs[1]
            p2i, p2p = pivs[2]
            p3i, p3p = pivs[3]
            p4i, p4p = pivs[4]

            # Dalga boyları
            wave1_len = abs(p1p - p0p)
            wave2_len = abs(p2p - p1p)
            wave3_len = abs(p3p - p2p)
            wave4_len = abs(p4p - p3p)

            # En basit yaklaşımla => Entry her zaman p4'ün barının close'u
            entry_price = get_close(p4i)

            if trend == "UP":

                direction = "LONG"
                # Stop Loss: en yakın major dip = max(0..4) içinde en dip nokta hangisiyse,
                # pratikte wave2 veya wave4 pivotu dip olur. Biraz basit yoldan "p2p, p3p, p4p" gibi kıyaslayabiliriz.
                # Ama en güvenlisi wave2 & wave4. Minimum pivot:
                stop_candidate = min(p2p, p3p)
                stop_loss = add_atr_below(stop_candidate)

                # Take Profit: wave3 genelde en büyük dalga => wave5 de wave3 kadar gidebilir
                # Basit yaklaşım: p4 + wave3_len
                take_profit = p4p + wave3_len
                pattern_score+=1

                # Mantık dışı bir durum olmasın diye kontrol
                # (Örn. eğer wave3_len = 0'a yakınsa vs.)
                if take_profit <= entry_price:
                    # fallback: ATR kadar bir ekleme
                    take_profit = add_atr_tp_up(entry_price)

            else:
                direction = "SHORT"
                # Stop Loss: Down trendde en yakın major tepe wave2 veya wave3. 
                # Hangisi daha yüksekse oraya ATR ekleyerek SL koyabiliriz.
                top_candidate = max(p2p, p3p)
                stop_loss = add_atr_above(top_candidate)

                # Take Profit: p4 - wave3 uzunluğu (veya p4'ün bar closu - wave3).
                take_profit = p4p - wave3_len
                pattern_score-=1

                # Mantık dışı durum kontrolü
                if take_profit >= entry_price:
                    # Örn. wave3 negatif ölçülmüş olabilir ya da pivotlar sıralı değil.
                    # fallback: ATR vs. ekle 
                    take_profit = entry_price - (wave3_len * 0.5)
                    # Yine hala entry'den yüksekse, mecburen 
                    if take_profit >= entry_price:
                        take_profit = add_atr_tp_down(entry_price)

            # Sonuç kaydet
            results["elliott"].append({
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction,
                "pattern_raw": ell_data
            })

    # -----------------------------------------------------
    # WOLFE
    # -----------------------------------------------------
    wol_data = patterns_dict.get("wolfe", {})
    wol_need = confirmed_map.get("wolfe", [])
    if (
        wol_data and 
        (0 in wol_need) and 
        wol_data.get("found") and 
        wol_data.get("breakout")
    ):
        cbar = last_i
        w5 = wol_data.get("w5", (None, None))
        w5_prc = w5[1] if (w5 and len(w5) == 2) else current_price
        direction = wol_data.get("direction", None)  # "LONG"/"SHORT" bekleniyor

        if direction is None:
            direction = "LONG"  # fallback

        # Ufak yardımcı fonksiyon
        def get_entry_on_breakout_or_retest_wolfe(pattern_dict, default_line_val, break_bar, side=None):
            rinfo = pattern_dict.get("retest_info", {})
            if rinfo and rinfo.get("retest_done", False):
                rb = rinfo.get("retest_bar", break_bar)
                price = get_close(rb)
            else:
                price = get_close(break_bar)

            if side == "long":
                price = price * 1.0005
            elif side == "short":
                price = price * 0.9995
            return price

        if direction == "LONG":
            entry_price = get_entry_on_breakout_or_retest_wolfe(wol_data, w5_prc, cbar, side="long")
            stop_loss = add_atr_below(w5_prc)
            take_profit = add_atr_tp_up(entry_price*1.01)
            pattern_score+=1

        else:  # SHORT
            entry_price = get_entry_on_breakout_or_retest_wolfe(wol_data, w5_prc, cbar, side="short")
            stop_loss = add_atr_above(w5_prc)
            take_profit = add_atr_tp_down(entry_price*0.99)
            pattern_score-=1


        results["wolfe"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": wol_data
        })

    # -----------------------------------------------------
    # HARMONIC
    # -----------------------------------------------------
    harm_data= patterns_dict.get("harmonic", {})
    harm_need= confirmed_map.get("harmonic", [])
    if harm_data and (0 in harm_need) and harm_data.get("found"):
        xabc= harm_data.get("xabc", [])
        if len(xabc)==5:
            d_idx, d_price, _ = xabc[-1]
        else:
            d_price= current_price
            d_idx= last_i
        cbar= d_idx if (d_idx<len(df)) else last_i
        entry_price= get_close(cbar)
        stop_loss= add_atr_below(d_price)
        take_profit= d_price + 2*(last_atr if last_atr else 10)
        direction="LONG"
        pattern_score+=1

        results["harmonic"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": harm_data
        })

    # -----------------------------------------------------
    # TRIANGLE
    # -----------------------------------------------------
    tri_data= patterns_dict.get("triangle", {})
    tri_need= confirmed_map.get("triangle", [])
    if tri_data and (0 in tri_need) and tri_data.get("found"):
        tri_type= tri_data.get("triangle_type","symmetrical")
        breakout= tri_data.get("breakout",False)
        cbar= last_i
        if breakout:
            if tri_type=="ascending":
                direction="LONG"
                stop_loss= add_atr_below(current_price)
                take_profit= current_price + 2*(last_atr if last_atr else 10)
                pattern_score+=1

            elif tri_type=="descending":
                direction="SHORT"
                stop_loss= add_atr_above(current_price)
                take_profit= current_price - 2*(last_atr if last_atr else 10)
                pattern_score-=1

            else:
                direction=None
            if direction:
                entry_price= get_entry_on_breakout_or_retest(tri_data, current_price, cbar, side=direction.lower())
                results["triangle"].append({
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "direction": direction,
                    "pattern_raw": tri_data
                })

    # -----------------------------------------------------
    # WEDGE
    # -----------------------------------------------------
    wedge_data= patterns_dict.get("wedge", {})
    wedge_need= confirmed_map.get("wedge", [])
    if wedge_data and (0 in wedge_need) and wedge_data.get("found") and wedge_data.get("breakout"):
        wtype= wedge_data.get("wedge_type","rising")
        cbar= last_i
        if wtype=="rising":
            direction="SHORT"
            entry_price= get_entry_on_breakout_or_retest(wedge_data, current_price, cbar, side="short")
            stop_loss= add_atr_above(current_price)
            take_profit= current_price - 2*(last_atr if last_atr else 10)
            pattern_score-=1

        else:
            direction="LONG"
            entry_price= get_entry_on_breakout_or_retest(wedge_data, current_price, cbar, side="long")
            stop_loss= add_atr_below(current_price)
            take_profit= current_price + 2*(last_atr if last_atr else 10)
            pattern_score+=1


        results["wedge"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": wedge_data
        })

    # -----------------------------------------------------
    # CUP & HANDLE
    # -----------------------------------------------------
    cup_data= patterns_dict.get("cup_handle", {})
    cup_need= confirmed_map.get("cup_handle", [])
    if cup_data and (0 in cup_need) and cup_data.get("found"):
        rim_line= cup_data.get("rim_line")
        if rim_line and isinstance(rim_line, tuple):
            (i1,p1),(i2,p2)= rim_line
            rim_val= (p1+p2)/2
        else:
            rim_val= current_price

        cbar= last_i
        entry_price= get_entry_on_breakout_or_retest(cup_data, rim_val, cbar, side="long")
        cbot= cup_data.get("cup_bottom",(None,None))
        if isinstance(cbot, tuple) and len(cbot)==2:
            c_bottom= cbot[1]
        else:
            c_bottom= rim_val*0.9

        dist= rim_val- c_bottom
        stop_loss= add_atr_below(c_bottom)
        take_profit= rim_val + dist
        direction="LONG"
        pattern_score+=1


        results["cup_handle"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": cup_data
        })

    # -----------------------------------------------------
    # FLAG/PENNANT
    # -----------------------------------------------------
    fp_data= patterns_dict.get("flag_pennant", {})
    fp_need= confirmed_map.get("flag_pennant", [])
    if fp_data and (0 in fp_need) and fp_data.get("found"):
        direction= fp_data.get("direction","LONG")
        cbar= last_i
        entry_price= get_entry_on_breakout_or_retest(fp_data, current_price, cbar, side=direction.lower())
        if direction=="LONG":
            stop_loss= add_atr_below(current_price)
            take_profit= current_price + 2*(last_atr if last_atr else 10)
            pattern_score+=1

        else:
            stop_loss= add_atr_above(current_price)
            take_profit= current_price - 2*(last_atr if last_atr else 10)
            #pattern_score-=1
            #direction=="SHORT"                   
        results["flag_pennant"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": fp_data
        })

    # -----------------------------------------------------
    # CHANNEL
    # -----------------------------------------------------
    ch_data= patterns_dict.get("channel", {})
    ch_need= confirmed_map.get("channel", [])
    if ch_data and (0 in ch_need) and ch_data.get("found"):
        ctype= ch_data.get("channel_type","horizontal")
        cbar= last_i
        if ctype=="ascending":
            direction="LONG"
        elif ctype=="descending":
            direction="SHORT"
        else:
            direction=None

        if direction:
            entry_price= get_entry_on_breakout_or_retest(ch_data, current_price, cbar, side=direction.lower())
            if direction=="LONG":
                stop_loss= add_atr_below(current_price)
                take_profit= current_price + 2*(last_atr if last_atr else 10)
                pattern_score+=1
            else:
                stop_loss= add_atr_above(current_price)
                take_profit= current_price - 2*(last_atr if last_atr else 10)
                pattern_score-=1


            results["channel"].append({
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction,
                "pattern_raw": ch_data
            })

    # -----------------------------------------------------
    # GANN
    # -----------------------------------------------------
    gann_data= patterns_dict.get("gann", {})
    gann_need= confirmed_map.get("gann", [])
    if gann_data and (0 in gann_need) and gann_data.get("found"):
        cbar= last_i
        gln= gann_data.get("gann_line", None)
        anchor_price = gann_data.get("best_anchor", {}).get("anchor_price", None)

        # slope
        direction= "LONG"
        if gln and isinstance(gln, tuple):
            (ixA, pxA),(ixB, pxB) = gln
            slope = (pxB - pxA)/((ixB - ixA)+1e-9)
            direction = "LONG" if slope>0 else "SHORT"

        entry_price = get_entry_on_breakout_or_retest(gann_data, current_price, cbar, side=direction.lower())
        sl_atr = (last_atr * atr_sl_multiplier) if (last_atr and not isnan(last_atr)) else 0.0
        tp_atr = (last_atr * atr_tp_multiplier) if (last_atr and not isnan(last_atr)) else 0.0

        if anchor_price is not None:
            if direction=="LONG":
                distance = entry_price - anchor_price
                if distance <= 0:
                    # Ters duruma düşerse
                    direction="SHORT"
                    distance = anchor_price - entry_price
                    baseline_stop = anchor_price + (distance * 0.2)
                    stop_loss = baseline_stop + sl_atr
                    baseline_tp = entry_price - distance
                    take_profit = baseline_tp - tp_atr
                    pattern_score-=1

                else:
                    baseline_stop = anchor_price - (distance * 0.2)
                    stop_loss = baseline_stop - sl_atr
                    baseline_tp = entry_price + distance
                    take_profit = baseline_tp + tp_atr
                    pattern_score+=1

            else:
                distance = anchor_price - entry_price
                if distance <= 0:
                    direction="LONG"
                    distance = entry_price - anchor_price
                    baseline_stop = anchor_price - (distance * 0.2)
                    stop_loss = baseline_stop - sl_atr
                    baseline_tp = entry_price + distance
                    take_profit = baseline_tp + tp_atr
                    pattern_score+=1

                else:
                    baseline_stop = anchor_price + (distance * 0.2)
                    stop_loss = baseline_stop + sl_atr
                    baseline_tp = entry_price - distance
                    take_profit = baseline_tp - tp_atr
        else:
            if direction=="LONG":
                stop_loss  = add_atr_below(entry_price)
                take_profit= add_atr_tp_up(entry_price*1.02)
                pattern_score+=1

            else:
                stop_loss  = add_atr_above(entry_price)
                take_profit= add_atr_tp_down(entry_price*0.98)

        results["gann"].append({
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": gann_data
        })
    # -----------------------------------------------------
    # RECTANGLE
    # -----------------------------------------------------
    rect_data = patterns_dict.get("rectangle", {})
    rect_need = confirmed_map.get("rectangle", [])
    if rect_data and (0 in rect_need) and rect_data.get("found"):
        direction = rect_data.get("direction", None)
        if direction in ("UP","DOWN"):
            top_line = rect_data.get("top_line")   # ((i1, topPrice),(i2, topPrice))
            bot_line = rect_data.get("bot_line")   # ((j1, botPrice),(j2, botPrice))
            confirmed = rect_data.get("confirmed", False)
            cbar = len(df)-1
            # basit: bant genişliği
            if top_line and bot_line:
                # parse top_line
                # ((start_bar, topP), (end_bar, topP)) => ortalama top
                top_val = (top_line[0][1] + top_line[1][1]) / 2
                bot_val = (bot_line[0][1] + bot_line[1][1]) / 2
                band_height = top_val - bot_val

                # entry
                if direction=="UP":
                    side = "long"
                    entry_price = get_entry_on_breakout_or_retest(rect_data, top_val, cbar, side="long")
                    stop_loss   = add_atr_below(bot_val)
                    take_profit = entry_price + band_height
                    # Kaydedelim
                    results["rectangle"] = [{
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "direction": "LONG",
                        "pattern_raw": rect_data
                    }]
                else:  # direction=="DOWN"
                    side = "short"
                    entry_price = get_entry_on_breakout_or_retest(rect_data, bot_val, cbar, side="short")
                    stop_loss   = add_atr_above(top_val)
                    take_profit = entry_price - band_height
                    results["rectangle"] = [{
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "direction": "SHORT",
                        "pattern_raw": rect_data
                    }]

    return results,pattern_score


##################################################################
# 6) İndikatör Kontrolleri (RSI, MACD vb.)
##################################################################
def check_rsi_macd_for_trade(
    df: pd.DataFrame,
    direction: str,
    bar_index: int = -1,
    rsi_col: str = "RSI",
    macd_col: str = "MACD",
    macd_signal_col: str = "MACD_Signal",
    rsi_upper: float = 70.0,
    rsi_lower: float = 30.0
) -> dict:
    """
    Tek bir trade (direction=LONG/SHORT) için RSI/MACD filtrelemesi.
    """
    result = {
        "rsi_ok": True,
        "macd_ok": True,
        "overall_ok": True,
        "reasons": []
    }

    if rsi_col not in df.columns:
        result["rsi_ok"] = False
        result["overall_ok"] = False
        result["reasons"].append(f"{rsi_col} not found in DataFrame.")
        return result
    
    if macd_col not in df.columns or macd_signal_col not in df.columns:
        result["macd_ok"] = False
        result["overall_ok"] = False
        result["reasons"].append("MACD veya MACD_Signal kolonu yok.")
        return result

    if bar_index >= len(df):
        bar_index = len(df) - 1
    elif bar_index < 0:
        bar_index = len(df) + bar_index

    # RSI
    rsi_value = df[rsi_col].iloc[bar_index]
    macd_line = df[macd_col].iloc[bar_index]
    macd_signal = df[macd_signal_col].iloc[bar_index]
    #print("......",rsi_value,macd_line,macd_signal)
    # Basit RSI Mantığı
    if direction == "LONG":
        if rsi_value > rsi_upper:
            result["rsi_ok"] = False
            result["reasons"].append(f"RSI={rsi_value:.2f} > {rsi_upper} (Overbought)")
    else:  # SHORT
        if rsi_value < rsi_lower:
            result["rsi_ok"] = False
            result["reasons"].append(f"RSI={rsi_value:.2f} < {rsi_lower} (Oversold)")

    # Basit MACD Mantığı
    if direction == "LONG":
        if not (macd_line > macd_signal):
            result["macd_ok"] = False
            result["reasons"].append(
                f"MACD line ({macd_line:.4f}) <= MACD signal ({macd_signal:.4f}) for LONG"
            )
    else:  # SHORT
        if not (macd_line < macd_signal):
            result["macd_ok"] = False
            result["reasons"].append(
                f"MACD line ({macd_line:.4f}) >= MACD signal ({macd_signal:.4f}) for SHORT"
            )

    # Genel onay
    if not (result["rsi_ok"] and result["macd_ok"]):
        result["overall_ok"] = False
        return False
    return True
    #return result


def filter_trades_with_indicators(
    trade_levels_dict: dict,
    df: pd.DataFrame,
    time_frame,
    rsi_col: str = "RSI",
    macd_col: str = "MACD",
    macd_signal_col: str = "MACD_Signal"
) -> dict:
    """
    extract_pattern_trade_levels_filtered(...) çıktısını,
    RSI/MACD gibi indikatör koşulları ile filtreler.
    """
    filtered_results = {}
    for pattern_name, trades_list in trade_levels_dict.items():
        new_list = []
        for trade in trades_list:
            direction = trade.get("direction", None)
            if direction not in ("LONG", "SHORT"):
                trade["indicator_check"] = {
                    "overall_ok": False,
                    "reasons": ["No valid direction"]
                }
                new_list.append(trade)
                continue

            # Son bar üzerinden veya pattern'ın confirmed_bar'ı üzerinden
            # indikatör kontrolü yapılabilir.
            # check_res = check_rsi_macd_for_trade(
            #         df,
            #         direction,
            #         bar_index = -1,
            #         rsi_col= rsi_col,
            #         macd_col= macd_col,
            #         macd_signal_col = macd_signal_col,
            #         rsi_upper = 70.0,
            #         rsi_lower = 30.0
            #     )
            check_res= signal_rsi_macd_adx_bollinger_volume(df=df,direction=direction,
                                              time_frame= time_frame,
                                           
                                              )
            print(check_res)
            if check_res:

                new_list.append(trade)

        filtered_results[pattern_name] = new_list

    return filtered_results


def signal_rsi_macd_adx_bollinger_volume(
    df: pd.DataFrame,
    direction: str = "SHORT",
    time_frame: str = "15m"
) -> bool:
    """
    Belirtilen 'time_frame' için, ilgili indikatör parametrelerini 'timeframe_config' sözlüğünden alarak
    hem SHORT hem LONG sinyallerini üretir. Fonksiyon sonunda 'direction' parametresine göre
    son barda (iloc[-1]) sinyal var mı yok mu diye True/False döndürür.

    Parametreler:
      - df (DataFrame): time_frame'e göre 'Close_15m', 'High_15m' vs. kolonları içermeli.
      - direction (str) : "SHORT" veya "LONG". Son bar için bu yönde sinyal var mı diye bakar.
      - time_frame (str): "15m", "1h" gibi. timeframe_config sözlüğünde tanımlı olmalı.

    Dönüş:
      - bool: Son barda istenen yönde sinyal varsa True, yoksa False döndürür.
    """

    # 1) time_frame'e ait parametre setini sözlükten yükleyelim
    #    Eğer time_frame sözlükte yoksa varsayılan olarak "15m" setini alır
    config = timeframe_config.get(time_frame, timeframe_config["15m"])
   
    # 2) Sözlükten ilgili indikatör parametrelerini çekiyoruz
    ema_trend_period = config["ema_trend_period"]
    rsi_period       = config["rsi_period"]
    rsi_overbought   = config["rsi_overbought"]
    rsi_oversold     = config["rsi_oversold"]
    macd_fast        = config["macd_fast"]
    macd_slow        = config["macd_slow"]
    macd_signal      = config["macd_signal"]
    adx_period       = config["adx_period"]
    adx_threshold    = config["adx_threshold"]
    bb_period        = config["bb_period"]
    bb_stddev        = config["bb_stddev"]
    volume_window    = config["volume_window"]
    #print(ema_trend_period)
    # 3) DataFrame'i kopyalayalım (orijinali bozmamak için)
    df = df.copy()

    # ---------------------------------------------------
    # Göstergeleri hesaplayalım (EMA, RSI, MACD, ADX, BB)
    # ---------------------------------------------------
    df[f"EMA_{ema_trend_period}"] = talib.EMA(df[f"Close_{time_frame}"], timeperiod=ema_trend_period)
    print( df[f"EMA_{ema_trend_period}"])
    df["RSI"] = talib.RSI(df[f"Close_{time_frame}"], timeperiod=rsi_period)

    macd_line, macd_signal_line, macd_hist = talib.MACD(
        df[f"Close_{time_frame}"],
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=macd_signal
    )
    df["MACD"]      = macd_line
    df["MACD_Sig"]  = macd_signal_line
    df["MACD_Hist"] = macd_hist

    df["ADX"] = talib.ADX(
        df[f"High_{time_frame}"],
        df[f"Low_{time_frame}"],
        df[f"Close_{time_frame}"],
        timeperiod=adx_period
    )
    df["+DI"] = talib.PLUS_DI(
        df[f"High_{time_frame}"],
        df[f"Low_{time_frame}"],
        df[f"Close_{time_frame}"],
        timeperiod=adx_period
    )
    df["-DI"] = talib.MINUS_DI(
        df[f"High_{time_frame}"],
        df[f"Low_{time_frame}"],
        df[f"Close_{time_frame}"],
        timeperiod=adx_period
    )

    bb_up, bb_mid, bb_low = talib.BBANDS(
        df[f"Close_{time_frame}"],
        timeperiod=bb_period,
        nbdevup=bb_stddev,
        nbdevdn=bb_stddev
    )
    df["BB_Up"]  = bb_up
    df["BB_Mid"] = bb_mid
    df["BB_Low"] = bb_low

    df["VolumeMean"] = df[f"Volume_{time_frame}"].rolling(window=volume_window).mean()

    # 4) Sinyal kolonları oluşturuyoruz: Hem SHORT hem LONG
    df["ShortSignal"] = False
    df["LongSignal"]  = False

    # --------------------------------------------------
    # Barları dolaşarak Short ve Long sinyal koşullarını
    # ayrı ayrı kontrol edelim.
    # --------------------------------------------------
    for i in range(1, len(df)):
        close_prev = df[f"Close_{time_frame}"].iloc[i-1]
        close_curr = df[f"Close_{time_frame}"].iloc[i]

        rsi_prev = df["RSI"].iloc[i-1]
        rsi_curr = df["RSI"].iloc[i]

        macd_prev = df["MACD"].iloc[i-1]
        macd_sig_prev = df["MACD_Sig"].iloc[i-1]
        macd_curr = df["MACD"].iloc[i]
        macd_sig_curr = df["MACD_Sig"].iloc[i]

        adx_val  = df["ADX"].iloc[i]
        plus_di  = df["+DI"].iloc[i]
        minus_di = df["-DI"].iloc[i]

        bb_up_prev  = df["BB_Up"].iloc[i-1]
        bb_up_curr  = df["BB_Up"].iloc[i]
        bb_low_prev = df["BB_Low"].iloc[i-1]
        bb_low_curr = df["BB_Low"].iloc[i]

        volume_curr = df[f"Volume_{time_frame}"].iloc[i]
        volume_mean = df["VolumeMean"].iloc[i]

        # -----------------------
        # SHORT sinyali koşulları
        # -----------------------
        in_downtrend = (close_curr < df[f"EMA_{ema_trend_period}"].iloc[i])  # Fiyat EMA altında
        rsi_condition_short = (rsi_prev >= rsi_overbought) and (rsi_curr < rsi_overbought)
        macd_condition_short = (macd_prev > macd_sig_prev) and (macd_curr < macd_sig_curr)
        adx_condition_short = (adx_val >= adx_threshold) and (minus_di > plus_di)
        # Bollinger üst bandı test (örnek)
        boll_condition_short = (close_prev > bb_up_prev) and (close_curr < bb_up_curr)
        # Hacim
        volume_condition_short = (volume_curr > volume_mean)

        shortCondition = (
            in_downtrend and
            rsi_condition_short and
            macd_condition_short and
            #adx_condition_short and
            #boll_condition_short and
            volume_condition_short
        )

        if shortCondition:
            df.at[i, "ShortSignal"] = True

        # ----------------------
        # LONG sinyali koşulları
        # ----------------------
        in_uptrend = (close_curr > df[f"EMA_{ema_trend_period}"].iloc[i])  # Fiyat EMA üstünde
        rsi_condition_long = (rsi_prev <= rsi_oversold) and (rsi_curr > rsi_oversold)
        macd_condition_long = (macd_prev < macd_sig_prev) and (macd_curr > macd_sig_curr)
        adx_condition_long = (adx_val >= adx_threshold) and (plus_di > minus_di)
        # Bollinger alt bandı test (örnek)
        boll_condition_long = (close_prev < bb_low_prev) and (close_curr > bb_low_curr)
        # Hacim
        volume_condition_long = (volume_curr > volume_mean)

        longCondition = (
            in_uptrend and
            rsi_condition_long and
            macd_condition_long and
            #adx_condition_long and
            #boll_condition_long and
            volume_condition_long
        )

        if longCondition:
            df.at[i, "LongSignal"] = True

    # 5) Son bar (iloc[-1]) için direction'a göre True/False dönüyoruz.
    #    "SHORT" seçildiyse ShortSignal'e,
    #    "LONG" seçildiyse LongSignal'e bakıyoruz.

    if direction == "SHORT":
        return bool(df["ShortSignal"].iloc[-1])
    else:  # direction == "LONG"
        return bool(df["LongSignal"].iloc[-1])


timeframe_config = {
    "5m": {
        "ema_trend_period": 50,
        "rsi_period": 14,
        "rsi_overbought": 70.0,   # RSI>70 aşırı alım
        "rsi_oversold": 30.0,     # RSI<30 aşırı satım
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "adx_period": 14,
        "adx_threshold": 20.0,    # 20 üzeri trend gücü
        "bb_period": 20,
        "bb_stddev": 2,
        "volume_window": 20
    },
    "15m": {
        "ema_trend_period": 50,
        "rsi_period": 14,
        "rsi_overbought": 60.0,
        "rsi_oversold": 40.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "adx_period": 14,
        "adx_threshold": 20.0,    # 15m'de biraz daha yüksek eşik
        "bb_period": 20,
        "bb_stddev": 2,
        "volume_window": 20
    },
    "1h": {
        "ema_trend_period": 100,  # 1 saatlik grafikte daha uzun EMA
        "rsi_period": 14,
        "rsi_overbought": 70.0,
        "rsi_oversold": 30.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "adx_period": 14,
        "adx_threshold": 25.0,
        "bb_period": 20,
        "bb_stddev": 2,
        "volume_window": 24       # 24 bar = 1 günün saatleri, örneğin hacim ortalaması
    },
    "4h": {
        "ema_trend_period": 200,  # 4 saatlikte daha da uzun EMA
        "rsi_period": 14,
        "rsi_overbought": 70.0,
        "rsi_oversold": 30.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "bb_period": 20,
        "bb_stddev": 2,
        "volume_window": 28       # 4h için ~ 4x7 gün gibi bir yaklaşım
    },
    "1d": {
        "ema_trend_period": 200,  # Günlükte çok daha uzun EMA yaygın
        "rsi_period": 14,
        "rsi_overbought": 70.0,
        "rsi_oversold": 30.0,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "adx_period": 14,
        "adx_threshold": 20.0,
        "bb_period": 20,
        "bb_stddev": 2,
        "volume_window": 30       # 30 bar = 30 gün (1 ay) hacim ortalaması gibi
    }
}
