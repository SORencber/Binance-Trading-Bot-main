
# Log örneği
try:
    from core.logging_setup import log
except ImportError:
    def log(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

def final_score(df,patterns,time_frame,check_rsi_macd,
                v_spike,rsi_macd_signal,b_up,b_down,
                ml_label,max_bars_ago,wave,require_confirmed,trade_levels ):  
    pattern_score = 0
    reasons = []

    def filter_patterns(pat_list):
        if isinstance(pat_list, dict):
            # Tek dictionary ise, listeye dönüştürelim
            pat_list = [pat_list]
        elif not isinstance(pat_list, list):
            # String veya başka tür gelmiş => logla ve boş dön
            log(f"Pattern type mismatch => {pat_list}", "error")
            return []

        filtered = []
        cutoff = len(df) - max_bars_ago
        for p in pat_list:
            end_bar = p.get("end_bar", None)
            cbar = p.get("confirmed_bar", None)
            if end_bar is None:
                end_bar = cbar if cbar else 0
            if end_bar >= cutoff:
                if require_confirmed:
                    if p.get("confirmed", False):
                        filtered.append(p)
                else:
                    filtered.append(p)
        return filtered


    # 6) Pattern skorunu hesapla
    pattern_score = 0
    reasons = []

    # ----------------------------------------------------------------
    # HEAD & SHOULDERS
    # ----------------------------------------------------------------
    hs_list = filter_patterns(patterns["headshoulders"])
    for hs in hs_list:
        val = -4
        if hs["confirmed"] and hs.get("volume_check", True):
            val = -5
        pattern_score += val
        reasons.append(f"headshoulders({val})")

    inv_hs_list = filter_patterns(patterns["inverse_headshoulders"])
    for inv in inv_hs_list:
        val = +3
        if inv.get("confirmed") and inv.get("volume_check", True):
            val = +4
        pattern_score += val
        reasons.append(f"inverseHS({val})")

    # ----------------------------------------------------------------
    # DOUBLE / TRIPLE TOP-BOTTOM
    # ----------------------------------------------------------------
    dtops = filter_patterns(patterns["double_top"])
    for dt in dtops:
        val = -2
        if dt.get("confirmed"):
            val -= 3  # -3
        pattern_score += val
        reasons.append(f"double_top({val})")

    dbots = filter_patterns(patterns["double_bottom"])
    for db in dbots:
        val = +2
        if db.get("confirmed"):
            val += 1  # +3
        pattern_score += val
        reasons.append(f"double_bottom({val})")

    # ----------------------------------------------------------------
    # ELLIOTT
    # ----------------------------------------------------------------
    ell = patterns["elliott"]
    if ell["found"] and wave:
        if wave[-1][0] >= (len(df)-max_bars_ago):
            if ell["trend"]=="UP":
                pattern_score += 3
                reasons.append("elliott_up")
            else:
                pattern_score -= 3
                reasons.append("elliott_down")

    # ----------------------------------------------------------------
    # WOLFE
    # ----------------------------------------------------------------
    wol = patterns["wolfe"]
    if wol["found"] and wave:
        if wave[-1][0] >= (len(df)-max_bars_ago):
            wol_val = +2
            if wol.get("breakout"):
                wol_val += 1
            pattern_score += wol_val
            reasons.append(f"wolfe({wol_val})")

    # ----------------------------------------------------------------
    # HARMONIC
    # ----------------------------------------------------------------
    harm = patterns["harmonic"]
    if harm["found"] and wave:
        if wave[-1][0] >= (len(df)-max_bars_ago):
            pattern_score -= 1
            reasons.append("harmonic(-1)")

    # ----------------------------------------------------------------
    # TRIANGLE
    # ----------------------------------------------------------------
    tri = patterns["triangle"]
    if tri["found"] and tri.get("breakout", False) and wave:
        if wave[-1][0] >= (len(df)-max_bars_ago):
            if tri["triangle_type"]=="ascending":
                pattern_score += 1
                reasons.append("triangle_asc(+1)")
            elif tri["triangle_type"]=="descending":
                pattern_score -= 1
                reasons.append("triangle_desc(-1)")
            else:
                pattern_score += 1
                reasons.append("triangle_sym(+1)")

    # ----------------------------------------------------------------
    # WEDGE
    # ----------------------------------------------------------------
    wd = patterns["wedge"]
    if wd["found"] and wd.get("breakout", False) and wave:
        if wave[-1][0] >= (len(df)-max_bars_ago):
            if wd["wedge_type"]=="rising":
                pattern_score -= 1
                reasons.append("wedge_rising(-1)")
            else:
                pattern_score += 1
                reasons.append("wedge_falling(+1)")

    # ----------------------------------------------------------------
    # CUP & HANDLE (EKLENDİ)
    # ----------------------------------------------------------------
    cup_list = filter_patterns(patterns.get("cup_handle", []))
    for ch in cup_list:
        # Basit senaryoda Cup&Handle genelde bullish => +2
        # Confirmed vs. ek parametreyi yoklayabilirsiniz
        val = +2
        if ch.get("confirmed"):
            val += 1  # Cup&Handle onaylanmışsa +3
        pattern_score += val
        reasons.append(f"cup_handle({val})")

    # ----------------------------------------------------------------
    # FLAG / PENNANT (EKLENDİ)
    # ----------------------------------------------------------------
    flag_list = filter_patterns(patterns.get("flag_pennant", []))
    for fp in flag_list:
        # Flag/Pennant genelde trendin devamına işaret.
        # Örnek: eğer pattern directional="bull" vs. yoksa +2 diyelim
        val = +2
        if fp.get("confirmed"):
            val += 1
        pattern_score += val
        reasons.append(f"flag_pennant({val})")

    # ----------------------------------------------------------------
    # CHANNEL (EKLENDİ)
    # ----------------------------------------------------------------
    chn_list = filter_patterns(patterns.get("channel", []))
    for cn in chn_list:
        # Yükselen channel => bullish +1, Düşen => -1
        # Yatay channel => 0 (range)
        # Burada basit => "found" ise +1
        val = +1
        if cn.get("channel_type")=="descending":
            val = -1
        elif cn.get("channel_type")=="horizontal":
            val = 0
        pattern_score += val
        reasons.append(f"channel({val})")

    # ----------------------------------------------------------------
    # GANN (EKLENDİ - Placeholder)
    # ----------------------------------------------------------------
    gann_info = patterns.get("gann", {})
    if gann_info.get("found", False):
        # Gann Patterns => genelde manual analiz daha değerli
        # Basit => +1
        pattern_score += 1
        reasons.append("gann(+1)")

    # ----------------------------------------------------------------
    # ML LABEL => 0=HOLD, 1=BUY, 2=SELL
    # ----------------------------------------------------------------
    if ml_label == 1:
        pattern_score += 3
        reasons.append("ml_buy")
    elif ml_label == 2:
        pattern_score -= 3
        reasons.append("ml_sell")

    # ----------------------------------------------------------------
    # Breakout/hacim => eğer score >0 => potansiyel alış => breakout_up => +1
    # aksi => breakout_down => -1
    # ----------------------------------------------------------------
    final_score = pattern_score
    if final_score > 0:
        if b_up:
            final_score += 1
            reasons.append("breakout_up")
            if v_spike:
                final_score += 1
                reasons.append("vol_spike_up")
    elif final_score < 0:
        if b_down:
            final_score -= 1
            reasons.append("breakout_down")
            if v_spike:
                final_score -= 1
                reasons.append("vol_spike_down")

    # ----------------------------------------------------------------
    # RSI/MACD => eğer false ise skoru 1 puan düşürelim (basit)
    # ----------------------------------------------------------------
    if check_rsi_macd:
        if not rsi_macd_signal:
            final_score -= 1
            reasons.append("rsi_macd_fail")

    # 7) Son Karar
    final_signal = "HOLD"
    if final_score >= 2:
        final_signal = "BUY"
    elif final_score <= -2:
        final_signal = "SELL"
    
    reason_str = ",".join(reasons) if reasons else "NONE"

    return {
        "signal": final_signal,
        "score": final_score,
        "reason": reason_str,
        "patterns": patterns,
        "ml_label": ml_label,
        "breakout_up": b_up,
        "breakout_down": b_down,
        "volume_spike": v_spike,
        "time_frame": time_frame,
        "pattern_trade_levels": trade_levels
    }


def parse_line_or_point(val):
    """
    Pattern sonuçlarında kritik seviye (ör. neckline, rim_line) 
    şu formatlardan biri olabilir:
      - float/int => direkt seviye (örnek: 5.23)
      - (idx, price) => => price döndürür
      - ((ix1, px1), (ix2, px2)) => iki pivot noktası => ortalama fiyat
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
        if (
            len(val) == 2 
            and isinstance(val[0], tuple) 
            and isinstance(val[1], tuple)
        ):
            (i1, p1), (i2, p2) = val
            return (p1 + p2) / 2.0

    return None

def measure_pattern_distances(
    patterns_dict: dict,
    current_price: float,
    tolerance: float = 0.01
) -> list:
    """
    'patterns_dict': detect_all_patterns_v2(...) çıktısı.
    'current_price': anlık fiyat
    'tolerance':     oransal yakınlık eşiği (örn. 0.01 => %1)

    Dönüş => [
      {
        "pattern"         : "...",         # pattern adı (örn. "double_bottom")
        "sub_index"       : 0,             # pattern listesi içerisindeki bulgu indexi (list pattern’larda)
        "line_label"      : "neckline" vs.
        "line_value"      : <float>,       # parse_line_or_point(...) ile tekil float
        "confirmed"       : True/False,    # pattern’e göre
        "distance_ratio"  : 0.007,         # oransal mesafe
        "within_tolerance": True/False,    
        "pattern_raw"     : {...}          # orijinal pattern dict
      },
      ...
    ]

    Açıklama:
      - Liste döndüren pattern’lar (örn. "double_bottom", "headshoulders"): 
        her item için "neckline" veya benzeri bir line varsa ölçülür.
      - Tek dict döndüren pattern’lar (örn. "cup_handle", "elliott"): 
        “rim_line”, “wave4_level” vs. alanlar üzerinden ölçülür.
      - "incomplete_head_and_shoulders" => boyun çizgisi yok, ölçüm yok.
      - "harmonic" => net bir “line” yok (isterseniz D pivotunu ekleyebilirsiniz).
    """
    results = []

    # Küçük inline fonksiyon: line_val'ı parse edip mesafeyi hesaplar
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
        results.append(info)

    # 1) Head & Shoulders => liste => "neckline", "confirmed"
    if "headshoulders" in patterns_dict:
        hs_list = patterns_dict["headshoulders"]
        if isinstance(hs_list, list):
            for i, hs in enumerate(hs_list):
                neckline_raw = hs.get("neckline")
                lvl = parse_line_or_point(neckline_raw)
                confirmed = hs.get("confirmed", False)
                _add_result("headshoulders", i, "neckline", lvl, confirmed, hs)

    # 2) Inverse Head & Shoulders => liste => "neckline", "confirmed"
    if "inverse_headshoulders" in patterns_dict:
        invhs_list = patterns_dict["inverse_headshoulders"]
        if isinstance(invhs_list, list):
            for i, invhs in enumerate(invhs_list):
                neckline_raw = invhs.get("neckline")
                lvl = parse_line_or_point(neckline_raw)
                confirmed = invhs.get("confirmed", False)
                _add_result("inverse_headshoulders", i, "neckline", lvl, confirmed, invhs)

    # 3) Double Top => liste => "neckline", "confirmed"
    if "double_top" in patterns_dict:
        dt_list = patterns_dict["double_top"]
        if isinstance(dt_list, list):
            for i, dt in enumerate(dt_list):
                neckline_raw = dt.get("neckline")
                lvl = parse_line_or_point(neckline_raw)
                confirmed = dt.get("confirmed", False)
                _add_result("double_top", i, "neckline", lvl, confirmed, dt)

    # 4) Double Bottom => liste => "neckline", "confirmed"
    if "double_bottom" in patterns_dict:
        db_list = patterns_dict["double_bottom"]
        if isinstance(db_list, list):
            for i, db in enumerate(db_list):
                neckline_raw = db.get("neckline")
                lvl = parse_line_or_point(neckline_raw)
                confirmed = db.get("confirmed", False)
                _add_result("double_bottom", i, "neckline", lvl, confirmed, db)

    # 5) Incomplete Head & Shoulders => liste
    #    Burada boyun çizgisi vs. yok; "L", "H" ve "comment" var.
    #    Retest ya da confirmed da yok => Distance ölçmeye gerek yok, SKIP:
    if "incomplete_head_and_shoulders" in patterns_dict:
        inc_list = patterns_dict["incomplete_head_and_shoulders"]
        if isinstance(inc_list, list):
            # Sadece loglama amaçlı, eğer ölçüm yapmak isterseniz "L","H" pivotuna bakabilirsiniz.
            # Ama "boyun çizgisi" yok. Dolayısıyla distance eklemiyoruz.
            pass

    # 6) Elliott => tek dict => "wave4_level" (float?), "found" = True
    if "elliott" in patterns_dict:
        ell = patterns_dict["elliott"]
        if isinstance(ell, dict):
            if ell.get("found"):
                lvl = parse_line_or_point(ell.get("wave4_level"))
                confirmed = ell.get("found", False)  # found=True => benzer mantık
                _add_result("elliott", 0, "wave4_level", lvl, confirmed, ell)

    # 7) Wolfe Wave => tek dict => "wolfe_line", "breakout", "found"
    if "wolfe" in patterns_dict:
        wol = patterns_dict["wolfe"]
        if isinstance(wol, dict):
            if wol.get("found"):
                wline = wol.get("wolfe_line")
                lvl = parse_line_or_point(wline)
                confirmed = wol.get("breakout", False)  # breakout=True => onay
                _add_result("wolfe", 0, "wolfe_line", lvl, confirmed, wol)

    # 8) Wedge => tek dict => "breakout_line", "found", "breakout"
    if "wedge" in patterns_dict:
        wd = patterns_dict["wedge"]
        if isinstance(wd, dict):
            if wd.get("found"):
                wline = wd.get("breakout_line")
                lvl = parse_line_or_point(wline)
                confirmed = wd.get("breakout", False)
                _add_result("wedge", 0, "breakout_line", lvl, confirmed, wd)

    # 9) Triangle => tek dict => "breakout_line", "found", "breakout"
    if "triangle" in patterns_dict:
        tri = patterns_dict["triangle"]
        if isinstance(tri, dict):
            if tri.get("found"):
                line_ = tri.get("breakout_line")
                lvl = parse_line_or_point(line_)
                confirmed = tri.get("breakout", False)
                _add_result("triangle", 0, "breakout_line", lvl, confirmed, tri)

    # 10) Cup & Handle => tek dict => "found", "confirmed", "rim_line"
    if "cup_handle" in patterns_dict:
        cup = patterns_dict["cup_handle"]
        if isinstance(cup, dict):
            if cup.get("found"):
                rim_raw = cup.get("rim_line")
                lvl = parse_line_or_point(rim_raw)
                confirmed = cup.get("confirmed", False)
                _add_result("cup_handle", 0, "rim_line", lvl, confirmed, cup)

    # 11) Flag/Pennant => tek dict => "found", "pattern_type", "breakout_line"
    if "flag_pennant" in patterns_dict:
        flg = patterns_dict["flag_pennant"]
        if isinstance(flg, dict):
            if flg.get("found"):
                line_ = flg.get("breakout_line")
                lvl = parse_line_or_point(line_)
                confirmed = flg.get("confirmed", False)
                _add_result("flag_pennant", 0, "breakout_line", lvl, confirmed, flg)

    # 12) Channel => tek dict => "found", "breakout_line"
    if "channel" in patterns_dict:
        ch = patterns_dict["channel"]
        if isinstance(ch, dict):
            if ch.get("found"):
                line_ = ch.get("breakout_line")
                lvl = parse_line_or_point(line_)
                confirmed = ch.get("breakout", False)
                _add_result("channel", 0, "breakout_line", lvl, confirmed, ch)

    # 13) Gann => tek dict => "found", "gann_line"
    if "gann" in patterns_dict:
        gn = patterns_dict["gann"]
        if isinstance(gn, dict):
            if gn.get("found"):
                gl = gn.get("gann_line")
                lvl = parse_line_or_point(gl)
                confirmed = gn.get("found", False) 
                _add_result("gann", 0, "gann_line", lvl, confirmed, gn)

    # 14) Harmonic => genelde "found": bool; net çizgi yok
    #    İsterseniz D pivotu ekleyebilirsiniz, veyahut pass diyebilirsiniz.
    if "harmonic" in patterns_dict:
        hm = patterns_dict["harmonic"]
        if isinstance(hm, dict):
            if hm.get("found"):
                # Örneğin "xabc" listesinde son elemanın D pivotu olduğunu varsayıyoruz
                xabc = hm.get("xabc", [])
                if len(xabc) >= 4:
                    # Son pivot (D noktası) => xabc[-1]
                    pivotD = xabc[-1]
                    d_price = parse_line_or_point(pivotD)
                    if d_price is not None:
                        # Harmonic için "onaylı mı" gibi bir mantık yoksa 
                        # 'found' true kabul edebilirsiniz veya 'confirmed' alanı varsa onu kullanın.
                        confirmed = hm.get("confirmed", hm.get("found", False))
                        _add_result("harmonic", 0, "pivotD", d_price, confirmed, hm)


    return results

def filter_confirmed_within_tolerance(distances_list: list) -> dict:
    """
    'distances_list': measure_pattern_distances(...) fonksiyonundan gelen liste
      [
        {
          "pattern": "double_bottom",
          "sub_index": 0,
          "line_label": "neckline",
          "line_value": 0.5162,
          "confirmed": True,
          "distance_ratio": 0.007,
          "within_tolerance": True,
          "pattern_raw": {...}
        },
        ...
      ]

    Dönüş: {
      "double_bottom": [0, 1, 2],  # sub_index listesi
      "headshoulders": [0, ...],
      ...
    }
    Sadece confirmed=True ve within_tolerance=True olan pattern/subindex'leri tutar.
    """
    filtered_map = {}
    for d in distances_list:
        pname   = d["pattern"]
        sindex  = d.get("sub_index", 0)
        conf    = d["confirmed"]
        within  = d["within_tolerance"]
        if conf and within:
            if pname not in filtered_map:
                filtered_map[pname] = []
            filtered_map[pname].append(sindex)
    return filtered_map

def extract_pattern_trade_levels_filtered(
    patterns_dict: dict,
    confirmed_map: dict,
    current_price: float
) -> dict:
    """
    - 'patterns_dict': detect_all_patterns_v2(...) çıktı sözlüğü
    - 'confirmed_map': filter_confirmed_within_tolerance(...) çıktı sözlüğü
      (örnek: { "headshoulders": [0], "double_bottom": [1], ... })
    - 'current_price': float

    Yalnızca confirmed_map içinde yer alan pattern + sub_index için 
    basit stop_loss / take_profit hesaplayıp döndürür.
    """
    results = {
      "headshoulders": [],
      "inverse_headshoulders": [],
      "double_top": [],
      "double_bottom": [],
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

    # 1) Head & Shoulders => dict döndüğünü varsayalım (single)
    hs_needed = confirmed_map.get("headshoulders", [])
    hs_data   = patterns_dict.get("headshoulders")
    if isinstance(hs_data, dict) and 0 in hs_needed:
        # Örnek basit H&S stop/target
        if hs_data.get("confirmed"):
            # Head => hs_data["H"]
            h_price = hs_data["H"][1]
            (nx1, px1), (nx2, px2) = hs_data["neckline"]
            neck_avg = (px1 + px2)/2
            stop_loss   = h_price * 1.02
            take_profit = neck_avg - (h_price - neck_avg)
            entry_price = neck_avg
            results["headshoulders"].append({
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": "SHORT",
                "pattern_raw": hs_data
            })

    # 2) Inverse HS => liste formatında
    inv_list = patterns_dict.get("inverse_headshoulders", [])
    inv_needed = confirmed_map.get("inverse_headshoulders", [])
    if isinstance(inv_list, list) and len(inv_list)>0 and inv_needed:
        for idx, inv in enumerate(inv_list):
            if idx in inv_needed and inv.get("confirmed"):
                # Basit inverse HS
                h_price  = inv["H"][1]
                (nx1, px1), (nx2, px2) = inv["neckline"]
                neck_avg = (px1 + px2)/2
                stop_loss   = h_price*0.98
                take_profit = neck_avg + (neck_avg - h_price)
                entry_price = neck_avg
                results["inverse_headshoulders"].append({
                    "entry_price": entry_price,
                    "stop_loss":   stop_loss,
                    "take_profit": take_profit,
                    "direction":   "LONG",
                    "pattern_raw": inv
                })

    # 3) Double Bottom
    db_list = patterns_dict.get("double_bottom", [])
    db_needed= confirmed_map.get("double_bottom", [])
    for i, db in enumerate(db_list):
        if i in db_needed and db.get("confirmed"):
            dip_price = min(b[1] for b in db["bottoms"])
            neck = db.get("neckline")
            if neck:
                neck_price = neck[1]
                distance   = neck_price - dip_price
                take_profit= neck_price + distance
                entry_price= neck_price
            else:
                entry_price= current_price
                take_profit= current_price * 1.1
            stop_loss   = dip_price*0.97
            results["double_bottom"].append({
                "entry_price": entry_price,
                "stop_loss":   stop_loss,
                "take_profit": take_profit,
                "direction": "LONG",
                "pattern_raw": db
            })

    # 4) Double Top
    dt_list = patterns_dict.get("double_top", [])
    dt_needed= confirmed_map.get("double_top", [])
    for i, dt in enumerate(dt_list):
        if i in dt_needed and dt.get("confirmed"):
            peak_price = max(t[1] for t in dt["tops"])
            neck = dt.get("neckline")
            if neck:
                neck_price= neck[1]
                distance  = peak_price - neck_price
                take_profit= neck_price - distance
                entry_price= neck_price
            else:
                entry_price= current_price
                take_profit= current_price * 0.9
            stop_loss= peak_price*1.02
            results["double_top"].append({
                "entry_price": entry_price,
                "stop_loss":   stop_loss,
                "take_profit": take_profit,
                "direction": "SHORT",
                "pattern_raw": dt
            })

    # 5) Elliott
    ell_needed= confirmed_map.get("elliott", [])
    ell = patterns_dict.get("elliott", {})
    if 0 in ell_needed and ell.get("found"):
        trend = ell.get("trend")
        if trend=="UP":
            p4_price = ell["pivots"][3][1]
            p5_price = ell["pivots"][4][1]
            results["elliott"].append({
                "entry_price": current_price,
                "stop_loss": p4_price*0.95,
                "take_profit": p5_price,
                "direction": "LONG",
                "pattern_raw": ell
            })
        elif trend=="DOWN":
            p4_price = ell["pivots"][3][1]
            p5_price = ell["pivots"][4][1]
            results["elliott"].append({
                "entry_price": current_price,
                "stop_loss": p4_price*1.05,
                "take_profit": p5_price,
                "direction": "SHORT",
                "pattern_raw": ell
            })

    # 6) Wolfe
    wol_needed= confirmed_map.get("wolfe", [])
    wol = patterns_dict.get("wolfe", {})
    if 0 in wol_needed and wol.get("found") and wol.get("breakout"):
        w5 = wol.get("w5")
        if w5:
            w5_price= w5[1]
            stop_loss= w5_price*0.95
            take_profit= current_price*1.1
            results["wolfe"].append({
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": "LONG", 
                "pattern_raw": wol
            })

    # 7) Harmonic
    hrm_needed= confirmed_map.get("harmonic", [])
    harm = patterns_dict.get("harmonic", {})
    if 0 in hrm_needed and harm.get("found"):
        d_price = harm["xabc"][-1][1]
        stop_loss= d_price*0.97
        take_profit= d_price*1.2
        results["harmonic"].append({
            "entry_price": d_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": "LONG",
            "pattern_raw": harm
        })

    # 8) Triangle
    tri_needed= confirmed_map.get("triangle", [])
    tri = patterns_dict.get("triangle", {})
    if 0 in tri_needed and tri.get("found"):
        tri_type= tri.get("triangle_type")
        if tri_type=="ascending":
            results["triangle"].append({
                "entry_price": current_price,
                "stop_loss": current_price*0.95,
                "take_profit": current_price*1.15,
                "direction": "LONG",
                "pattern_raw": tri
            })
        elif tri_type=="descending":
            results["triangle"].append({
                "entry_price": current_price,
                "stop_loss": current_price*1.05,
                "take_profit": current_price*0.85,
                "direction": "SHORT",
                "pattern_raw": tri
            })
        else:
            # symmetrical => pass
            pass

    # 9) Wedge
    wd_needed= confirmed_map.get("wedge", [])
    wedge= patterns_dict.get("wedge", {})
    if 0 in wd_needed and wedge.get("found") and wedge.get("breakout"):
        wtype = wedge.get("wedge_type","")
        if wtype=="rising":
            direction="SHORT"
            stop_loss = current_price*1.05
            take_profit= current_price*0.85
        else:
            direction="LONG"
            stop_loss = current_price*0.95
            take_profit= current_price*1.15
        results["wedge"].append({
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": wedge
        })

    # 10) Cup & Handle
    cup_needed= confirmed_map.get("cup_handle", [])
    cup = patterns_dict.get("cup_handle", {})
    if 0 in cup_needed and cup.get("found"):
        direction="LONG"
        stop_loss= current_price*0.95
        take_profit= current_price*1.2
        results["cup_handle"].append({
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": cup
        })

    # 11) Flag/Pennant
    fp_needed= confirmed_map.get("flag_pennant", [])
    fp= patterns_dict.get("flag_pennant", {})
    if 0 in fp_needed and fp.get("found"):
        direction = fp.get("direction","LONG")
        if direction=="LONG":
            stop_loss= current_price*0.96
            take_profit= current_price*1.15
        else:
            stop_loss= current_price*1.04
            take_profit= current_price*0.85
        results["flag_pennant"].append({
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "direction": direction,
            "pattern_raw": fp
        })

    # 12) Channel
    ch_needed= confirmed_map.get("channel", [])
    ch= patterns_dict.get("channel", {})
    if 0 in ch_needed and ch.get("found"):
        ctype= ch.get("channel_type","horizontal")
        if ctype=="ascending":
            direction="LONG"
            stop_loss= current_price*0.95
            take_profit= current_price*1.15
        elif ctype=="descending":
            direction="SHORT"
            stop_loss= current_price*1.05
            take_profit= current_price*0.85
        else:
            direction="HOLD"
        if direction!="HOLD":
            results["channel"].append({
                "entry_price": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "direction": direction,
                "pattern_raw": ch
            })

    # 13) GANN
    gann_needed= confirmed_map.get("gann", [])
    gann_data= patterns_dict.get("gann", {})
    if 0 in gann_needed and gann_data.get("found"):
        results["gann"].append({
            "entry_price": current_price,
            "stop_loss": current_price*0.95,
            "take_profit": current_price*1.1,
            "direction": "LONG",
            "pattern_raw": gann_data
        })

    return results
