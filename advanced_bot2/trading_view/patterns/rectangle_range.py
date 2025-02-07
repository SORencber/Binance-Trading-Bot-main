import pandas as pd
def detect_rectangle_range(
    df: pd.DataFrame,
    pivots: list,
    time_frame: str = "1m",
    parallel_thresh: float = 0.01,
    min_top_pivots: int = 2,
    min_bot_pivots: int = 2,
    min_bars_width: int = 10,
    max_bars_width: int = 300,
    breakout_confirm: bool = True,
    check_retest: bool = True,
    retest_tolerance: float = 0.01
) -> dict:
    """
    Dikdörtgen (Yatay Kanal) formasyonunu tespit eden fonksiyon.
    * Fiyat bir süre yatay bir üst bant (direnç) ve alt bant (destek) arasında konsolide olur.
    * Kırılma anında breakout testi yapılır.
    
    Dönen sözlükteki alanlar:
        {
          "pattern": "rectangle",
          "found": bool,
          "confirmed": bool,
          "top_line": ((x1,y1),(x2,y2)),
          "bot_line": ((x1b,y1b),(x2b,y2b)),
          "bar_start": int,
          "bar_end": int,
          "width_bars": int,
          "direction": "UP"/"DOWN"/None,   # kırılım yönü
          "breakout_bar": int veya None,
          "retest_info": { ... } veya None,
          "msgs": [list of str]
        }
    """
    result = {
        "pattern": "rectangle",
        "found": False,
        "confirmed": False,
        "top_line": None,
        "bot_line": None,
        "bar_start": None,
        "bar_end": None,
        "width_bars": 0,
        "direction": None,
        "breakout_bar": None,
        "retest_info": None,
        "msgs": []
    }
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        result["msgs"].append(f"Missing {close_col} column in DataFrame.")
        return result

    # 1) Uygun sayıda pivot var mı?
    top_piv = [p for p in pivots if p[2] == +1]
    bot_piv = [p for p in pivots if p[2] == -1]
    if len(top_piv) < min_top_pivots or len(bot_piv) < min_bot_pivots:
        result["msgs"].append("Not enough top/bottom pivots.")
        return result

    # 2) Zaman sıralamasına göre pivotlar
    top_sorted = sorted(top_piv, key=lambda x: x[0])
    bot_sorted = sorted(bot_piv, key=lambda x: x[0])
    first_top = top_sorted[0]   # (bar_idx, price)
    last_top  = top_sorted[-1]
    first_bot = bot_sorted[0]
    last_bot  = bot_sorted[-1]

    # 3) Aralık (start_bar, end_bar)
    start_bar = min(first_top[0], first_bot[0])
    end_bar   = max(last_top[0], last_bot[0])
    width_bars = end_bar - start_bar
    if width_bars < min_bars_width or width_bars > max_bars_width:
        result["msgs"].append(
            f"width_bars={width_bars} not in [{min_bars_width}, {max_bars_width}]"
        )
        return result

    # 4) Üst çizgi (direnç) ve alt çizgi (destek) neredeyse yatay mı?
    #    Basit yaklaşım => top_pivots ortalama price, bot_pivots ortalama price al, 
    #                     sanki birer yatay çizgi gibi işleme koy.
    #    Dilerseniz line_equation ile min_2 pivot parametrelerini alıp slope < parallel_thresh gibi bakabilirsiniz.
    mean_top = sum([t[1] for t in top_piv]) / len(top_piv)
    mean_bot = sum([b[1] for b in bot_piv]) / len(bot_piv)
    if mean_top <= mean_bot:
        result["msgs"].append("mean_top <= mean_bot => not valid rectangle.")
        return result

    # slope check => en az iki pivot bulalım (örnek: first_top & last_top):
    # top line
    ft_idx, ft_prc = first_top[0], first_top[1]
    lt_idx, lt_prc = last_top[0], last_top[1]
    slope_top = (lt_prc - ft_prc) / (lt_idx - ft_idx + 1e-9)
    if abs(slope_top) > parallel_thresh:
        result["msgs"].append(
            f"top slope {slope_top:.5f} > parallel_thresh={parallel_thresh}"
        )
        return result

    # bottom line
    fb_idx, fb_prc = first_bot[0], first_bot[1]
    lb_idx, lb_prc = last_bot[0], last_bot[1]
    slope_bot = (lb_prc - fb_prc) / (lb_idx - fb_idx + 1e-9)
    if abs(slope_bot) > parallel_thresh:
        result["msgs"].append(
            f"bot slope {slope_bot:.5f} > parallel_thresh={parallel_thresh}"
        )
        return result

    # 5) found => True
    result["found"] = True
    # Yatay çizgileri temsil eden line (bar_start -> bar_end)
    # Basit => (start_bar, mean_top) -> (end_bar, mean_top)
    # Ama isterseniz ilk pivot ile son pivot line_equation da kullanabilirsiniz
    # for better accuracy. Aşağıda ortalama alıyoruz (daha basit).
    result["top_line"] = ((start_bar, mean_top), (end_bar, mean_top))
    result["bot_line"] = ((start_bar, mean_bot), (end_bar, mean_bot))
    result["bar_start"] = start_bar
    result["bar_end"]   = end_bar
    result["width_bars"] = width_bars

    # 6) Breakout Kontrolü
    if not breakout_confirm:
        return result  # confirmed=False, normal bulduk

    n = len(df)
    last_i = n - 1
    last_close = df[close_col].iloc[-1]

    # a) Yukarı kırılım?
    if last_close > mean_top:
        result["confirmed"] = True
        result["direction"] = "UP"
        result["breakout_bar"] = last_i
    elif last_close < mean_bot:
        result["confirmed"] = True
        result["direction"] = "DOWN"
        result["breakout_bar"] = last_i
    else:
        # Hala dikdörtgen içinde
        result["msgs"].append("No breakout => still in rectangle range.")

    # 7) Retest Kontrolü
    if check_retest and result["confirmed"]:
        # retest => if up => retest top line
        # if down => retest bottom line
        retest_done = False
        retest_bar  = None

        # line sabit => y= mean_top (veya mean_bot)
        line_level = mean_top if result["direction"]=="UP" else mean_bot
        for i in range(result["breakout_bar"]+1, n):
            c = df[close_col].iloc[i]
            dist_ratio = abs(c - line_level)/(abs(line_level)+1e-9)
            if dist_ratio <= retest_tolerance:
                retest_done = True
                retest_bar = i
                break
        result["retest_info"] = {
            "retest_done": retest_done,
            "retest_bar" : retest_bar
        }

    return result
