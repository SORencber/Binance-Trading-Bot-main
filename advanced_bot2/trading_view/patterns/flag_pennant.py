import pandas as pd
import numpy as np

def detect_flag_pennant_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    min_flagpole_bars: int = 15,
    impulse_pct: float = 0.05,
    max_cons_bars: int = 40,
    pivot_channel_tolerance: float = 0.02,
    pivot_triangle_tolerance: float = 0.02,
    require_breakout: bool = True,
    check_retest: bool = False,
    retest_tolerance: float = 0.01,
    # -- Gelişmiş Parametreler --
    volume_col: str = None,             # Hacim kolonu adı (ör: "Volume_1m")
    volume_drop_ratio: float = 0.5,     # Bayrak/Pennant oluşumu sırasında hacmin impuls döneme göre düşme oranı
    rsi_period: int = 14,              # RSI periyodu
    rsi_threshold: tuple = (30, 70),    # RSI aşırı alım/satım eşiği
    adx_period: int = 14,              # ADX periyodu
    adx_impulse_threshold: float = 20,  # İmpuls sırasında ADX min eşiği
    atr_period: int = 14,              # ATR periyodu
    atr_drop_ratio: float = 0.7,        # Konsolidasyon ATR ortalamasının, impuls ATR ortalamasına göre düşmesi beklenir
    # -- Multi Timeframe Parametreleri --
    higher_tf_df: pd.DataFrame = None,  # Daha büyük zaman dilimi DataFrame (Opsiyonel)
    higher_tf_adx_period: int = 14,     # Daha yüksek zaman dilimi ADX periyodu
    higher_tf_adx_min: float = 20,      # Daha yüksek zaman diliminde minimum ADX
    higher_tf_direction_confirm: bool = True,  # Daha yüksek zaman diliminde trend yönü ile uyum aranacak mı?
    # -- Fibonacci Kontrolü --
    fib_check: bool = False,           # Fibonacci düzeltme kontrolü yap
    fib_min: float = 0.382,            # min Fibonacci düzeltme (örn. 0.382)
    fib_max: float = 0.618,            # max Fibonacci düzeltme (örn. 0.618)
    # -- Pivot Katılığı --
    pivot_strictness: str = "normal",   # "strict", "loose", "normal" vs. pivot tespiti için ek parametre
    # -- Sahte Kırılım (Fake Breakout) Filtreleri --
    breakout_volume_factor: float = 1.2,   # Kırılım mumunda hacmin önceki ort. hacme göre en az ne kadar yüksek olması beklenir?
    breakout_body_factor: float = 1.5,     # Kırılım mumunun gövdesinin ortalama mum gövdesine göre büyüklüğü
    # vs. buraya ek parametreler yerleştirilebilir
) -> dict:
    """
    Gelişmiş/Ticari Seviye Bayrak-Pennant (Flama) Tespit Fonksiyonu
    
    Çok sayıda ek filtre ve multi-timeframe (MTF) kontroller içerir.
    'pivots' => [(index, price, pivot_type), ...] şeklinde pivot noktalarını tutan liste
    
    Geri dönüş:
    {
        "pattern": "flag_pennant",
        "found": bool,
        "direction": "bull"/"bear"/None,
        "pattern_type": "flag"/"pennant"/None,
        "consolidation_pivots": [],
        "upper_line": ((ixA, pxA), (ixB, pxB)),
        "lower_line": ((ixC, pxC), (ixD, pxD)),
        "confirmed": bool,
        "breakout_bar": int or None,
        "breakout_line": ((ix, px), (ix, px)) or None,
        "retest_info": { "retest_done": bool, "retest_bar": int or None } or None,
        "msgs": [liste halinde açıklama mesajları]
    }
    """
    # --------------------
    # 0) Dönüş Dict'i
    # --------------------
    result = {
        "pattern": "flag_pennant",
        "found": False,
        "direction": None,
        "pattern_type": None,
        "consolidation_pivots": [],
        "upper_line": None,
        "lower_line": None,
        "confirmed": False,
        "breakout_bar": None,
        "breakout_line": None,
        "retest_info": None,
        "msgs": []
    }

    # Hedef kolonları bul
    close_col = get_col_name("Close", time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"Missing {close_col}")
        return result
    
    n = len(df)
    # ----------------------------------------------------------------
    # 1) PIVOT TESPİTİ VE PIVOT KATILIĞI (opsiyonel) 
    # ----------------------------------------------------------------
    # pivot_strictness => "strict", "normal", "loose" gibi modlara göre 
    # pivotları süzebilir veya manipüle edebilirsiniz (daha gelişmiş pivot algılamada).
    # Örneğin "strict" => arka arkaya 2-3 bar onaylı pivot, vs.
    # Biz burada basit bir örnek veriyoruz:
    if pivot_strictness == "strict":
        # Örnek: Sadece belirli açılara sahip pivotları al vs...
        # (Gerçek hayatta daha gelişmiş bir pivot onay mekanizması ekleyebilirsiniz.)
        pivots = filter_strict_pivots(pivots)
    elif pivot_strictness == "loose":
        # Daha geniş bir pivot seti
        pivots = filter_loose_pivots(pivots)
    # normal => olduğu gibi

    # ---------------------------------------------------
    # 2) Temel kontrol: minimum bar sayısı
    # ---------------------------------------------------
    if n < min_flagpole_bars:
        result["msgs"].append("Not enough bars for flagpole check.")
        return result

    # ---------------------------------------------------
    # 3) İmpuls (bayrak direği) tespiti
    # ---------------------------------------------------
    start_i = n - min_flagpole_bars
    price_start = df[close_col].iloc[start_i]
    price_end = df[close_col].iloc[-1]
    pct_chg = (price_end - price_start) / (price_start + 1e-9)

    if abs(pct_chg) < impulse_pct:
        result["msgs"].append(f"No strong impulse (< {impulse_pct * 100}%).")
        return result

    direction = "bull" if (pct_chg > 0) else "bear"
    result["direction"] = direction

    # ---------------------------------------------------
    # 4) Konsolidasyon bölgesinin (bayrak/pennant) sınırlanması
    # ---------------------------------------------------
    cons_start = n - min_flagpole_bars
    cons_end = min(n - 1, cons_start + max_cons_bars)
    if cons_end <= cons_start:
        result["msgs"].append("Consolidation not enough bars.")
        return result

    cons_piv = [p for p in pivots if (p[0] >= cons_start and p[0] <= cons_end)]
    result["consolidation_pivots"] = cons_piv

    top_pivs = [p for p in cons_piv if p[2] == +1]
    bot_pivs = [p for p in cons_piv if p[2] == -1]
    if len(top_pivs) < 2 or len(bot_pivs) < 2:
        result["msgs"].append("Not enough top/bottom pivots => can't form mini-channel or triangle.")
        return result

    top_sorted = sorted(top_pivs, key=lambda x: x[0])
    bot_sorted = sorted(bot_pivs, key=lambda x: x[0])
    up1, up2 = top_sorted[0], top_sorted[1]
    dn1, dn2 = bot_sorted[0], bot_sorted[1]

    # ---------------------------------------------------
    # 5) Bayrak/Pennant çizgisi (üst ve alt) eğimleri
    # ---------------------------------------------------
    s_up = slope(up1[0], up1[1], up2[0], up2[1])
    s_dn = slope(dn1[0], dn1[1], dn2[0], dn2[1])
    if (s_up is None) or (s_dn is None):
        result["msgs"].append("Channel lines vertical => cannot form slope.")
        return result

    slope_diff = abs(s_up - s_dn) / (abs(s_up) + 1e-9)
    is_parallel = (slope_diff < pivot_channel_tolerance)
    is_opposite_sign = (s_up * s_dn < 0)

    upper_line = ((up1[0], up1[1]), (up2[0], up2[1]))
    lower_line = ((dn1[0], dn1[1]), (dn2[0], dn2[1]))
    result["upper_line"] = upper_line
    result["lower_line"] = lower_line

    pattern_type = None
    if is_parallel:
        pattern_type = "flag"
    elif is_opposite_sign and slope_diff > pivot_triangle_tolerance:
        pattern_type = "pennant"

    if not pattern_type:
        result["msgs"].append("No definitive mini-flag or mini-pennant.")
        return result

    result["pattern_type"] = pattern_type
    result["found"] = True  # En azından temel pattern bulundu

    # ---------------------------------------------------
    # 6) Gelişmiş Filtreler (RSI, ADX, ATR, Volume vb.)
    # ---------------------------------------------------

    # -- 6a) Hacim (Volume) Kontrolü --
    if volume_col and volume_col in df.columns:
        impulse_volume_slice = df[volume_col].iloc[start_i:n]  # impuls dönemi
        if len(impulse_volume_slice) == 0:
            impulse_volume = 1e-9
        else:
            impulse_volume = impulse_volume_slice.mean()

        cons_slice = df[volume_col].iloc[cons_start:cons_end+1]
        cons_volume = cons_slice.mean() if len(cons_slice) > 0 else 0
        
        if cons_volume > impulse_volume * volume_drop_ratio:
            msg_v = (f"Volume in consolidation not sufficiently lower than impulse. "
                     f"cons_vol={cons_volume:.2f} / impulse_vol={impulse_volume:.2f}")
            result["msgs"].append(msg_v)
            result["found"] = False
            return result

    # -- 6b) RSI Filtrelemesi --
    if rsi_period > 0:
        if "rsi_col" not in df.columns:
            df["rsi_col"] = compute_rsi(df[close_col], period=rsi_period)
        
        # İmpuls RSI ort.
        impulse_rsi_mean = df["rsi_col"].iloc[start_i:n].mean()
        # Konsolidasyon RSI ort.
        cons_rsi_mean = df["rsi_col"].iloc[cons_start:cons_end+1].mean()

        # Bull ise RSI impulse ort. çok düşükse pas geç, Bear ise çok yüksekse pas geç vb.
        low_thr, high_thr = rsi_threshold
        if direction == "bull" and impulse_rsi_mean < low_thr:
            result["msgs"].append(f"RSI impulse average < {low_thr} => not strong bullish.")
            result["found"] = False
            return result
        elif direction == "bear" and impulse_rsi_mean > high_thr:
            result["msgs"].append(f"RSI impulse average > {high_thr} => not strong bearish.")
            result["found"] = False
            return result

    # -- 6c) ADX Filtrelemesi --
    if adx_period > 0:
        if "adx_col" not in df.columns:
            df["adx_col"] = compute_adx(df, period=adx_period)
        adx_impulse_mean = df["adx_col"].iloc[start_i:n].mean()
        if adx_impulse_mean < adx_impulse_threshold:
            result["msgs"].append(f"ADX impulse avg < {adx_impulse_threshold} => trend gücü düşük.")
            result["found"] = False
            return result

    # -- 6d) ATR (Volatilite) Karşılaştırması --
    if atr_period > 0:
        if "atr_col" not in df.columns:
            df["atr_col"] = compute_atr(df, period=atr_period)
        impulse_atr_mean = df["atr_col"].iloc[start_i:n].mean()
        cons_atr_mean = df["atr_col"].iloc[cons_start:cons_end+1].mean()
        if cons_atr_mean > impulse_atr_mean * atr_drop_ratio:
            msg_atr = (f"Consolidation ATR not sufficiently lower than impulse ATR. "
                       f"cons_atr={cons_atr_mean:.4f}, impulse_atr={impulse_atr_mean:.4f}")
            result["msgs"].append(msg_atr)
            result["found"] = False
            return result

    # ---------------------------------------------------
    # 7) Multi-Timeframe (MTF) Onayı 
    # ---------------------------------------------------
    if higher_tf_df is not None and higher_tf_direction_confirm:
        # Daha büyük zaman diliminde ADX ve trend yönü kontrolü
        if "adx_col_higher" not in higher_tf_df.columns:
            higher_tf_df["adx_col_higher"] = compute_adx(higher_tf_df, period=higher_tf_adx_period)
        
        # Son veriye yakın ADX ortalaması
        # (örneğin son 10 barı alabiliriz, ya da son X bar.)
        last_x = 10
        adx_htf_recent = higher_tf_df["adx_col_higher"].iloc[-last_x:].mean()
        
        if adx_htf_recent < higher_tf_adx_min:
            msg_mtf = f"HTF ADX < {higher_tf_adx_min}, higher timeframe trend is weak."
            result["msgs"].append(msg_mtf)
            result["found"] = False
            return result
        
        # HTF yönü tespiti (basitçe: son close > son 50 SMA => bull, else => bear vb.)
        # Tabii ki daha gelişmiş bir metodla (ör. son pivot) da bakabilirsiniz.
        if "Close" not in higher_tf_df.columns:
            # Ufak bir koruma
            pass
        else:
            htf_direction = get_trend_direction(higher_tf_df["Close"], short_window=20, long_window=50)
            # direction: "bull" / "bear"
            if htf_direction != direction:
                msg_mtf2 = f"HTF direction={htf_direction} but LTF direction={direction}; conflict."
                result["msgs"].append(msg_mtf2)
                result["found"] = False
                return result

    # ---------------------------------------------------
    # 8) Fibonacci Düzeltme Kontrolü (Opsiyonel)
    # ---------------------------------------------------
    if fib_check:
        # Impuls dalgası (start_i -> n-1) aralığı
        high_price = df[close_col].iloc[start_i:n].max()
        low_price  = df[close_col].iloc[start_i:n].min()
        if direction == "bull":
            # bull => low'dan high'a bak
            fib_range = high_price - low_price
            # Konsolidasyonun alt/high ortalamasını alınabilir, ya da en son close vs.
            cons_min = df[close_col].iloc[cons_start:cons_end+1].min()
            cons_max = df[close_col].iloc[cons_start:cons_end+1].max()
            # En az 0.382, en fazla 0.618 civarı düzeltme
            # Düzeltme: (current_low - impuls_low)/fib_range vs.
            # Basit bir check olarak cons_min'i veya son_close'u referans alabilirsiniz.
            corr_amount = (cons_min - low_price) / (fib_range + 1e-9)
            if corr_amount < fib_min or corr_amount > fib_max:
                msg_fib = f"Fib correction not in [{fib_min}, {fib_max}]. Actual={corr_amount:.3f}"
                result["msgs"].append(msg_fib)
                result["found"] = False
                return result
        else:
            # bear => high'dan low'a bak
            fib_range = high_price - low_price
            cons_min = df[close_col].iloc[cons_start:cons_end+1].min()
            cons_max = df[close_col].iloc[cons_start:cons_end+1].max()
            # Düzeltme: (high_price - cons_max)/fib_range
            corr_amount = (high_price - cons_max) / (fib_range + 1e-9)
            if corr_amount < fib_min or corr_amount > fib_max:
                msg_fib = f"Fib correction not in [{fib_min}, {fib_max}]. Actual={corr_amount:.3f}"
                result["msgs"].append(msg_fib)
                result["found"] = False
                return result

    # ---------------------------------------------------
    # 9) Breakout Kontrolü
    # ---------------------------------------------------
    if not require_breakout:
        return result

    last_i = n - 1
    last_close = df[close_col].iloc[-1]
    up_line_last = line_val(upper_line[0], upper_line[1], last_i)
    dn_line_last = line_val(lower_line[0], lower_line[1], last_i)

    conf = False
    brk_bar = None
    if direction == "bull":
        if last_close > up_line_last:
            conf = True
            brk_bar = last_i
    else:
        if last_close < dn_line_last:
            conf = True
            brk_bar = last_i

    result["confirmed"] = conf
    result["breakout_bar"] = brk_bar
    if conf:
        if direction == "bull":
            result["breakout_line"] = upper_line
        else:
            result["breakout_line"] = lower_line

        # ---------------------------------------------------
        # 9a) Sahte Kırılım (Fake Breakout) Filtreleri
        # ---------------------------------------------------
        # Örnek: breakout mumunun hacmi, ortalama hacmin en az X katı olmalı
        # veya breakout mumunun gövdesi ortalama mum gövdesine göre vs.
        if volume_col and (breakout_volume_factor > 1.0):
            recent_volume_mean = df[volume_col].iloc[-20:].mean()  # son 20 bar ortalama hacim
            breakout_volume = df[volume_col].iloc[-1]
            if breakout_volume < breakout_volume_factor * recent_volume_mean:
                msg_break_vol = (f"Breakout volume not large enough. "
                                 f"{breakout_volume:.2f} < {breakout_volume_factor} * {recent_volume_mean:.2f}")
                result["msgs"].append(msg_break_vol)
                result["confirmed"] = False
                return result

        # Gövde büyüklüğü vs. (örn. close - open)
        open_col = get_col_name("Open", time_frame)
        if breakout_body_factor > 1.0 and open_col in df.columns:
            # son 20 barın ortalama gövde büyüklüğü
            df["body_size"] = (df[close_col] - df[open_col]).abs()
            avg_body_20 = df["body_size"].iloc[-20:].mean()
            breakout_body = df["body_size"].iloc[-1]
            if breakout_body < breakout_body_factor * avg_body_20:
                msg_break_body = (f"Breakout candle body not large enough. "
                                  f"{breakout_body:.2f} < {breakout_body_factor} * {avg_body_20:.2f}")
                result["msgs"].append(msg_break_body)
                result["confirmed"] = False
                return result

        # ---------------------------------------------------
        # 10) Retest Kontrolü
        # ---------------------------------------------------
        if check_retest and result["breakout_line"]:
            (ixA, pxA), (ixB, pxB) = result["breakout_line"]
            mF, bF = line_equation(ixA, pxA, ixB, pxB)
            if mF is not None:
                retest_done = False
                retest_bar = None
                for i in range(ixB + 1, n):
                    c = df[close_col].iloc[i]
                    line_y = mF * i + bF
                    diff_perc = abs(c - line_y) / (abs(line_y) + 1e-9)
                    if diff_perc <= retest_tolerance:
                        retest_done = True
                        retest_bar = i
                        break
                result["retest_info"] = {
                    "retest_done": retest_done,
                    "retest_bar": retest_bar
                }

    return result

# ---------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def slope(x1, y1, x2, y2):
    if (x2 - x1) == 0:
        return None
    return (y2 - y1) / (x2 - x1)

def line_val(p1, p2, x: int):
    """
    p1 = (ix1, px1), p2 = (ix2, px2)
    """
    (ix1, px1) = p1
    (ix2, px2) = p2
    if (ix2 - ix1) == 0:
        return px1
    m = (px2 - px1) / (ix2 - ix1)
    b = px1 - m * ix1
    return m * x + b

def line_equation(x1, y1, x2, y2):
    """
    Returns slope (m) and intercept (b) of the line y = m*x + b
    If x2 == x1 => returns (None, None)
    """
    if x2 == x1:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return m, b

def compute_rsi(series: pd.Series, period: int = 14):
    """
    Basit RSI hesabı (EMA tabanlı).
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df: pd.DataFrame, period: int = 14):
    """
    Basit ATR hesabı.
    High, Low, Close kolonları olduğu varsayıldı (veya _timeframe suffix'li).
    """
    # Kolonları bulmaya çalış
    high_col = [c for c in df.columns if "High" in c][0]
    low_col = [c for c in df.columns if "Low" in c][0]
    close_col = [c for c in df.columns if "Close" in c][0]

    high = df[high_col]
    low = df[low_col]
    close_prev = df[close_col].shift(1).bfill()


    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr

def compute_adx(df: pd.DataFrame, period: int = 14):
    """
    Basit ADX hesabı.
    """
    high_col = [c for c in df.columns if "High" in c][0]
    low_col = [c for c in df.columns if "Low" in c][0]
    close_col = [c for c in df.columns if "Close" in c][0]

    high = df[high_col]
    low = df[low_col]
    close_prev = df[close_col].shift(1)

    plus_dm = (high - high.shift(1)).clip(lower=0)
    minus_dm = (low.shift(1) - low).clip(lower=0)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) ) * 100
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx

def filter_strict_pivots(pivots):
    """
    Örnek: Daha katı kurallara göre pivotları filtreleyin.
    (Gerçek hayatta kendi pivot onay mekanizmanızı buraya yazabilirsiniz.)
    """
    # Bu sadece basit bir örnek, gerçekte "yakın pivotlar"ı elemek, min mesafe vb. yapabilirsiniz.
    return pivots

def filter_loose_pivots(pivots):
    """
    Örnek: Daha gevşek kurallara göre pivotları bırakın.
    """
    return pivots

def get_trend_direction(series: pd.Series, short_window=20, long_window=50):
    """
    Basit bir 'trend direction' fonksiyonu.
    short_window SMA > long_window SMA => bull, else => bear
    """
    sma_short = series.rolling(short_window).mean()
    sma_long = series.rolling(long_window).mean()
    if sma_short.iloc[-1] > sma_long.iloc[-1]:
        return "bull"
    else:
        return "bear"
