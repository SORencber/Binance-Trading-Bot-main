import pandas as pd
from core.logging_setup import log

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def build_zigzag_wave(pivots, zigzag_min_delta: float = 0.0):
    """
    Belirli bir min delta olmadan (varsayılan 0), orijinal fonksiyon gibi çalışır.
    zigzag_min_delta > 0 ise (ör. %0.5 = 0.005), 
    arka arkaya pivotlar arasında yeterli fark yoksa pivot oluşturma gürültüsü azaltılır.
    pivots: List[ (index, price, pivotType), ... ]
    pivotType: +1 => tepe, -1 => dip
    """
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda x: x[0])
    wave = [sorted_p[0]]
    for i in range(1, len(sorted_p)):
        curr = sorted_p[i]
        prev = wave[-1]

        # Aynı tip pivot ise, en uç (tepe/dip) değeri güncelle
        if curr[2] == prev[2]:
            # tepe ise en yükseği al
            if curr[2] == +1:
                if curr[1] > prev[1]:
                    # yeterli fark var mı?
                    if zigzag_min_delta > 0:
                        if (abs(curr[1] - prev[1]) / (abs(prev[1]) + 1e-9)) >= zigzag_min_delta:
                            wave[-1] = curr
                    else:
                        wave[-1] = curr
            else:
                # dip ise en düşüğü al
                if curr[1] < prev[1]:
                    if zigzag_min_delta > 0:
                        if (abs(curr[1] - prev[1]) / (abs(prev[1]) + 1e-9)) >= zigzag_min_delta:
                            wave[-1] = curr
                    else:
                        wave[-1] = curr
        else:
            # tip değiştiyse yeni pivot ekle
            # yine istenirse zigzag_min_delta kontrolü yapabiliriz
            if zigzag_min_delta > 0:
                if (abs(curr[1] - prev[1]) / (abs(prev[1]) + 1e-9)) >= zigzag_min_delta:
                    wave.append(curr)
            else:
                wave.append(curr)

    return wave


def _check_retest_elliott_wave4(
    df: pd.DataFrame,
    time_frame: str,
    wave4_index: int,
    wave4_price: float,
    tolerance: float=0.01,
    trend: str="UP"
):
    """
    Dalga4 seviyesine sonraki barlarda bir 'retest' var mı?
    wave4_index sonrasındaki barların Close'u wave4_price'a ne kadar yaklaşmış, vb. kontrol.
    """
    close_col = get_col_name("Close", time_frame)
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}
    
    n = len(df)
    if wave4_index >= n-1:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    retest_done = False
    retest_bar = None
    retest_dist = None
    
    for i in range(wave4_index + 1, n):
        c = df[close_col].iloc[i]
        dist_ratio = abs(c - wave4_price) / (abs(wave4_price) + 1e-9)
        if dist_ratio <= tolerance:
            retest_done = True
            retest_bar = i
            retest_dist = dist_ratio
            break

    return {
        "retest_done": retest_done,
        "retest_bar": retest_bar,
        "retest_price": wave4_price,
        "distance_ratio": retest_dist
    }

############################
# ELLIOTT 5 WAVE (Güncel / Geliştirilmiş)
############################

def detect_elliott_5wave_strict(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    fib_tolerance: float = 0.1,
    wave_min_bars: int = 5,
    # Elliott katı kuralları
    rule_3rdwave_min_percent: float = 1.618,
    rule_3rdwave_not_shortest: bool = True,
    allow_4th_overlap: bool = False,
    allow_wave2_above_wave1_start: bool = False,  # wave2, wave1'in başlangıcını geçmemeli (klasik kural)
    # Fibonacci tipik aralıklar
    wave2_fib_range: tuple = (0.382, 0.618),
    wave4_fib_range: tuple = (0.382, 0.618),
    fib_tolerance_range: float = 0.1,  # dalga2/dalga4 retracement aralık toleransı
    # Uzatılmış dalga vb. opsiyonel
    check_extended_5th: bool = True,
    rule_5thwave_ext_range: tuple = (1.0, 1.618),
    # ABC isteğe bağlı
    check_abc_correction: bool = True,
    # Bar aralığı
    min_bar_distance: int = 3,
    # Retest kontrolü
    check_retest: bool = False,
    retest_tolerance: float = 0.01,
    # ZigZag min delta
    zigzag_min_delta: float = 0.005,
    # Ek: entry/stop hesaplaması
    calc_trade_levels: bool = True,
    stop_loss_buffer: float = 0.01  # dalga4'ün bir miktar altı/üstü
):
    """
    Daha katı (ve ticari kullanım için) Elliott 5-dalga tespit fonksiyonu.
    Tüm temel Elliott kuralları sağlanmazsa found=False döner.
    
    Geri dönüş: 
    {
        "pattern": "elliott",
        "found": bool,
        "trend": "UP" veya "DOWN",
        "pivots": [(i0,p0), (i1,p1), ...],
        "check_msgs": [...],
        "abc": bool veya None,
        "extended_5th": bool,
        "wave4_level": float,
        "retest_info": {...} veya None,
        -- opsiyonel trade seviyeleri --
        "entry_price": float veya None,
        "stop_loss": float veya None
    }
    """
    result = {
        "pattern": "elliott",
        "found": False,
        "trend": None,
        "pivots": [],
        "check_msgs": [],
        "abc": None,
        "extended_5th": False,
        "wave4_level": None,
        "retest_info": None,
        "entry_price": None,  # eklendi
        "stop_loss": None     # eklendi
    }

    # 1) Zigzag wave oluştur (gürültü filtreleme için zigzag_min_delta kullanılabilir)
    wave = build_zigzag_wave(pivots, zigzag_min_delta=zigzag_min_delta)
    if len(wave) < wave_min_bars:
        result["check_msgs"].append("Not enough pivots for Elliott 5-wave.")
        return result

    # 2) Son 5 pivotun tipi (yukarı +1 / aşağı -1) 5-dalga sıralamasına uymalı
    last5 = wave[-5:]
    types = [p[2] for p in last5]
    up_pattern   = [+1, -1, +1, -1, +1]
    down_pattern = [-1, +1, -1, +1, -1]

    # Trend tespiti
    if types == up_pattern:
        trend = "UP"
    elif types == down_pattern:
        trend = "DOWN"
    else:
        result["check_msgs"].append("Pivot pattern not matching up or down 5-wave.")
        return result
    result["trend"] = trend

    # 3) Pivotların (index, price) ayrıştırılması
    p0i,p0p,_ = last5[0]
    p1i,p1p,_ = last5[1]
    p2i,p2p,_ = last5[2]
    p3i,p3p,_ = last5[3]
    p4i,p4p,_ = last5[4]
    result["pivots"] = [(p0i,p0p),(p1i,p1p),(p2i,p2p),(p3i,p3p),(p4i,p4p)]

    def wave_len(a, b):
        return abs(b - a)

    w1 = wave_len(p0p, p1p)
    w2 = wave_len(p1p, p2p)
    w3 = wave_len(p2p, p3p)
    w4 = wave_len(p3p, p4p)

    d1 = p1i - p0i
    d2 = p2i - p1i
    d3 = p3i - p2i
    d4 = p4i - p3i

    # 4) Dalga aralarında min bar mesafesi kontrolü
    if any(d < min_bar_distance for d in [d1,d2,d3,d4]):
        result["check_msgs"].append("Bar distance too small between waves.")
        return result

    # 5) Dalga2, dalga1'in başlangıcını geçmemeli (klasik kural, opsiyonel esneme)
    if not allow_wave2_above_wave1_start:
        if trend == "UP":
            if p2p <= p0p:
                result["check_msgs"].append("Wave2 price retraced below Wave1 start (not typical).")
                return result
        else:
            if p2p >= p0p:
                result["check_msgs"].append("Wave2 price retraced above Wave1 start (not typical).")
                return result

    # 6) Dalga3 min uzunluk
    if w3 < (rule_3rdwave_min_percent * w1):
        result["check_msgs"].append("3rd wave not long enough vs wave1.")
        return result

    # 7) Wave3 en kısa olmamalı
    if rule_3rdwave_not_shortest:
        if (w3 < w1) and (w3 < w4):
            result["check_msgs"].append("3rd wave is the shortest wave (invalid Elliott).")
            return result

    # 8) Dalga4, Dalga1 alanına girmemeli
    if not allow_4th_overlap:
        if trend == "UP":
            if p4p <= p1p:
                result["check_msgs"].append("4th wave overlap in UP trend (invalid).")
                return result
        else:
            if p4p >= p1p:
                result["check_msgs"].append("4th wave overlap in DOWN trend (invalid).")
                return result

    # 9) Fibonacci retracement kontrolleri
    w2r = w2 / (w1 + 1e-9)
    w4r = w4 / (w3 + 1e-9)

    min_w2 = wave2_fib_range[0] - fib_tolerance_range
    max_w2 = wave2_fib_range[1] + fib_tolerance_range
    if not (min_w2 <= w2r <= max_w2):
        result["check_msgs"].append(f"Wave2 retracement ratio {w2r:.2f} not in [{min_w2:.2f}, {max_w2:.2f}].")
        return result

    min_w4 = wave4_fib_range[0] - fib_tolerance_range
    max_w4 = wave4_fib_range[1] + fib_tolerance_range
    if not (min_w4 <= w4r <= max_w4):
        result["check_msgs"].append(f"Wave4 retracement ratio {w4r:.2f} not in [{min_w4:.2f}, {max_w4:.2f}].")
        return result

    # 10) 5. dalga uzatma kontrolü
    if check_extended_5th:
        wave5_ratio = w4 / (w1 + 1e-9)
        if rule_5thwave_ext_range[0] <= wave5_ratio <= rule_5thwave_ext_range[1]:
            result["extended_5th"] = True

    # 11) ABC düzeltme
    if check_abc_correction and len(wave) >= 8:
        maybe_abc = wave[-3:]
        abc_types = [p[2] for p in maybe_abc]
        if trend == "UP":
            if abc_types == [-1, +1, -1]:
                result["abc"] = True
        else:
            if abc_types == [+1, -1, +1]:
                result["abc"] = True

    # 12) Retest kontrolü
    result["wave4_level"] = p4p
    if check_retest:
        retest_info = _check_retest_elliott_wave4(
            df, time_frame,
            wave4_index=p4i,
            wave4_price=p4p,
            tolerance=retest_tolerance,
            trend=trend
        )
        result["retest_info"] = retest_info

    # 13) Tüm kritik kurallar geçti -> found=True
    result["found"] = True

    # 14) Opsiyonel trade seviyeleri (5. dalga henüz oluşmadı varsayarak)
    #     Genellikle dalga4 -> dalga5 başlarken trade planı.
    #     trend=UP ise dalga4 dibi civarında entry, dip biraz altına stop
    #     trend=DOWN ise dalga4 tepesi civarında entry, tepe biraz üstüne stop
    if calc_trade_levels and result["found"]:
        if trend == "UP":
            entry_price = p4p
            stop_loss = p4p * (1 - stop_loss_buffer)  # örn. %1 altına
        else:
            entry_price = p4p
            stop_loss = p4p * (1 + stop_loss_buffer)  # örn. %1 üstüne

        result["entry_price"] = entry_price
        result["stop_loss"] = stop_loss

    return result
