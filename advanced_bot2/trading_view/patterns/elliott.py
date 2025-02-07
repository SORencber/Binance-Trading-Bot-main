# patterns/elliott.py
import pandas as pd
from core.logging_setup import log

############################
# ELLIOTT 5 WAVE
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
    allow_wave2_above_wave1_start: bool = False,  # Klasik kural: wave2, wave1'in başlangıcını geçmemeli
    # Fibonacci tipik aralıklar
    wave2_fib_range: tuple = (0.382, 0.618),  # tipik retracement
    wave4_fib_range: tuple = (0.382, 0.618),
    # Uzatılmış dalga vb. opsiyonel
    check_extended_5th: bool = True,
    rule_5thwave_ext_range: tuple = (1.0, 1.618),
    # ABC isteğe bağlı
    check_abc_correction: bool = True,
    # Bar aralığı
    min_bar_distance: int = 3,
    # Retest kontrolü
    check_retest: bool = False,
    retest_tolerance: float = 0.01
):
    """
    Daha katı Elliott 5-dalga tespit fonksiyonu.
    Tüm temel Elliott kuralları sağlanmazsa found=False döner.
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
        "retest_info": None
    }

    # 1) Zigzag wave oluştur
    wave = build_zigzag_wave(pivots)
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

    # 5) Dalga 2, dalga1'i aşırı aşmamalı (Klasik kural: wave2 asla wave1'in başlangıcını geçmez)
    #    Örnek: Trend UP -> wave2 'nin dibinin (p2p) p0p'dan düşük olmaması gerek vs.
    #    Fakat "allow_wave2_above_wave1_start=True" vererek bu kuralı esnetebilirsiniz.
    if not allow_wave2_above_wave1_start:
        if trend == "UP":
            # Wave2 dipi, wave1 başlangıcı altına inmemeli
            if p2p <= p0p:
                result["check_msgs"].append("Wave2 price retraced below Wave1 start (not typical).")
                return result
        else:
            # Trend DOWN -> wave2 tepesi, wave1 başlangıcı üstüne çıkmamalı
            if p2p >= p0p:
                result["check_msgs"].append("Wave2 price retraced above Wave1 start (not typical).")
                return result

    # 6) Dalga3 minimum uzunluk (çoğu zaman Wave3 en uzundur veya en azından wave1'den kısadır)
    #    En az 1.618 * wave1 kuralı (sık kullanılan)
    if w3 < (rule_3rdwave_min_percent * w1):
        result["check_msgs"].append("3rd wave not long enough vs wave1.")
        return result

    # 7) "Wave3 asla en kısa dalga olamaz" -> wave3'ü w1, w5 ile de kıyaslayabilirsiniz.
    #    Ama wave5 henüz tam net değil diyebilirsiniz. Yine de bir kontrol koyuyoruz:
    if rule_3rdwave_not_shortest:
        # w3, w1 ve w4'le kıyaslanmalı. wave5'i henüz bilmeyebiliriz, çünkü p4 -> p5 pivotu eksik gibi vs.
        # Ama 5 pivot var, demek ki wave5 = p4->??? yok, ya da p4->p5??? 
        # Aslında p4->p5 bizde yoksa tam 5 dalga bitmemiş olabilir. 
        # Burada "wave3 en kısa olmamalı" diyorsanız, w3 < w1 VE w3 < w4 ise fail diyebilirsiniz.
        if (w3 < w1) and (w3 < w4):
            result["check_msgs"].append("3rd wave is the shortest wave (invalid Elliott).")
            return result

    # 8) Dalga4, dalga1 alanına girmemeli (çakışmamalı) - Klasik kural
    if not allow_4th_overlap:
        if trend == "UP":
            # Wave4 (p4) dalgasının dipi, wave1 (p1) tepesinin üstünde (veya aynı) kalmalı
            # Overlap yoksa p4p > p1p olmalı (UP trendinde 1.dalga tepesi p1p).
            if p4p <= p1p:
                result["check_msgs"].append("4th wave overlap in UP trend (invalid).")
                return result
        else:
            # DOWN trend: wave4 tepesi, wave1 dibinden aşağıda kalmalı
            if p4p >= p1p:
                result["check_msgs"].append("4th wave overlap in DOWN trend (invalid).")
                return result

    # 9) Fibonacci retracement kontrolü (wave2 = wave1'in %?? kadarı)
    #    wave2_fib_range = (0.382, 0.618) gibi bir aralık
    #    wave4_fib_range = (0.382, 0.618)
    w2r = w2 / (w1 + 1e-9)
    w4r = w4 / (w3 + 1e-9)

    # wave2 tipik aralık
    min_w2 = wave2_fib_range[0] - fib_tolerance
    max_w2 = wave2_fib_range[1] + fib_tolerance
    if not(min_w2 <= w2r <= max_w2):
        result["check_msgs"].append(f"Wave2 retracement ratio {w2r:.2f} not in [{min_w2:.2f}, {max_w2:.2f}].")
        return result

    # wave4 tipik aralık
    min_w4 = wave4_fib_range[0] - fib_tolerance
    max_w4 = wave4_fib_range[1] + fib_tolerance
    if not(min_w4 <= w4r <= max_w4):
        result["check_msgs"].append(f"Wave4 retracement ratio {w4r:.2f} not in [{min_w4:.2f}, {max_w4:.2f}].")
        return result

    # 10) 5. dalga uzatması? (opsiyonel)
    #     wave5 demek aslında p4->p5 pivotu gerekir. Bu kodda son pivot p4 diye saydık.
    #     Oysa tam 5 dalga = p0->p1 (wave1), p1->p2 (wave2), p2->p3 (wave3), p3->p4 (wave4), p4->p5 (wave5).
    #     Sizin paylaştığınız pivot sıralamasına göre p4 son dalga sonu diye kabul ediyorsanız,
    #     wave5 = w4 diyorsunuz (p3p -> p4p). Bu biraz kavramsal çelişki olabilir.
    #     Yine de "extended wave" mantığı ekleyelim:
    if check_extended_5th:
        # wave5 = abs(p4 - p2)? Klasik formül vs. 
        # Ama elinizde wave5 pivotu yoksa tahmini olur. 
        # Basit şekilde w5_ratio = w4 / w1
        wave5_ratio = w4 / (w1 + 1e-9)
        if rule_5thwave_ext_range[0] <= wave5_ratio <= rule_5thwave_ext_range[1]:
            result["extended_5th"] = True

    # 11) ABC düzeltme kontrolü (isteğe bağlı).
    #     ABC'yi tespit için en az 3 pivot daha gerekebilir (toplam 8 pivot).
    #     Elinizde wave[-3:] vs. diyerek son 3 pivotu ABC diye bakabilirsiniz.
    if check_abc_correction and len(wave) >= 8:
        maybe_abc = wave[-3:]
        abc_types = [p[2] for p in maybe_abc]
        # Trend UP ise ABC: [-1, +1, -1], Trend DOWN ise [+1, -1, +1]
        if trend == "UP":
            if abc_types == [-1, +1, -1]:
                result["abc"] = True
        else:
            if abc_types == [+1, -1, +1]:
                result["abc"] = True

    # Tüm kritik kurallar geçildi, formasyon geçerli
    result["found"] = True
    result["wave4_level"] = p4p

    # 12) Retest kontrolü
    if check_retest:
        retest_info = _check_retest_elliott_wave4(
            df, time_frame,
            wave4_index=p4i,
            wave4_price=p4p,
            tolerance=retest_tolerance,
            trend=trend
        )
        result["retest_info"] = retest_info

    return result


def _check_retest_elliott_wave4(
    df: pd.DataFrame,
    time_frame: str,
    wave4_index: int,
    wave4_price: float,
    tolerance: float=0.01,
    trend: str="UP"
):
    """
    Dalga4 seviyesine daha sonraki barlarda bir 'retest' var mı?
    wave4_index sonrasındaki barların Close'u,
    wave4_price'a ne kadar yaklaşmış, vs. arıyoruz.
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
# ELLIOTT 5 WAVE
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
    allow_wave2_above_wave1_start: bool = False,  # Klasik kural: wave2, wave1'in başlangıcını geçmemeli
    # Fibonacci tipik aralıklar
    wave2_fib_range: tuple = (0.382, 0.618),  # tipik retracement
    wave4_fib_range: tuple = (0.382, 0.618),
    # Uzatılmış dalga vb. opsiyonel
    check_extended_5th: bool = True,
    rule_5thwave_ext_range: tuple = (1.0, 1.618),
    # ABC isteğe bağlı
    check_abc_correction: bool = True,
    # Bar aralığı
    min_bar_distance: int = 3,
    # Retest kontrolü
    check_retest: bool = False,
    retest_tolerance: float = 0.01
):
    """
    Daha katı Elliott 5-dalga tespit fonksiyonu.
    Tüm temel Elliott kuralları sağlanmazsa found=False döner.
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
        "retest_info": None
    }

    # 1) Zigzag wave oluştur
    wave = build_zigzag_wave(pivots)
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

    # 5) Dalga 2, dalga1'i aşırı aşmamalı (Klasik kural: wave2 asla wave1'in başlangıcını geçmez)
    #    Örnek: Trend UP -> wave2 'nin dibinin (p2p) p0p'dan düşük olmaması gerek vs.
    #    Fakat "allow_wave2_above_wave1_start=True" vererek bu kuralı esnetebilirsiniz.
    if not allow_wave2_above_wave1_start:
        if trend == "UP":
            # Wave2 dipi, wave1 başlangıcı altına inmemeli
            if p2p <= p0p:
                result["check_msgs"].append("Wave2 price retraced below Wave1 start (not typical).")
                return result
        else:
            # Trend DOWN -> wave2 tepesi, wave1 başlangıcı üstüne çıkmamalı
            if p2p >= p0p:
                result["check_msgs"].append("Wave2 price retraced above Wave1 start (not typical).")
                return result

    # 6) Dalga3 minimum uzunluk (çoğu zaman Wave3 en uzundur veya en azından wave1'den kısadır)
    #    En az 1.618 * wave1 kuralı (sık kullanılan)
    if w3 < (rule_3rdwave_min_percent * w1):
        result["check_msgs"].append("3rd wave not long enough vs wave1.")
        return result

    # 7) "Wave3 asla en kısa dalga olamaz" -> wave3'ü w1, w5 ile de kıyaslayabilirsiniz.
    #    Ama wave5 henüz tam net değil diyebilirsiniz. Yine de bir kontrol koyuyoruz:
    if rule_3rdwave_not_shortest:
        # w3, w1 ve w4'le kıyaslanmalı. wave5'i henüz bilmeyebiliriz, çünkü p4 -> p5 pivotu eksik gibi vs.
        # Ama 5 pivot var, demek ki wave5 = p4->??? yok, ya da p4->p5??? 
        # Aslında p4->p5 bizde yoksa tam 5 dalga bitmemiş olabilir. 
        # Burada "wave3 en kısa olmamalı" diyorsanız, w3 < w1 VE w3 < w4 ise fail diyebilirsiniz.
        if (w3 < w1) and (w3 < w4):
            result["check_msgs"].append("3rd wave is the shortest wave (invalid Elliott).")
            return result

    # 8) Dalga4, dalga1 alanına girmemeli (çakışmamalı) - Klasik kural
    if not allow_4th_overlap:
        if trend == "UP":
            # Wave4 (p4) dalgasının dipi, wave1 (p1) tepesinin üstünde (veya aynı) kalmalı
            # Overlap yoksa p4p > p1p olmalı (UP trendinde 1.dalga tepesi p1p).
            if p4p <= p1p:
                result["check_msgs"].append("4th wave overlap in UP trend (invalid).")
                return result
        else:
            # DOWN trend: wave4 tepesi, wave1 dibinden aşağıda kalmalı
            if p4p >= p1p:
                result["check_msgs"].append("4th wave overlap in DOWN trend (invalid).")
                return result

    # 9) Fibonacci retracement kontrolü (wave2 = wave1'in %?? kadarı)
    #    wave2_fib_range = (0.382, 0.618) gibi bir aralık
    #    wave4_fib_range = (0.382, 0.618)
    w2r = w2 / (w1 + 1e-9)
    w4r = w4 / (w3 + 1e-9)

    # wave2 tipik aralık
    min_w2 = wave2_fib_range[0] - fib_tolerance
    max_w2 = wave2_fib_range[1] + fib_tolerance
    if not(min_w2 <= w2r <= max_w2):
        result["check_msgs"].append(f"Wave2 retracement ratio {w2r:.2f} not in [{min_w2:.2f}, {max_w2:.2f}].")
        return result

    # wave4 tipik aralık
    min_w4 = wave4_fib_range[0] - fib_tolerance
    max_w4 = wave4_fib_range[1] + fib_tolerance
    if not(min_w4 <= w4r <= max_w4):
        result["check_msgs"].append(f"Wave4 retracement ratio {w4r:.2f} not in [{min_w4:.2f}, {max_w4:.2f}].")
        return result

    # 10) 5. dalga uzatması? (opsiyonel)
    #     wave5 demek aslında p4->p5 pivotu gerekir. Bu kodda son pivot p4 diye saydık.
    #     Oysa tam 5 dalga = p0->p1 (wave1), p1->p2 (wave2), p2->p3 (wave3), p3->p4 (wave4), p4->p5 (wave5).
    #     Sizin paylaştığınız pivot sıralamasına göre p4 son dalga sonu diye kabul ediyorsanız,
    #     wave5 = w4 diyorsunuz (p3p -> p4p). Bu biraz kavramsal çelişki olabilir.
    #     Yine de "extended wave" mantığı ekleyelim:
    if check_extended_5th:
        # wave5 = abs(p4 - p2)? Klasik formül vs. 
        # Ama elinizde wave5 pivotu yoksa tahmini olur. 
        # Basit şekilde w5_ratio = w4 / w1
        wave5_ratio = w4 / (w1 + 1e-9)
        if rule_5thwave_ext_range[0] <= wave5_ratio <= rule_5thwave_ext_range[1]:
            result["extended_5th"] = True

    # 11) ABC düzeltme kontrolü (isteğe bağlı).
    #     ABC'yi tespit için en az 3 pivot daha gerekebilir (toplam 8 pivot).
    #     Elinizde wave[-3:] vs. diyerek son 3 pivotu ABC diye bakabilirsiniz.
    if check_abc_correction and len(wave) >= 8:
        maybe_abc = wave[-3:]
        abc_types = [p[2] for p in maybe_abc]
        # Trend UP ise ABC: [-1, +1, -1], Trend DOWN ise [+1, -1, +1]
        if trend == "UP":
            if abc_types == [-1, +1, -1]:
                result["abc"] = True
        else:
            if abc_types == [+1, -1, +1]:
                result["abc"] = True

    # Tüm kritik kurallar geçildi, formasyon geçerli
    result["found"] = True
    result["wave4_level"] = p4p

    # 12) Retest kontrolü
    if check_retest:
        retest_info = _check_retest_elliott_wave4(
            df, time_frame,
            wave4_index=p4i,
            wave4_price=p4p,
            tolerance=retest_tolerance,
            trend=trend
        )
        result["retest_info"] = retest_info

    return result


def _check_retest_elliott_wave4(
    df: pd.DataFrame,
    time_frame: str,
    wave4_index: int,
    wave4_price: float,
    tolerance: float=0.01,
    trend: str="UP"
):
    """
    Dalga4 seviyesine daha sonraki barlarda bir 'retest' var mı?
    wave4_index sonrasındaki barların Close'u,
    wave4_price'a ne kadar yaklaşmış, vs. arıyoruz.
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
def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

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

