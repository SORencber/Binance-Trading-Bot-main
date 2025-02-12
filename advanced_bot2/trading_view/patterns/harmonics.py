import pandas as pd
import numpy as np

################################################
# 1) ANA FONKSİYON (ÇOKLU TIMEFRAME & PATTERN SET)
################################################

def detect_harmonic_patterns_multiple_sets(df: pd.DataFrame, pivots, config: dict, time_frame: str):
    """
    Verilen 'time_frame' için config'te tanımlı çoklu pattern setlerini tarar.
    Her pattern seti için detect_harmonic_pattern_advanced fonksiyonunu çalıştırır.
    
    Dönüş:
      {
        "time_frame": "15m",  # örnek
        "results": [
          {
            "pattern_set_idx": 0,  # hangi pattern set
            "found": bool,
            "pattern_name": str veya None,
            "xabc": [...],
            "msgs": [...],
            "retest_info": {... veya None}
          },
          {
            "pattern_set_idx": 1,
            "found": bool,
            "pattern_name": ...
          },
          ...
        ]
      }
    """

    # 1) Config'ten gerekli parametreleri çek
    cfg = config.get(time_frame, {})
    fib_tolerance_list  = cfg.get("fib_tolerance", [0.02])    # default: [0.02]
    patterns_list       = cfg.get("patterns", [["gartley","bat","crab","butterfly","shark","cipher"]])
    check_volume_list   = cfg.get("check_volume", [False])
    volume_factor_list  = cfg.get("volume_factor", [1.3])
    check_retest_list   = cfg.get("check_retest", [False])
    retest_tolerance_list = cfg.get("retest_tolerance", [0.01])

    # Not: fib_tolerance_list, patterns_list vs. hepsi birer liste; 
    # birden fazla kombinasyon loop içinde denenebilir.

    # 2) Sonuçları toplamak için yapı
    final_output = {
        "time_frame": time_frame,
        "results": []
    }

    # 3) Tüm kombinasyonları loop'lamak isterseniz:
    # (Ancak config'te bazen fib_tolerance vs. birden fazla olabilir)
    # Biz burada "pattern set" sayısını esas alıp,
    # pattern set kadar tur döneceğiz. (İhtiyaca göre farklılaşabilir.)
    max_sets_count = len(patterns_list)
    
    # Toleranslar, volume_factors vs. eğer tek elemanlıysa hep onu kullan,
    # eğer birden çok varsa her pattern set loopunda indexe göre sec vb.
    # Aşağıda basit bir "index mod" yaklaşımı kullandık.

    for idx in range(max_sets_count):
        # indexe göre parametreleri alalım (out of range olmaması için mod alabiliriz)
        fib_tol = fib_tolerance_list[idx % len(fib_tolerance_list)]
        p_set   = patterns_list[idx % len(patterns_list)]
        chk_vol = check_volume_list[idx % len(check_volume_list)]
        vol_fac = volume_factor_list[idx % len(volume_factor_list)]
        chk_ret = check_retest_list[idx % len(check_retest_list)]
        ret_tol = retest_tolerance_list[idx % len(retest_tolerance_list)]

        # 4) Gelişmiş harmonic fonksiyonunu çağır
        detection_result = detect_harmonic_pattern_advanced(
            df           = df,
            pivots       = pivots,
            time_frame   = time_frame,
            fib_tolerance= fib_tol,
            patterns     = p_set,
            check_volume = chk_vol,
            volume_factor= vol_fac,
            check_retest = chk_ret,
            retest_tolerance = ret_tol,
            # check_rsi örnek: True/False sabit verebilirsiniz
            check_rsi    = False
        )

        # 5) Ek bilgileri detection_result içine ekleyebilir veya
        # separate key'lerle final_output'a atayabiliriz.
        # Burada detection_result'a pattern_set_idx ekleyelim
        detection_result["pattern_set_idx"] = idx
        final_output["results"].append(detection_result)

    return final_output


########################################
# 2) TEK BİR PATTERN SETİ TESPİT FONKSİYONU
########################################

def detect_harmonic_pattern_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    fib_tolerance: float = 0.02,
    patterns: list = None,
    check_volume: bool = False,
    volume_factor: float = 1.3,
    check_retest: bool = False,
    retest_tolerance: float = 0.01,
    check_rsi: bool = False,
    rsi_overbought: float = 70.0,
    rsi_oversold: float = 30.0
) -> dict:
    """
    (Önceden paylaşılan) Gelişmiş Harmonic Pattern tespit fonksiyonu.
    - 5 pivot: X,A,B,C,D
    - Belirli fib oranları (AB/XA, BC/AB, CD/BC) ile pattern bulma
    - Opsiyonel hacim, RSI, retest kontrolü vb.
    
    Dönüş:
        {
          "pattern": "harmonic",
          "found": bool,
          "pattern_name": str veya None,
          "xabc": [X, A, B, C, D],
          "msgs": list,
          "retest_info": dict veya None
        }
    """
    if patterns is None:
        patterns = ["gartley","bat","crab","butterfly","shark","cipher"]

    result = {
        "pattern": "harmonic",
        "found": False,
        "pattern_name": None,
        "xabc": [],
        "msgs": [],
        "retest_info": None
    }

    # 1) Zigzag dalga oluştur
    wave = build_zigzag_wave(pivots)
    if len(wave) < 5:
        result["msgs"].append("Not enough pivots for harmonic (need at least 5).")
        return result

    # 2) Son 5 pivot: X,A,B,C,D
    X = wave[-5]
    A = wave[-4]
    B = wave[-3]
    C = wave[-2]
    D = wave[-1]
    idxX, pxX, _ = X
    idxA, pxA, _ = A
    idxB, pxB, _ = B
    idxC, pxC, _ = C
    idxD, pxD, _ = D
    result["xabc"] = [X, A, B, C, D]

    # 3) Yardımcılar
    def length(a, b):
        return abs(b - a)

    def in_range(val, rng, tol):
        mn, mx = rng
        if abs(mn - mx) < 1e-9:
            return abs(val - mn) <= abs(mn) * tol
        else:
            low_ = mn - abs(mn) * tol
            high_ = mx + abs(mx) * tol
            return low_ <= val <= high_

    # 4) Fiyat hareket oranları
    XA = length(pxX, pxA)
    AB = length(pxA, pxB)
    BC = length(pxB, pxC)
    CD = length(pxC, pxD)

    AB_XA = AB / (XA + 1e-9)
    BC_AB = BC / (AB + 1e-9)
    CD_BC = CD / (BC + 1e-9)

    # 5) Pattern fib tanımları
    harmonic_map = {
        "gartley": {
            "AB_XA": (0.618, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.13, 1.618)
        },
        "bat": {
            "AB_XA": (0.382, 0.5),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.618)
        },
        "crab": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (2.24, 3.618)
        },
        "butterfly": {
            "AB_XA": (0.786, 0.786),
            "BC_AB": (0.382, 0.886),
            "CD_BC": (1.618, 2.24)
        },
        "shark": {
            "AB_XA": (0.886, 1.13),
            "BC_AB": (1.13, 1.618),
            "CD_BC": (0.886, 1.13)
        },
        "cipher": {
            "AB_XA": (0.382, 0.618),
            "BC_AB": (1.27, 2.0),
            "CD_BC": (1.13, 1.414)
        }
    }

    # 6) Pattern kontrol
    found_any = False
    matched_pattern = None
    for pat in patterns:
        if pat not in harmonic_map:
            continue
        spec = harmonic_map[pat]
        rngAB_XA = spec["AB_XA"]
        rngBC_AB = spec["BC_AB"]
        rngCD_BC = spec["CD_BC"]

        ok1 = in_range(AB_XA, rngAB_XA, fib_tolerance)
        ok2 = in_range(BC_AB, rngBC_AB, fib_tolerance)
        ok3 = in_range(CD_BC, rngCD_BC, fib_tolerance)

        if ok1 and ok2 and ok3:
            found_any = True
            matched_pattern = pat
            break

    # 7) Bulundu ise detay kontroller
    if found_any:
        result["found"] = True
        result["pattern_name"] = matched_pattern
        # Basit bullish/bearish tahmini
        if pxA < pxX:
            pattern_direction = "bullish"
        else:
            pattern_direction = "bearish"
        result["msgs"].append(f"Pattern direction guess: {pattern_direction}")

        # Hacim kontrolü
        if check_volume:
            vol_col = get_col_name("Volume", time_frame)
            if vol_col in df.columns and idxD < len(df):
                vol_now = df[vol_col].iloc[idxD]
                # Hazırlık (MA)
                prepare_volume_ma(df, time_frame, period=20)
                ma_col = f"Volume_MA_20_{time_frame}"
                if ma_col in df.columns:
                    v_mean = df[ma_col].iloc[idxD]
                    if v_mean > 0:
                        vol_ratio = vol_now / v_mean
                        result["msgs"].append(f"Volume ratio @D: {vol_ratio:.2f}")
                        if vol_ratio > volume_factor:
                            result["msgs"].append("High volume at D pivot (possible strong reaction).")

        # RSI kontrolü (örnek)
        if check_rsi:
            rsi_col = f"RSI_14_{time_frame}"
            if rsi_col in df.columns and idxD < len(df):
                rsi_val = df[rsi_col].iloc[idxD]
                result["msgs"].append(f"RSI @D: {rsi_val:.2f}")
                if rsi_val >= rsi_overbought:
                    result["msgs"].append("RSI is overbought at D pivot!")
                elif rsi_val <= rsi_oversold:
                    result["msgs"].append("RSI is oversold at D pivot!")

        # Retest kontrolü
        if check_retest:
            close_col = get_col_name("Close", time_frame)
            if close_col in df.columns:
                retest_done = False
                retest_bar = None
                for i in range(idxD + 1, len(df)):
                    c = df[close_col].iloc[i]
                    dist_ratio = abs(c - pxD) / (abs(pxD) + 1e-9)
                    if dist_ratio <= retest_tolerance:
                        retest_done = True
                        retest_bar = i
                        break
                if retest_done:
                    result["retest_info"] = {
                        "retest_done": True,
                        "retest_bar": retest_bar,
                        "retest_price": df[close_col].iloc[retest_bar]
                    }
                else:
                    result["retest_info"] = {"retest_done": False}

    else:
        result["msgs"].append("No harmonic pattern matched.")

    return result


################################################
# 3) YARDIMCI FONKSİYONLAR
################################################

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def prepare_volume_ma(df: pd.DataFrame, time_frame: str = "1m", period: int = 20):
    """
    Volume hareketli ortalamasını (örn. 20 periyot) df'e ekle.
    Kolon adı: "Volume_MA_{period}_{time_frame}"
    """
    vol_col = get_col_name("Volume", time_frame)
    ma_col = f"Volume_MA_{period}_{time_frame}"
    if (vol_col in df.columns) and (ma_col not in df.columns):
        df[ma_col] = df[vol_col].rolling(period).mean()

def build_zigzag_wave(pivots):
    """
    Zigzag pivot list: (index, price, direction +1/-1)
    Sort edip dalga oluşturma.
    """
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda x: x[0])
    wave = [sorted_p[0]]
    for i in range(1, len(sorted_p)):
        curr = sorted_p[i]
        prev = wave[-1]
        if curr[2] == prev[2]:
            # Aynı tip pivot (ikisi de tepe veya dip)
            # Daha ekstrem olanı güncelle
            if curr[2] == +1:
                if curr[1] > prev[1]:
                    wave[-1] = curr
            else:
                if curr[1] < prev[1]:
                    wave[-1] = curr
        else:
            wave.append(curr)
    return wave
