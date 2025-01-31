# pattern_detector.py

from patterns.elliott import detect_elliott_5wave
from patterns.wolfe import detect_wolfe_wave
from patterns.harmonics import detect_harmonic_pattern
from patterns.headshoulders import detect_head_and_shoulders, detect_inverse_head_and_shoulders
from patterns.double_triple import detect_double_top, detect_double_bottom
from patterns.triangle_wedge import detect_triangle, detect_wedge

def detect_all_patterns(pivots, wave, df=None, config=None):
    """
    Tüm pattern fonksiyonlarını çağırır, dict döndürür
    """
    if config is None:
        config = {}

    # HeadShoulders => liste
    hs_list = detect_head_and_shoulders(pivots, **config.get("headshoulders",{}))
    inv_hs_list = detect_inverse_head_and_shoulders(pivots, **config.get("headshoulders",{}))

    results = {
        "elliott": detect_elliott_5wave(wave, **config.get("elliott",{})),
        "wolfe": detect_wolfe_wave(wave, **config.get("wolfe",{}), df=df),
        "harmonic": detect_harmonic_pattern(wave, **config.get("harmonic",{})),
        "headshoulders": hs_list,
        "inverse_headshoulders": inv_hs_list,
        "double_top": detect_double_top(pivots, **config.get("doubletriple",{})),
        "double_bottom": detect_double_bottom(pivots, **config.get("doubletriple",{})),
        "triangle": detect_triangle(wave, **config.get("triangle_wedge",{}), df=df),
        "wedge": detect_wedge(wave, **config.get("triangle_wedge",{}), df=df)
    }
    return results
