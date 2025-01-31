# config.py

PATTERN_PARAMS = {
    "elliott": {
        # Elliott alt dalgalar için ek parametreler:
        "fib_tolerance": 0.1,
        "check_alt_scenarios": True,
        "wave_min_bars": 5,
        "extended_waves": True,   # Elliott 1-5, + A-B-C
    },
    "wolfe": {
        "price_tolerance": 0.03,
        "strict_lines": False,
        "breakout_confirm": True,
        "line_projection_check": True  # 1-4 line, 2-4 line angle
    },
    "harmonic": {
        "patterns": ["gartley","bat","crab","butterfly","cipher","shark"],
        "fib_tolerance": 0.03,
        # oransal aralıklar, mesela "gartley":(0.618,0.786), vs. => alt parametre
    },
    "headshoulders": {
        "shoulder_tolerance": 0.02,
        "min_distance_bars": 3,
        "check_volume_decline": True  # H&S'te volume vs
    },
    "doubletriple": {
        "tolerance": 0.01,
        "min_distance_bars": 2,
        "triple_variation": True
    },
    "triangle_wedge": {
        "triangle_tolerance": 0.02,
        "wedge_tolerance": 0.02,
        "check_breakout": True,
        "check_retest": False
    },
    "ml": {
        "model_path": "models/pattern_ensemble.pkl"
    }
}

SYSTEM_PARAMS = {
    "pivot_left_bars": 2,
    "pivot_right_bars":2,
    "volume_filter": True,
    "min_atr_factor":0.5   # wave min length
}
