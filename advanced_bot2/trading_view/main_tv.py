
from  trading_view.patterns.all_patterns import detect_all_patterns_v2,run_detection_with_filters ,PivotScanner,indicator_checks
from trading_view.ml_model import PatternEnsembleModel
import pandas as pd
from trading_view.helper import filter_trades_with_indicators,extract_pattern_trade_levels_filtered,measure_pattern_distances,filter_confirmed_within_tolerance

# Log örneği
try:
    from core.logging_setup import log
except ImportError:
    def log(msg, level="info"):
        print(f"[{level.upper()}] {msg}")

from trading_view.utils import run_scanner

##############################################################################
# 1) TIMEFRAME CONFIGS (v2)
##############################################################################
TIMEFRAME_CONFIGS = {

    "1m": {
        "system_params": {
            "pivot_left_bars": 5,
            "pivot_right_bars": 5,
            "volume_filter": True,
            "min_atr_factor": 0.3
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 10,
                "right_bars": 10,
                "min_distance_bars": 10,
                "shoulder_tolerance": 0.03,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 50,
                "atr_filter": 0.2,
                "check_rsi_macd": False,
                "check_retest": False,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 10,
                "right_bars": 10,
                "min_distance_bars": 10,
                "shoulder_tolerance": 0.03,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 50,
                "atr_filter": 0.2,
                "check_rsi_macd": False,
                "check_retest": False,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.015,
                "min_distance_bars": 20,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.08,
                "wave_min_bars": 20,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.5,
                "rule_5thwave_ext_range": (1.0, 1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.03,
                "strict_lines": False,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.35
            },
            "harmonic": {
                "fib_tolerance": 0.03,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "check_volume": False,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.02,
                "check_breakout": True,
                "check_retest": False,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.02,
                "check_breakout": True,
                "check_retest": False
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 10,   # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 40,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.02,   # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
        "use_ultra": True, "sq9_variant": "sqrt_plus_360", "sq9_steps": 5, "w24_variant": "typeB", "w24_steps": 5            
            }
        }
    },

    "5m": {
        "system_params": {
            "pivot_left_bars": 10,
            "pivot_right_bars": 10,
            "volume_filter": True,
            "min_atr_factor": 0.5
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 15,
                "right_bars": 15,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 15,
                "right_bars": 15,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 25,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.08,
                "wave_min_bars": 25,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.618,
                "rule_5thwave_ext_range": (1.0, 1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.03,
                "strict_lines": False,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.3
            },
            "harmonic": {
                "fib_tolerance": 0.02,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "check_volume": True,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.02,
                "check_breakout": True,
                "check_retest": True,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.02,
                "check_breakout": True,
                "check_retest": True
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 15,  # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 50,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.02,  # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
        "use_ultra": True, "sq9_variant": "sqrt_plus_360", "sq9_steps": 5, "w24_variant": "typeB", "w24_steps": 5            
            }
        }
    },

    "15m": {
        "system_params": {
            "pivot_left_bars": 15,
            "pivot_right_bars": 15,
            "volume_filter": True,
            "min_atr_factor": 0.7
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 20,
                "right_bars": 20,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 20,
                "right_bars": 20,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 30,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.07,
                "wave_min_bars": 30,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.618,
                "rule_5thwave_ext_range": (1.0,1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.025,
                "strict_lines": True,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.25
            },
            "harmonic": {
                "fib_tolerance": 0.02,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 10,
                "right_bars": 10,
                "check_volume": True,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.018,
                "check_breakout": True,
                "check_retest": True,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.018,
                "check_breakout": True,
                "check_retest": True
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 15,  # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 60,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.018, # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
        "use_ultra": True, "sq9_variant": "sqrt_plus_360", "sq9_steps": 5, "w24_variant": "typeB", "w24_steps": 5            
            }
        }
    },

    "30m": {
        "system_params": {
            "pivot_left_bars": 20,
            "pivot_right_bars": 20,
            "volume_filter": True,
            "min_atr_factor": 0.8
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.3,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.3,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 40,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.08,
                "wave_min_bars": 40,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.618,
                "rule_5thwave_ext_range": (1.0,1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.025,
                "strict_lines": False,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.3
            },
            "harmonic": {
                "fib_tolerance": 0.02,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 12,
                "right_bars": 12,
                "check_volume": True,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.02,
                "check_breakout": True,
                "check_retest": False,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.02,
                "check_breakout": True,
                "check_retest": False
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 20,   # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 60,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.02,   # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
        "use_ultra": True, "sq9_variant": "sqrt_plus_360", "sq9_steps": 5, "w24_variant": "typeB", "w24_steps": 5            
            }
        }
    },

    "1h": {
        "system_params": {
            "pivot_left_bars": 30,
            "pivot_right_bars": 30,
            "volume_filter": True,
            "min_atr_factor": 1.0
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.5,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.5,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 40,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.06,
                "wave_min_bars": 40,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.618,
                "rule_5thwave_ext_range": (1.0,1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.02,
                "strict_lines": True,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.25
            },
            "harmonic": {
                "fib_tolerance": 0.02,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 12,
                "right_bars": 12,
                "check_volume": True,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.015,
                "check_breakout": True,
                "check_retest": True,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.015,
                "check_breakout": True,
                "check_retest": True
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 20,  # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 80,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.02,  # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
                "use_ultra": False
            }
        }
    },

    "4h": {
        "system_params": {
            "pivot_left_bars": 40,
            "pivot_right_bars": 40,
            "volume_filter": True,
            "min_atr_factor": 1.2
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 35,
                "right_bars": 35,
                "min_distance_bars": 35,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 0.5,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 35,
                "right_bars": 35,
                "min_distance_bars": 35,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 0.5,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 45,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.05,
                "wave_min_bars": 45,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.618,
                "rule_5thwave_ext_range": (1.0, 1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.02,
                "strict_lines": True,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.25
            },
            "harmonic": {
                "fib_tolerance": 0.018,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 15,
                "right_bars": 15,
                "check_volume": True,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.015,
                "check_breakout": True,
                "check_retest": True,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.015,
                "check_breakout": True,
                "check_retest": True
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 25,  # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 100,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.015,  # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
        "use_ultra": True, "sq9_variant": "sqrt_plus_360", "sq9_steps": 5, "w24_variant": "typeB", "w24_steps": 5            
            }
        }
    },

    "1d": {
        "system_params": {
            "pivot_left_bars": 50,
            "pivot_right_bars": 50,
            "volume_filter": True,
            "min_atr_factor": 1.5
        },
        "pattern_config": {
            "headshoulders": {
                "left_bars": 25,
                "right_bars": 25,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 1.0,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },"triple_top_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},

"triple_bottom_advanced": {
    "tolerance": 0.01,
    "min_distance_bars": 20,
    "volume_check": True,
    "volume_col_factor": 0.8,
    "neckline_break": True,
    "check_retest": True,
    "retest_tolerance": 0.01
},
            "inverse_headshoulders": {
                "left_bars": 25,
                "right_bars": 25,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 1.0,
                "check_rsi_macd": False,
                "check_retest": True,
                "retest_tolerance": 0.01
            },
            "doubletriple": {
                "tolerance": 0.008,
                "min_distance_bars": 50,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
            },
            "elliott": {
                "fib_tolerance": 0.05,
                "wave_min_bars": 50,
                "extended_waves": True,
                "rule_3rdwave_min_percent": 1.618,
                "rule_5thwave_ext_range": (1.0, 1.618),
                "check_alt_scenarios": True,
                "check_abc_correction": True,
                "allow_4th_overlap": False,
                "min_bar_distance": 5,
                "check_fib_retracements": True
            },
            "wolfe": {
                "price_tolerance": 0.02,
                "strict_lines": True,
                "breakout_confirm": True,
                "line_projection_check": True,
                "check_2_4_slope": True,
                "check_1_4_intersection_time": True,
                "check_time_symmetry": True,
                "max_time_ratio": 0.2
            },
            "harmonic": {
                "fib_tolerance": 0.018,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 18,
                "right_bars": 18,
                "check_volume": True,
                "volume_factor": 1.3
            },
            "triangle_wedge": {
                "triangle_tolerance": 0.01,
                "check_breakout": True,
                "check_retest": True,
                "triangle_types": ["ascending","descending","symmetrical"]
            },
            "wedge_params": {
                "wedge_tolerance": 0.01,
                "check_breakout": True,
                "check_retest": True
            },
            "cuphandle": {
                "tolerance": 0.02,
                "volume_drop_check": True
            },
            "flagpennant": {
                "min_flagpole_bars": 30,  # Değiştirildi
                "impulse_pct": 0.05,
                "max_cons_bars": 120,
                "pivot_channel_tolerance": 0.02,
                "pivot_triangle_tolerance": 0.02,
                "require_breakout": True
            },
            "channel": {
                "parallel_thresh": 0.01,  # Değiştirildi
                "min_top_pivots": 3,
                "min_bot_pivots": 3,
                "max_iter": 10
            },
            "gann": {
             
             
        "use_ultra": True, "sq9_variant": "sqrt_plus_360", "sq9_steps": 5, "w24_variant": "typeB", "w24_steps": 5            
            
            
            }
        }
    }
}


##############################################################################
# Helper: get_col_name (time_frame'li kolon ismi)
##############################################################################
def get_col_name(base_col: str, time_frame: str) -> str:
    """ 'High' + '5m' -> 'High_5m' """
    return f"{base_col}_{time_frame}"

def check_breakout_volume(df: pd.DataFrame, time_frame: str="1m",
                          atr_window: int=14, vol_window: int=20) -> tuple:
    """
    Basit breakout + hacim spike kontrolü, time_frame'e göre kolonları okur:
      breakout_up, breakout_down, volume_spike döner.
    """
    high_col   = get_col_name("High", time_frame)
    low_col    = get_col_name("Low",  time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume", time_frame)
    atr_col    = get_col_name("ATR", time_frame)

    if atr_col not in df.columns:
        df[f"H-L_{time_frame}"]  = df[high_col] - df[low_col]
        df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
        df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()
        df[f"TR_{time_frame}"]   = df[[f"H-L_{time_frame}",
                                       f"H-PC_{time_frame}",
                                       f"L-PC_{time_frame}"]].max(axis=1)
        df[atr_col] = df[f"TR_{time_frame}"].rolling(atr_window).mean()

    if len(df)<2:
        return (False,False,False)

    last_close= df[close_col].iloc[-1]
    prev_close= df[close_col].iloc[-2]
    last_atr  = df[atr_col].iloc[-1]
    if pd.isna(last_atr):
        last_atr= 0

    breakout_up=   (last_close- prev_close)> last_atr
    breakout_down= (prev_close- last_close)> last_atr

    volume_spike= False
    if volume_col in df.columns and len(df)> vol_window:
        v_now= df[volume_col].iloc[-1]
        v_mean= df[volume_col].rolling(vol_window).mean().iloc[-2]
        volume_spike= (v_now> 1.5* v_mean)

    return (breakout_up, breakout_down, volume_spike)



##############################################################################
# 3) ZigZag / Wave Builder
##############################################################################
def build_zigzag_wave(pivots):
    """
    Pivots listesini ( (idx,price,type) ) zigzag dalga halinde birleştirir.
    """
    if not pivots:
        return []
    sorted_p = sorted(pivots, key=lambda x: x[0])
    wave = [sorted_p[0]]
    for i in range(1, len(sorted_p)):
        curr = sorted_p[i]
        prev = wave[-1]
        if curr[2] == prev[2]:
            # Aynı tip => Daha ekstrem pivotu koru
            if curr[2] == +1:
                if curr[1] > prev[1]:
                    wave[-1] = curr
            else:
                if curr[1] < prev[1]:
                    wave[-1] = curr
        else:
            wave.append(curr)
    return wave







##############################################################################
# 6) SIGNAL ENGINE (generate_signals) => v2 (EK PATTERNLER DAHİL)
##############################################################################
async def generate_signals(
    df: pd.DataFrame,
    symbol:str = "BTCUSDT",
    time_frame: str = "1m",
    ml_model: PatternEnsembleModel = None,
    max_bars_ago: int = 300,
    retest_tolerance=0.05,
    require_confirmed: bool = False,
    check_rsi_macd: bool = False
) -> dict:
    """
    Gelişmiş Pattern + ML + Breakout & Hacim + opsiyonel RSI/MACD analizi => final sinyal üretimi (v2).

    EK: Cup&Handle, Flag/Pennant, Channel, Gann patternleri de skorlamaya eklendi.

    Dönüş formatı:
      {
        "signal": "BUY"/"SELL"/"HOLD",
        "score": <int>,
        "reason": "<metin>",
        "patterns": {...},             
        "ml_label": 0/1/2,
        "breakout_up": bool,
        "breakout_down": bool,
        "volume_spike": bool,
        "time_frame": "1m",
        "pattern_trade_levels": {...}
      }
    """
    if time_frame not in TIMEFRAME_CONFIGS:
        raise ValueError(f"Invalid time_frame='{time_frame}'")

    tf_settings = TIMEFRAME_CONFIGS[time_frame]
    system_params = tf_settings["system_params"]
    pattern_conf  = tf_settings["pattern_config"]
    atr_factor,volume_factor,left_bars,right_bars=run_scanner(df, symbol, time_frame, system_params)
    # 1) Pivot & ZigZag
    # scanner = PivotScanner(
    #     df=df,
    #     time_frame=time_frame,
    #     left_bars= system_params["pivot_left_bars"],
    #     right_bars=system_params["pivot_right_bars"],
    #     volume_factor= 1.2 if system_params["volume_filter"] else 0.0,
    #     atr_factor= system_params["min_atr_factor"],
    # )
    scanner = PivotScanner(
        df=df,
        time_frame=time_frame,
        left_bars= left_bars,
        right_bars= right_bars,
        volume_factor= volume_factor,
        atr_factor= atr_factor
    )
    
    #optimize_parameters(df=df,time_frame: str="1m", param_grid: dict = None)
    pivots = scanner.find_pivots()
    wave   = build_zigzag_wave(pivots)
    
    filename = f"{symbol.lower()}.json"

    # 2) Tüm pattern tespiti (v2) => Cup&Handle, Flag, Channel, Gann vs. dahil
    patterns = await detect_all_patterns_v2(
        
        df=df,symbol=symbol,
        time_frame=time_frame,
        filename=filename,
            scoring_fn = None

      
    )
    
   # print(patterns)

    # 3) ML tahmini (opsiyonel)
    ml_label= None
    if ml_model is not None and wave:
        feats = ml_model.extract_features(wave)
        ml_label = ml_model.predict(feats)[0]  # 0=HOLD,1=BUY,2=SELL

    # 4) Basit Breakout + Hacim kontrolü
    b_up, b_down, v_spike = check_breakout_volume(df, time_frame=time_frame)

    # 5) RSI/MACD => opsiyonel bar => son bar
    rsi_macd_signal = True
    if check_rsi_macd and len(df)>0:
        check_idx = len(df)-1
        ind_res = indicator_checks(df, check_idx, time_frame=time_frame)
        rsi_macd_signal = ind_res["signal"]
    # 4) RSI/MACD filtrelemesi
 

   
    #retest_tolerance=0.05
    current_price = df[get_col_name("Close", time_frame)].iloc[-1] if len(df)>0 else 0.0
 
    # 3) Distance / Retest ölçümü
    distances = measure_pattern_distances(patterns, current_price, retest_tolerance)

    # 4) Confirmed + within_tolerance filtre haritası
    confirmed_map = filter_confirmed_within_tolerance(distances)
    # Örnek dönüş: { "inverse_headshoulders": [0,1], "double_bottom": [0], ... }

    # 5) Yalnızca bu filtreye uyan pattern’lar için trade seviyeleri
    trades,pattern_score = extract_pattern_trade_levels_filtered(
        patterns, 
        confirmed_map=confirmed_map,
        df=df,
        time_frame=time_frame,
        atr_period=14,
        atr_sl_multiplier=1.0,
        atr_tp_multiplier=1.0,
        default_break_offset=0.001
    )
    #print(trades)
   # 4) RSI/MACD filtrelemesi
    # filtered_trades = filter_trades_with_indicators(
    #     trades,
    #     df,
    #     time_frame,
    #     rsi_col=get_col_name("RSI", time_frame),
    #     macd_col=get_col_name("MACD", time_frame),
    #     macd_signal_col=get_col_name("MACDSig", time_frame),
    # )
   
    return {
         "score": pattern_score,   
         "time_frame": time_frame,
         "pattern_trade_levels": trades
    }
   