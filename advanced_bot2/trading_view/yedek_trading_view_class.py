
import os
import math
import sqlite3
from typing import Optional, Callable, List, Dict

import pandas as pd
import numpy as np
from  trading_view.patterns.all_patterns import detect_all_patterns_v2 ,PivotScanner
# sklearn / joblib sadece örnek, her zaman gerekmeyebilir
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Log örneği
try:
    from core.logging_setup import log
except ImportError:
    def log(msg, level="info"):
        print(f"[{level.upper()}] {msg}")


##############################################################################
# 0.1) ENV / CONFIG
##############################################################################
# Bazı parametreleri environment variable üzerinden çekme örneği:
DB_PATH = os.getenv("DB_PATH", "trades.db")


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
                "use_ultra": False
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
                "use_ultra": False
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
                "use_ultra": False
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
                "use_ultra": False
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
                "use_ultra": False
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
                "use_ultra": False
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
# 4.1) ML MODEL
##############################################################################
class PatternEnsembleModel:
    """
    Örnek ML modeli (RandomForest) pipeline (v2).
    """
    def __init__(self, model_path:str=None):
        self.model_path = model_path
        self.pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier())
        ])
        self.is_fitted = False

    def fit(self, X, y):
        self.pipe.fit(X, y)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        return self.pipe.predict(X)

    def extract_features(self, wave):
        """
        Wave yapısından basit feature çıkarma. 
        (Pivot sayısı, son pivot tip, son pivot fiyatı, amplitude, vs.)
        """
        n= len(wave)
        if n<2:
            return np.zeros((1,5))
        last= wave[-1]
        second= wave[-2]
        maxi= max([w[1] for w in wave])
        mini= min([w[1] for w in wave])
        amp= maxi- mini
        arr= [n, last[2], last[1], second[2], amp]
        return np.array([arr])

    def save(self):
        if self.model_path:
            joblib.dump(self.pipe, self.model_path)
            log(f"Model saved to {self.model_path}","info")

    def load(self):
        if self.model_path and os.path.exists(self.model_path):
            self.pipe= joblib.load(self.model_path)
            self.is_fitted= True
            log(f"Model loaded from {self.model_path}","info")


##############################################################################
# 5.2) Pattern trade levels (örnek)
##############################################################################
def extract_pattern_trade_levels(patterns_dict, current_price):
    """
    (v2) Her pattern için (entry, stop, target, direction) gibi detayları 
    toplu halde döndüren örnek fonksiyon.

    Dönüş örneği:
    {
      "inverse_headshoulders": [ { ... }, ... ],
      "double_bottom": [...],
      "double_top": [...],
      ...
    }
    """
    results = {
        "inverse_headshoulders": [],
        "double_bottom": [],
        "double_top": [],
        "elliott": [],
        "wolfe": [],
        "triangle": [],
        "wedge": [],
        "harmonic": [],
        "cup_handle": [],
        "flag_pennant": [],
        "channel": []
    }

    # 1) Inverse HS => Long
    inv_list = patterns_dict.get("inverse_headshoulders", [])
    if isinstance(inv_list, dict):
        inv_list = [inv_list]
    for inv in inv_list:
        if inv.get("confirmed"):
            head_price = inv["H"][1]
            neckline_avg = None
            if inv.get("neckline"):
                (nx1, px1), (nx2, px2) = inv["neckline"]
                neckline_avg = (px1 + px2)/2
            entry_price  = neckline_avg if neckline_avg else current_price
            stop_price   = head_price*0.98
            if neckline_avg:
                mm = neckline_avg - head_price
                target_price = neckline_avg + mm
            else:
                target_price = current_price*1.1
            results["inverse_headshoulders"].append({
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "direction": "LONG",
                "pattern_raw": inv
            })

    # 2) Double Bottom => Long
    db_list = patterns_dict.get("double_bottom", [])
    for db in db_list:
        if db.get("confirmed"):
            dip_price = min(db["bottoms"][0][1], db["bottoms"][-1][1])
            stop_price = dip_price*0.97
            neck = db.get("neckline")
            if neck:
                neck_price = neck[1]
                mm = neck_price - dip_price
                target_price = neck_price + mm
                entry_price  = neck_price
            else:
                target_price = current_price*1.1
                entry_price  = current_price
            results["double_bottom"].append({
                "entry_price":  entry_price,
                "stop_price":   stop_price,
                "target_price": target_price,
                "direction": "LONG",
                "pattern_raw": db
            })

    # 3) Double Top => Short
    dt_list = patterns_dict.get("double_top", [])
    for dt in dt_list:
        if dt.get("confirmed"):
            peak_price = max(dt["tops"][0][1], dt["tops"][-1][1])
            stop_price = peak_price*1.03
            neck = dt.get("neckline")
            if neck:
                neck_price = neck[1]
                mm = peak_price - neck_price
                target_price = neck_price - mm
                entry_price  = neck_price
            else:
                target_price = current_price*0.9
                entry_price  = current_price
            results["double_top"].append({
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "direction": "SHORT",
                "pattern_raw": dt
            })

    # 4) Elliott
    ell = patterns_dict.get("elliott", {})
    if ell.get("found"):
        wave_pivots = ell.get("pivots", [])
        trend = ell.get("trend")
        if len(wave_pivots)==5:
            p4_price = wave_pivots[3][1]
            p5_price = wave_pivots[4][1]
            if trend=="UP":
                entry_price  = current_price
                stop_price   = p4_price*0.98
                target_price = p5_price
                direction    = "LONG"
            else:
                entry_price  = current_price
                stop_price   = p4_price*1.02
                target_price = p5_price
                direction    = "SHORT"
            results["elliott"].append({
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "direction": direction,
                "pattern_raw": ell
            })

    # 5) Wolfe
    wol = patterns_dict.get("wolfe", {})
    if wol.get("found") and wol.get("breakout"):
        w5_data = wol.get("w5", None)
        if w5_data:
            w5_price = w5_data[1]
        else:
            w5_price = current_price
        direction = "LONG"  # simplification
        stop_price = w5_price*0.98
        target_price = current_price*1.1
        results["wolfe"].append({
            "entry_price": current_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "direction": direction,
            "pattern_raw": wol
        })

    # 6) Triangle
    tri = patterns_dict.get("triangle", {})
    if tri.get("found"):
        ttype = tri.get("triangle_type","")
        if ttype=="ascending":
            results["triangle"].append({
                "entry_price": current_price,
                "stop_price": current_price*0.95,
                "target_price": current_price*1.15,
                "direction": "LONG",
                "pattern_raw": tri
            })
        elif ttype=="descending":
            results["triangle"].append({
                "entry_price": current_price,
                "stop_price": current_price*1.05,
                "target_price": current_price*0.85,
                "direction": "SHORT",
                "pattern_raw": tri
            })
        # symmetrical => no trade or custom logic

    # 7) Wedge
    wedge = patterns_dict.get("wedge", {})
    if wedge.get("found"):
        wtype = wedge.get("wedge_type","")
        if wtype=="rising":
            results["wedge"].append({
                "entry_price": current_price,
                "stop_price": current_price*1.05,
                "target_price": current_price*0.85,
                "direction": "SHORT",
                "pattern_raw": wedge
            })
        elif wtype=="falling":
            results["wedge"].append({
                "entry_price": current_price,
                "stop_price": current_price*0.95,
                "target_price": current_price*1.15,
                "direction": "LONG",
                "pattern_raw": wedge
            })

    # 8) Harmonic
    harm = patterns_dict.get("harmonic", {})
    if harm.get("found"):
        xabc = harm.get("xabc", [])
        if len(xabc)==5:
            d_price = xabc[-1][1]
            direction = "LONG"  # or check pattern_name
            stop_price = d_price*0.97
            target_price = d_price*1.20
            results["harmonic"].append({
                "entry_price": d_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "direction": direction,
                "pattern_raw": harm
            })

     # Cup&Handle => genelde bullish
    cup = patterns_dict.get("cup_handle", {})
    if isinstance(cup, dict):
        cup = [cup]
    for c_ in cup:
        if c_.get("found", False):
            # Basit => bullish assumption
            results["cup_handle"].append({
                "entry_price": current_price,
                "stop_price": current_price*0.95,
                "target_price": current_price*1.2,
                "direction": "LONG",
                "pattern_raw": c_
            })

    # Flag/Pennant => trend continuation
    fp = patterns_dict.get("flag_pennant", {})
    if isinstance(fp, dict):
        fp = [fp]
    for f_ in fp:
        if f_.get("found", False):
            results["flag_pennant"].append({
                "entry_price": current_price,
                "stop_price": current_price*0.96,
                "target_price": current_price*1.15,
                "direction": "LONG",
                "pattern_raw": f_
            })

    # Channel => ascending => long vs.
    chn = patterns_dict.get("channel", {})
    if isinstance(chn, dict):
        chn = [chn]
    for c_ in chn:
        if c_.get("found", False):
            direction= "LONG"
            if c_.get("channel_type")=="descending":
                direction= "SHORT"
            elif c_.get("channel_type")=="horizontal":
                # belki no-trade
                pass
            results["channel"].append({
                "entry_price": current_price,
                "stop_price": current_price*0.95,
                "target_price": current_price*1.15,
                "direction": direction,
                "pattern_raw": c_
            })

    return results


##############################################################################
# 6) SIGNAL ENGINE (generate_signals) => v2 (EK PATTERNLER DAHİL)
##############################################################################
def generate_signals(df: pd.DataFrame,
                     time_frame: str="1m",
                     ml_model: PatternEnsembleModel = None,
                     max_bars_ago: int = 300,
                     require_confirmed: bool= False,
                     check_rsi_macd: bool= False) -> dict:
    """
    Tüm patternleri (Cup&Handle, Flag, Channel, Gann vs.) + ML + basit breakout/hacim 
    + RSI/MACD opsiyonel => final sinyal.

    Dönüş => {
      "signal": "BUY"/"SELL"/"HOLD",
      "score": int,
      "reason": str,
      "patterns": {...},
      "ml_label": 0/1/2,
      "breakout_up": bool,
      "breakout_down": bool,
      "volume_spike": bool,
      "time_frame": str,
      "pattern_trade_levels": {...}
    }
    """
    # 1) TIMEFRAME_CONFIGS
    if time_frame not in TIMEFRAME_CONFIGS:
        raise ValueError(f"Invalid time_frame='{time_frame}'")

    tf_settings = TIMEFRAME_CONFIGS[time_frame]
    sys_params  = tf_settings["system_params"]
    pat_conf    = tf_settings["pattern_config"]

    # 2) Pivot + ZigZag
    scanner = PivotScanner(
        df=df,
        time_frame=time_frame,
        left_bars= sys_params["pivot_left_bars"],
        right_bars=sys_params["pivot_right_bars"],
        volume_factor= 1.2 if sys_params["volume_filter"] else 0.0,
        atr_factor= sys_params["min_atr_factor"],
    )
    pivots= scanner.find_pivots()
    wave  = build_zigzag_wave(pivots)

    # 3) Pattern detection
    patterns= detect_all_patterns_v2(pivots, wave, df=df, time_frame=time_frame, config=pat_conf)

    # 4) ML
    ml_label= None
    if ml_model is not None and wave:
        feats= ml_model.extract_features(wave)
        ml_label= ml_model.predict(feats)[0]

    # 5) Breakout + volume
    b_up, b_down, v_spike= check_breakout_volume(df, time_frame=time_frame)

    # 6) RSI/MACD => opsiyonel
    rsi_macd_signal= True
    if check_rsi_macd and len(df)>0:
        from . import indicator_checks  # ya da yukarıda tanımladığınız
        idx_check= len(df)-1
        ind_res= indicator_checks(df, idx_check, time_frame=time_frame)
        rsi_macd_signal= ind_res["verdict"]

    # 7) Skor => benzer mantık (H&S => -3, inverse => +3, vb.)
    pattern_score= 0
    reasons= []

    # Filtrelemeye yarayan fonksiyon (patternde end_bar vs.)
    def filter_patterns(pat_list):
        if isinstance(pat_list, dict):
            pat_list = [pat_list]
        cutoff = len(df) - max_bars_ago
        out = []
        for p in pat_list:
            endb = p.get("end_bar", p.get("confirmed_bar", 0))
            
            # Eğer endb None ise, pattern’i yok say
            if endb is None:
                continue
            
            if endb >= cutoff:
                if require_confirmed:
                    if p.get("confirmed", False):
                        out.append(p)
                else:
                    out.append(p)
        return out


    hs_list= filter_patterns(patterns["headshoulders"])
    for hs in hs_list:
        val= -3
        if hs.get("confirmed") and hs.get("volume_check",True):
            val= -4
        pattern_score += val
        reasons.append(f"headshoulders({val})")

    inv_hs_list= filter_patterns(patterns["inverse_headshoulders"])
    for inv in inv_hs_list:
        val= +3
        if inv.get("confirmed") and inv.get("volume_check",True):
            val= +4
        pattern_score+= val
        reasons.append(f"inverseHS({val})")

    # Double / Triple ...
    # ...

    # Cup&Handle => +2 => example
    cpl= filter_patterns(patterns["cup_handle"])
    for cp in cpl:
        val= +2
        if cp.get("confirmed"): val+=1
        pattern_score+= val
        reasons.append(f"cup_handle({val})")

    # Flag/Pennant => +2 => ...
    # Channel => +1 (ascending), -1 (descending), 0 (horizontal)...

    # Gann => +1 if found
    gann_data= patterns["gann"]
    if gann_data.get("found"): 
        pattern_score +=1
        reasons.append("gann(+1)")

    # (Bu bölümü kısalttık, benzer mantık.)

    # ML => 0=HOLD,1=BUY,2=SELL
    if ml_label==1:
        pattern_score+= 3
        reasons.append("ml_buy")
    elif ml_label==2:
        pattern_score-= 3
        reasons.append("ml_sell")

    final_score= pattern_score
    if final_score>0:
        if b_up:
            final_score+=1
            reasons.append("breakout_up")
            if v_spike:
                final_score+=1
                reasons.append("vol_spike_up")
    elif final_score<0:
        if b_down:
            final_score-=1
            reasons.append("breakout_down")
            if v_spike:
                final_score-=1
                reasons.append("vol_spike_down")

    if check_rsi_macd:
        if not rsi_macd_signal:
            final_score-=1
            reasons.append("rsi_macd_fail")

    final_signal= "HOLD"
    if final_score>=2:
        final_signal= "BUY"
    elif final_score<= -2:
        final_signal= "SELL"

    reason_str= ",".join(reasons) if reasons else "NONE"
    current_price= df[get_col_name("Close", time_frame)].iloc[-1] if len(df)>0 else 0.0

    # pattern_trade_levels => entry/stop/target
    pattern_trade_levels= extract_pattern_trade_levels(patterns, current_price)

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
        "pattern_trade_levels": pattern_trade_levels
    }
