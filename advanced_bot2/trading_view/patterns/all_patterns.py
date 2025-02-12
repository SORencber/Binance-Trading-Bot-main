import math
import os
import json
import warnings
import pandas as pd
import numpy as np
import concurrent.futures
from itertools import product
from datetime import datetime, date
from trading_view.patterns.harmonics import detect_harmonic_pattern_advanced
from trading_view.patterns.cup_handle import detect_cup_and_handle_advanced
from trading_view.patterns.channel import detect_channel_advanced
from trading_view.patterns.double_bottom import detect_double_bottom
from trading_view.patterns.double_top import detect_double_top
from trading_view.patterns.elliott import detect_elliott_5wave_strict
from trading_view.patterns.flag_pennant import detect_flag_pennant_advanced
from trading_view.patterns.gann import detect_gann_pattern_ultra_final
from trading_view.patterns.headshoulders import (detect_head_and_shoulders_advanced)
from trading_view.patterns.inverse_headshoulders import detect_inverse_head_and_shoulders_advanced
from trading_view.patterns.rectangle_range import detect_rectangle_range
from trading_view.patterns.triangle import detect_triangle_advanced
from trading_view.patterns.triple_bottom import detect_triple_bottom_advanced
from trading_view.patterns.triple_top import detect_triple_top_advanced
from trading_view.patterns.wedge import detect_wedge_advanced
from trading_view.patterns.wolfe import detect_wolfe_wave_advanced


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
                "": 1.3
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

pattern_param_grids = {
    "head_and_shoulders": {
        "1m": {
            "left_bars": [3, 5],
            "right_bars": [3, 5],
            "min_distance_bars": [10, 20],
            "shoulder_tolerance": [0.02, 0.03],
            "volume_decline": [True, False],
            "neckline_break": [True],
            "max_shoulder_width_bars": [40, 50],
            "atr_filter": [0.0, 0.2],
            "check_rsi_macd": [False,True],
            "check_retest": [False,True],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "left_bars": [5, 10],
            "right_bars": [5, 10],
            "min_distance_bars": [15, 25],
            "shoulder_tolerance": [0.02, 0.04],
            "volume_decline": [True, False],
            "neckline_break": [True],
            "max_shoulder_width_bars": [50, 60],
            "atr_filter": [0.2, 0.3],
            "check_rsi_macd": [False, True],
            "check_retest": [False, True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [20, 30],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
            "30m": {
            "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1h": {
               "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
               "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
               "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        }
    },

    "inverse_head_and_shoulders": {
        "1m": {
            "left_bars": [3, 5],
            "right_bars": [3, 5],
            "min_distance_bars": [10, 20],
            "shoulder_tolerance": [0.02, 0.03],
            "volume_decline": [True, False],
            "neckline_break": [True],
            "max_shoulder_width_bars": [40, 50],
            "atr_filter": [0.0, 0.2],
            "check_rsi_macd": [False],
            "check_retest": [False],
            "retest_tolerance": [0.015,0.02]
        },
        "5m": {
            "left_bars": [5, 10],
            "right_bars": [5, 10],
            "min_distance_bars": [15, 25],
            "shoulder_tolerance": [0.02, 0.04],
            "volume_decline": [True, False],
            "neckline_break": [True],
            "max_shoulder_width_bars": [50, 60],
            "atr_filter": [0.2, 0.3],
            "check_rsi_macd": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.015,0.02]
        },
        "15m": {
            "left_bars": [8, 12, 15],
            "right_bars": [8, 12, 15],
            "min_distance_bars": [20, 30],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },  "30m": {
               "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1h": {
             "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
              "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
             "left_bars": [5, 15],
            "right_bars": [5, 15],
            "min_distance_bars": [25, 35],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [60, 80],
            "atr_filter": [0.2, 0.5],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        }
    },

    "double_top": {
        "1m": {
            "tolerance": [0.008, 0.01],
            "min_distance_bars": [10, 20],
            "triple_variation": [True, False],
            "volume_check": [True, False],
            "neckline_break": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.015,0.02]
        },
        "5m": {
            "tolerance": [0.01, 0.015],
            "min_distance_bars": [20, 30],
            "triple_variation": [True, False],
            "volume_check": [True],
            "neckline_break": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.015,0.02]
        },
        "15m": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
         "30m": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1h": {
             "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
             "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
             "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        }
    },

    "double_bottom": {
        "1m": {
            "tolerance": [0.008, 0.01],
            "min_distance_bars": [10, 20],
            "triple_variation": [True, False],
            "volume_check": [True, False],
            "neckline_break": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.015,0.02]
        },
        "5m": {
            "tolerance": [0.01, 0.015],
            "min_distance_bars": [20, 30],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.015,0.02]
        },
        "15m": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
          "30m": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1h": {
         "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [30, 40],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        }
    },

    "triple_top_advanced": {
        "1m": {
            "tolerance": [0.01, 0.02],
            "min_distance_bars": [10, 20],
            "volume_check": [True, False],
            "volume_col_factor": [0.8, 1.0],
            "neckline_break": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [15, 25],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
          "30m": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
              "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
              "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
             "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "triple_bottom_advanced": {
        "1m": {
            "tolerance": [0.01, 0.02],
            "min_distance_bars": [10, 20],
            "volume_check": [True, False],
            "volume_col_factor": [0.8, 1.0],
            "neckline_break": [True, False],
            "check_retest": [False, True],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "tolerance": [0.015, 0.02],
            "min_distance_bars": [15, 25],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }, 
        "30m": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
             "tolerance": [0.02, 0.03],
            "min_distance_bars": [20, 30],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "elliott": {
        "1m": {
            # minimum pivot sayısı (dalga sayısı) için
            "wave_min_bars": [5, 7],

            # FIB toleransı
            "fib_tolerance": [0.1, 0.15],

            # 3. dalga en az kaç katı: wave3 >= 1.5 * wave1 veya wave3 >= 1.618 * wave1
            "rule_3rdwave_min_percent": [1.5, 1.618],

            # 3.dalga en kısa dalga olmasın
            "rule_3rdwave_not_shortest": [True],

            # 4.dalga, 1.dalga bölgesine girmesin mi?
            "allow_4th_overlap": [False],

            # 2.dalga, 1.dalganın başlangıcını aşıyor mu? (Klasik Elliott'ta genelde aşmaz)
            "allow_wave2_above_wave1_start": [False],

            # Dalga2 ve Dalga4 için beklenen FIB retracement aralıkları
            "wave2_fib_range": [(0.382, 0.618), (0.382, 0.786)],
            "wave4_fib_range": [(0.382, 0.618), (0.382, 0.786)],

            # 5.dalga uzaması kontrolü (extended 5th) => eski 'extended_waves' yerine
            "check_extended_5th": [True, False],

            # 5.dalga uzama aralığı
            "rule_5thwave_ext_range": [(1.0, 1.618)],

            # ABC düzeltme kontrolü
            "check_abc_correction": [True, False],

            # Dalga aralarındaki minimum bar mesafesi
            "min_bar_distance": [3, 5],

            # Dalga4 retest kontrolü
            "check_retest": [False, True],
        },
        "5m": {
            "wave_min_bars": [7, 10],
            "fib_tolerance": [0.12, 0.15],
            "rule_3rdwave_min_percent": [1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618), (0.382, 0.786)],
            "wave4_fib_range": [(0.382, 0.618), (0.382, 0.786)],
            "check_extended_5th": [True],
            "rule_5thwave_ext_range": [(1.0, 1.618)],
            "check_abc_correction": [True],
            "min_bar_distance": [3, 5],
            "check_retest": [False, True],
        },
        "15m": {
            "wave_min_bars": [10, 15],
            "fib_tolerance": [0.1, 0.2],
            "rule_3rdwave_min_percent": [1.5, 1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618)],
            "wave4_fib_range": [(0.382, 0.618)],
            "check_extended_5th": [True],
            "rule_5thwave_ext_range": [(1.0, 1.618)],
            "check_abc_correction": [True],
            "min_bar_distance": [5],
            "check_retest": [True],
        },
        "30m": {
            "wave_min_bars": [10, 15],
            "fib_tolerance": [0.1, 0.2],
            "rule_3rdwave_min_percent": [1.5, 1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618)],
            "wave4_fib_range": [(0.382, 0.618)],
            "check_extended_5th": [True],
            "rule_5thwave_ext_range": [(1.0, 1.618)],
            "check_abc_correction": [True],
            "min_bar_distance": [5],
            "check_retest": [True],
        },
        "1h": {
            "wave_min_bars": [15, 20],
            "fib_tolerance": [0.1, 0.2],
            "rule_3rdwave_min_percent": [1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618)],
            "wave4_fib_range": [(0.382, 0.618)],
            "check_extended_5th": [True],
            "rule_5thwave_ext_range": [(1.0, 1.618)],
            "check_abc_correction": [True],
            "min_bar_distance": [5],
            "check_retest": [True],
        },
        "4h": {
            "wave_min_bars": [15, 20],
            "fib_tolerance": [0.1, 0.2],
            "rule_3rdwave_min_percent": [1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618)],
            "wave4_fib_range": [(0.382, 0.618)],
            "check_extended_5th": [True],
            "rule_5thwave_ext_range": [(1.0, 1.618)],
            "check_abc_correction": [True],
            "min_bar_distance": [5],
            "check_retest": [True],
        },
        "1d": {
            "wave_min_bars": [15, 20],
            "fib_tolerance": [0.1, 0.2],
            "rule_3rdwave_min_percent": [1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618)],
            "wave4_fib_range": [(0.382, 0.618)],
            "check_extended_5th": [True],
            "rule_5thwave_ext_range": [(1.0, 1.618)],
            "check_abc_correction": [True],
            "min_bar_distance": [5],
            "check_retest": [True],
        }
    },
   
    "wolfe": {
        "1m": {
            "price_tolerance": [0.02, 0.03],
            "strict_lines": [False],
            "breakout_confirm": [True, False],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [False],
            "check_time_symmetry": [False],
            "max_time_ratio": [0.3, 0.4],
            "check_retest": [False]
        },
        "5m": {
            "price_tolerance": [0.03, 0.04],
            "strict_lines": [False],
            "breakout_confirm": [True],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [True],
            "check_time_symmetry": [True],
            "max_time_ratio": [0.3],
            "check_retest": [True]
        },
        "15m": {
            "price_tolerance": [0.03, 0.05],
            "strict_lines": [False, True],
            "breakout_confirm": [True],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [True],
            "check_time_symmetry": [True],
            "max_time_ratio": [0.3],
            "check_retest": [True]
        },
         "30m": {
            "price_tolerance": [0.03, 0.05],
            "strict_lines": [False, True],
            "breakout_confirm": [True],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [True],
            "check_time_symmetry": [True],
            "max_time_ratio": [0.3],
            "check_retest": [True]
        },
        "1h": {
            "price_tolerance": [0.02, 0.04],
            "strict_lines": [True],
            "breakout_confirm": [True],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [True],
            "check_time_symmetry": [True],
            "max_time_ratio": [0.25],
            "check_retest": [True]
        },
        "4h": {
            "price_tolerance": [0.02, 0.03],
            "strict_lines": [True],
            "breakout_confirm": [True],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [True],
            "check_time_symmetry": [True],
            "max_time_ratio": [0.25],
            "check_retest": [True]
        },
        "1d": {
            "price_tolerance": [0.02, 0.03],
            "strict_lines": [True],
            "breakout_confirm": [True],
            "line_projection_check": [True],
            "check_2_4_slope": [True],
            "check_1_4_intersection_time": [True],
            "check_time_symmetry": [True],
            "max_time_ratio": [0.2],
            "check_retest": [True]
        }
    },

    "harmonic": {
        "1m": {
            "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
             "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
             "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
         "30m": {
              "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
             "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
              "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","bat"], ["crab","butterfly","shark","cipher"]],
            "check_volume": [False],
            "volume_factor": [1.2, 1.3],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "triangle": {
        "1m": {
            "triangle_tolerance": [0.01, 0.02],
            "check_breakout": [True, False],
            "check_retest": [False],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "5m": {
            "triangle_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True, False],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "15m": {
            "triangle_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
          "30m": {
            "triangle_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "1h": {
            "triangle_tolerance": [0.015, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "4h": {
            "triangle_tolerance": [0.015, 0.025],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "1d": {
            "triangle_tolerance": [0.01, 0.02],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        }
    },

    "wedge": {
        "1m": {
            "wedge_tolerance": [0.01, 0.02],
            "check_breakout": [True, False],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "wedge_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [False, True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "wedge_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
         "30m": {
            "wedge_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
            "wedge_tolerance": [0.015, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "wedge_tolerance": [0.015, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "wedge_tolerance": [0.01, 0.02],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "cup_handle": {
        "1m": {
            "tolerance": [0.02, 0.03],
            "volume_drop_check": [True, False],
            "volume_drop_ratio": [0.2, 0.3],
            "cup_min_bars": [20, 30],
            "cup_max_bars": [100, 150],
            "handle_ratio": [0.2, 0.3],
            "handle_max_bars": [30, 50],
            "close_above_rim": [True],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "tolerance": [0.02, 0.04],
            "volume_drop_check": [True],
            "volume_drop_ratio": [0.2, 0.4],
            "cup_min_bars": [30, 50],
            "cup_max_bars": [150, 200],
            "handle_ratio": [0.3, 0.4],
            "handle_max_bars": [40, 60],
            "close_above_rim": [True],
            "check_retest": [True, False],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "tolerance": [0.02, 0.05],
            "volume_drop_check": [True],
            "volume_drop_ratio": [0.2, 0.4],
            "cup_min_bars": [50, 70],
            "cup_max_bars": [200, 300],
            "handle_ratio": [0.3, 0.5],
            "handle_max_bars": [50, 80],
            "close_above_rim": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
          "30m": {
            "tolerance": [0.02, 0.05],
            "volume_drop_check": [True],
            "volume_drop_ratio": [0.2, 0.4],
            "cup_min_bars": [50, 70],
            "cup_max_bars": [200, 300],
            "handle_ratio": [0.3, 0.5],
            "handle_max_bars": [50, 80],
            "close_above_rim": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
            "tolerance": [0.02, 0.05],
            "volume_drop_check": [True],
            "volume_drop_ratio": [0.3, 0.5],
            "cup_min_bars": [80, 120],
            "cup_max_bars": [300, 400],
            "handle_ratio": [0.3, 0.5],
            "handle_max_bars": [80, 120],
            "close_above_rim": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "tolerance": [0.02, 0.05],
            "volume_drop_check": [True],
            "volume_drop_ratio": [0.3, 0.5],
            "cup_min_bars": [120, 200],
            "cup_max_bars": [400, 600],
            "handle_ratio": [0.3, 0.5],
            "handle_max_bars": [100, 150],
            "close_above_rim": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "tolerance": [0.02, 0.05],
            "volume_drop_check": [True],
            "volume_drop_ratio": [0.3, 0.5],
            "cup_min_bars": [200, 300],
            "cup_max_bars": [600, 900],
            "handle_ratio": [0.3, 0.5],
            "handle_max_bars": [150, 200],
            "close_above_rim": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "flag_pennant": {
        "1m": {
            "min_flagpole_bars": [10, 15],
            "impulse_pct": [0.03, 0.05],
            "max_cons_bars": [30, 40],
            "pivot_channel_tolerance": [0.02, 0.03],
            "pivot_triangle_tolerance": [0.02, 0.03],
            "require_breakout": [True],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "min_flagpole_bars": [15, 20],
            "impulse_pct": [0.05, 0.07],
            "max_cons_bars": [40, 50],
            "pivot_channel_tolerance": [0.02],
            "pivot_triangle_tolerance": [0.02],
            "require_breakout": [True],
            "check_retest": [False, True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "min_flagpole_bars": [20, 30],
            "impulse_pct": [0.05, 0.1],
            "max_cons_bars": [50, 60],
            "pivot_channel_tolerance": [0.02],
            "pivot_triangle_tolerance": [0.02],
            "require_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
          "30m": {
            "min_flagpole_bars": [20, 30],
            "impulse_pct": [0.05, 0.1],
            "max_cons_bars": [50, 60],
            "pivot_channel_tolerance": [0.02],
            "pivot_triangle_tolerance": [0.02],
            "require_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
            "min_flagpole_bars": [30, 40],
            "impulse_pct": [0.05, 0.1],
            "max_cons_bars": [60, 80],
            "pivot_channel_tolerance": [0.02],
            "pivot_triangle_tolerance": [0.02],
            "require_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "min_flagpole_bars": [40, 60],
            "impulse_pct": [0.05, 0.1],
            "max_cons_bars": [80, 100],
            "pivot_channel_tolerance": [0.02, 0.03],
            "pivot_triangle_tolerance": [0.02, 0.03],
            "require_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "min_flagpole_bars": [60, 80],
            "impulse_pct": [0.05, 0.1],
            "max_cons_bars": [100, 150],
            "pivot_channel_tolerance": [0.02, 0.03],
            "pivot_triangle_tolerance": [0.02, 0.03],
            "require_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "channel": {
        "1m": {
            "parallel_thresh": [0.02, 0.03],
            "min_top_pivots": [2, 3],
            "min_bot_pivots": [2, 3],
            "max_iter": [10],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02]
        },
        "5m": {
            "parallel_thresh": [0.03, 0.05],
            "min_top_pivots": [3, 4],
            "min_bot_pivots": [3, 4],
            "max_iter": [10],
            "check_retest": [False, True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "parallel_thresh": [0.03, 0.05],
            "min_top_pivots": [3, 4],
            "min_bot_pivots": [3, 4],
            "max_iter": [10],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
         "30m": {
            "parallel_thresh": [0.03, 0.05],
            "min_top_pivots": [3, 4],
            "min_bot_pivots": [3, 4],
            "max_iter": [10],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
            "parallel_thresh": [0.02, 0.04],
            "min_top_pivots": [3, 4],
            "min_bot_pivots": [3, 4],
            "max_iter": [10],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "parallel_thresh": [0.02, 0.03],
            "min_top_pivots": [3, 4, 5],
            "min_bot_pivots": [3, 4, 5],
            "max_iter": [10],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "parallel_thresh": [0.015, 0.03],
            "min_top_pivots": [4, 5],
            "min_bot_pivots": [4, 5],
            "max_iter": [10],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "gann": {
        "1m": {
            "use_ultra": [True, False],
            "pivot_window": [100, 150,250],
            "anchor_count": [2, 3],
            "pivot_select_mode": ["extremes_vol","extremes_only"],
            "angles": [None],
            "line_tolerance": [0.005, 0.01],
            "min_line_respects": [2, 3],
            "sq9_variant": ["sqrt_plus_360", "sqrt_basic"],
            "sq9_steps": [3, 5],
            "sq9_tolerance": [0.01],
            "w24_variant": ["typeB","typeA"],
            "w24_steps": [3, 5],
            "w24_tolerance": [0.01],
            "cycles": [None],
            "astro_cycles": [None],
            "cycle_pivot_tolerance": [2, 3],
            "pivot_left_bars": [2,3],
            "pivot_right_bars": [2,3],
            "atr_filter": [True, False],
            "volume_filter": [False],
            "additional_angle_shift": [180.0],
            "check_retest": [False],
            "retest_tolerance": [0.01,0.02]
        },
        "5m": {
            "use_ultra": [True, False],
            "pivot_window": [150, 200,350],
            "anchor_count": [3],
            "pivot_select_mode": ["extremes_vol","extremes_only"],
            "angles": [None],
            "line_tolerance": [0.005, 0.01],
            "min_line_respects": [3],
            "sq9_variant": ["sqrt_plus_360"],
            "sq9_steps": [5, 7],
            "sq9_tolerance": [0.01],
            "w24_variant": ["typeB","typeA"],
            "w24_steps": [5],
            "w24_tolerance": [0.01],
            "cycles": [None],
            "astro_cycles": [None],
            "cycle_pivot_tolerance": [2, 3],
            "pivot_left_bars": [3,4],
            "pivot_right_bars": [3,4],
            "atr_filter": [True, False],
            "volume_filter": [False, True],
            "additional_angle_shift": [180.0],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.015]
        },
        "15m": {
            "use_ultra": [True],
            "pivot_window": [200, 300,400],
            "anchor_count": [3],
            "pivot_select_mode": ["extremes_vol"],
            "angles": [None],
            "line_tolerance": [0.01],
            "min_line_respects": [3,4],
            "sq9_variant": ["sqrt_plus_360","log_spiral"],
            "sq9_steps": [5, 7],
            "sq9_tolerance": [0.01],
            "w24_variant": ["typeB"],
            "w24_steps": [5],
            "w24_tolerance": [0.01],
            "cycles": [[30,90]],
            "astro_cycles": [[90,180]],
            "cycle_pivot_tolerance": [2, 4],
            "pivot_left_bars": [3,5],
            "pivot_right_bars": [3,5],
            "atr_filter": [True],
            "volume_filter": [True],
            "additional_angle_shift": [180.0],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.015]
        },
         "30m": {
            "use_ultra": [True],
            "pivot_window": [200, 300,400],
            "anchor_count": [3],
            "pivot_select_mode": ["extremes_vol"],
            "angles": [None],
            "line_tolerance": [0.01],
            "min_line_respects": [3,4],
            "sq9_variant": ["sqrt_plus_360","log_spiral"],
            "sq9_steps": [5, 7],
            "sq9_tolerance": [0.01],
            "w24_variant": ["typeB"],
            "w24_steps": [5],
            "w24_tolerance": [0.01],
            "cycles": [[30,90]],
            "astro_cycles": [[90,180]],
            "cycle_pivot_tolerance": [2, 4],
            "pivot_left_bars": [3,5],
            "pivot_right_bars": [3,5],
            "atr_filter": [True],
            "volume_filter": [True],
            "additional_angle_shift": [180.0],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.015]
        },
        "1h": {
            "use_ultra": [True],
            "pivot_window": [300, 400,500],
            "anchor_count": [3],
            "pivot_select_mode": ["extremes_vol"],
            "angles": [None],
            "line_tolerance": [0.01],
            "min_line_respects": [3,4],
            "sq9_variant": ["sqrt_plus_360","log_spiral"],
            "sq9_steps": [5, 8],
            "sq9_tolerance": [0.01, 0.02],
            "w24_variant": ["typeB","typeA"],
            "w24_steps": [5],
            "w24_tolerance": [0.01],
            "cycles": [[30,90,180]],
            "astro_cycles": [[90,180]],
            "cycle_pivot_tolerance": [2, 4],
            "pivot_left_bars": [5,6],
            "pivot_right_bars": [5,6],
            "atr_filter": [True],
            "volume_filter": [True],
            "additional_angle_shift": [180.0],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.015]
        },
        "4h": {
            "use_ultra": [True],
            "pivot_window": [400, 600,900],
            "anchor_count": [3],
            "pivot_select_mode": ["extremes_vol"],
            "angles": [None],
            "line_tolerance": [0.01, 0.02],
            "min_line_respects": [3,4],
            "sq9_variant": ["sqrt_plus_360","log_spiral"],
            "sq9_steps": [5, 9],
            "sq9_tolerance": [0.01, 0.02],
            "w24_variant": ["typeB"],
            "w24_steps": [5, 7],
            "w24_tolerance": [0.01, 0.02],
            "cycles": [[30,90,180]],
            "astro_cycles": [[90,180,360]],
            "cycle_pivot_tolerance": [3,5],
            "pivot_left_bars": [5,8],
            "pivot_right_bars": [5,8],
            "atr_filter": [True],
            "volume_filter": [True],
            "additional_angle_shift": [180.0],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.015]
        },
        "1d": {
            "use_ultra": [True],
            "pivot_window": [600, 900,1200],
            "anchor_count": [3],
            "pivot_select_mode": ["extremes_vol"],
            "angles": [None],
            "line_tolerance": [0.01, 0.02],
            "min_line_respects": [3,5],
            "sq9_variant": ["sqrt_plus_360","log_spiral"],
            "sq9_steps": [5, 9],
            "sq9_tolerance": [0.01, 0.02],
            "w24_variant": ["typeB","typeA"],
            "w24_steps": [5, 8],
            "w24_tolerance": [0.01, 0.02],
            "cycles": [[30,90,180]],
            "astro_cycles": [[90,180,360]],
            "cycle_pivot_tolerance": [3,5],
            "pivot_left_bars": [8,10],
            "pivot_right_bars": [8,10],
            "atr_filter": [True],
            "volume_filter": [True],
            "additional_angle_shift": [180.0],
            "check_retest": [True],
            "retest_tolerance": [0.01,0.015]
        }
    },
  
    "rectangle":{
    "1m": {
        "parallel_thresh":      [0.01, 0.02],
        "min_top_pivots":       [2],
        "min_bot_pivots":       [2],
        "min_bars_width":       [10, 20],
        "max_bars_width":       [200, 300],
        "breakout_confirm":     [True, False],
        "check_retest":         [True, False],
        "retest_tolerance":     [0.005, 0.01]
    },
    "5m": {
        "parallel_thresh":      [0.01, 0.02],
        "min_top_pivots":       [2, 3],
        "min_bot_pivots":       [2, 3],
        "min_bars_width":       [15, 20],
        "max_bars_width":       [300, 400],
        "breakout_confirm":     [True, False],
        "check_retest":         [True],
        "retest_tolerance":     [0.005, 0.01]
    },
    "15m": {
        "parallel_thresh":      [0.01, 0.02],
        "min_top_pivots":       [2, 3],
        "min_bot_pivots":       [2, 3],
        "min_bars_width":       [20, 30],
        "max_bars_width":       [400, 600],
        "breakout_confirm":     [True],
        "check_retest":         [True],
        "retest_tolerance":     [0.005, 0.01, 0.015]
    },
    "30m": {
        "parallel_thresh":      [0.01, 0.02],
        "min_top_pivots":       [2, 3],
        "min_bot_pivots":       [2, 3],
        "min_bars_width":       [30, 40],
        "max_bars_width":       [600, 800],
        "breakout_confirm":     [True],
        "check_retest":         [True],
        "retest_tolerance":     [0.01, 0.015]
    },
    "1h": {
        "parallel_thresh":      [0.01, 0.02],
        "min_top_pivots":       [2, 3],
        "min_bot_pivots":       [2, 3],
        "min_bars_width":       [40, 60],
        "max_bars_width":       [800, 1000],
        "breakout_confirm":     [True],
        "check_retest":         [True],
        "retest_tolerance":     [0.01, 0.02]
    },
    "4h": {
        "parallel_thresh":      [0.01, 0.02],
        "min_top_pivots":       [2, 3],
        "min_bot_pivots":       [2, 3],
        "min_bars_width":       [50, 80],
        "max_bars_width":       [1000, 1500],
        "breakout_confirm":     [True],
        "check_retest":         [True],
        "retest_tolerance":     [0.01, 0.02]
    },
    "1d": {
        "parallel_thresh":      [0.005, 0.01],
        "min_top_pivots":       [2, 3],
        "min_bot_pivots":       [2, 3],
        "min_bars_width":       [60, 100],
        "max_bars_width":       [2000, 3000],
        "breakout_confirm":     [True],
        "check_retest":         [True],
        "retest_tolerance":     [0.01, 0.02]
    }
}

}
##############################################################################
# 4) ATR ve Volume için Hazırlık
##############################################################################

def get_col_name(base_col: str, time_frame: str) -> str:
    return f"{base_col}_{time_frame}"

def prepare_atr(df: pd.DataFrame, time_frame: str = "1m", period: int = 14):
    """
    ATR hesaplama ve df'ye ekleme
    """
    high_col  = get_col_name("High",  time_frame)
    low_col   = get_col_name("Low",   time_frame)
    close_col = get_col_name("Close", time_frame)
    atr_col   = get_col_name("ATR",   time_frame)

    if atr_col in df.columns:
        return
    df[f"H-L_{time_frame}"] = df[high_col] - df[low_col]
    df[f"H-PC_{time_frame}"] = (df[high_col] - df[close_col].shift(1)).abs()
    df[f"L-PC_{time_frame}"] = (df[low_col]  - df[close_col].shift(1)).abs()

    df[f"TR_{time_frame}"] = df[[f"H-L_{time_frame}",
                                 f"H-PC_{time_frame}",
                                 f"L-PC_{time_frame}"]].max(axis=1)
    df[atr_col] = df[f"TR_{time_frame}"].rolling(period).mean()

def prepare_volume_ma(df: pd.DataFrame, time_frame: str="1m", period: int=20):
    """
    Volume_MA_20_{time_frame} hesaplayıp ekler.
    """
    vol_col = get_col_name("Volume", time_frame)
    ma_col  = f"Volume_MA_{period}_{time_frame}"
    if (vol_col in df.columns) and (ma_col not in df.columns):
        df[ma_col] = df[vol_col].rolling(period).mean()

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI hesaplama (basit).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_macd(series: pd.Series, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    MACD line, Signal line, and Histogram.
    """
    fast_ema = series.ewm(span=fastperiod).mean()
    slow_ema = series.ewm(span=slowperiod).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signalperiod).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def indicator_checks(df: pd.DataFrame,
                     idx: int,
                     time_frame: str="1m",
                     rsi_check: bool = True,
                     macd_check: bool = True,
                     rsi_period=14,
                     macd_fast=12,
                     macd_slow=26,
                     macd_signal=9) -> dict:
    """
    Örnek RSI & MACD kontrol fonksiyonu (örnek).
    """
    res = {
        "rsi": None,
        "macd": None,
        "macd_signal": None,
        "verdict": True,
        "msgs": []
    }
    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        res["verdict"] = False
        res["msgs"].append("Close column not found, indicator checks skipped.")
        return res

    # RSI
    if rsi_check:
        rsi_col = f"RSI_{time_frame}"
         
        if rsi_col not in df.columns:
            df[rsi_col] = compute_rsi(df[close_col])
            df[rsi_col]
        if idx < len(df):
            rsi_val = df[rsi_col].iloc[idx]
           
            res["rsi"] = rsi_val
            if (not pd.isna(rsi_val)) and (rsi_val < 50):
                res["verdict"]= False
                res["msgs"].append(f"RSI {rsi_val:.2f} <50 => negative.")
        else:
            res["verdict"]= False
            res["msgs"].append("RSI idx out of range")
    #print("-----",time_frame,rsi_check,rsi_val)

    # MACD
    if macd_check:
        macd_col   = f"MACD_{time_frame}"
        macds_col  = f"MACD_signal_{time_frame}"
        if macd_col not in df.columns or macds_col not in df.columns:
            macd_line, macd_signal, _ = compute_macd(df[close_col], macd_fast, macd_slow, macd_signal)
            df[macd_col]  = macd_line
            df[macds_col] = macd_signal

        if idx < len(df):
            macd_val = df[macd_col].iloc[idx]
            macd_sig = df[macds_col].iloc[idx]
            res["macd"] = macd_val
            res["macd_signal"] = macd_sig
            if macd_val < macd_sig:
                res["verdict"]=False
                res["msgs"].append(f"MACD < Signal at index={idx}")
        else:
            res["verdict"]=False
            res["msgs"].append("MACD idx out of range")

    return res


##############################################################################
# 5) PIVOT SCANNER SINIFI
##############################################################################

class PivotScanner:
    """
    Local maxima/minima bulur. Opsiyonel volume ve ATR filtresi uygular.
    pivot: (index, price, +1/-1)
    """
    def __init__(self,
                 df: pd.DataFrame,
                 time_frame: str = "1m",
                 left_bars: int = 5,
                 right_bars: int= 5,
                 volume_factor: float = 1.2,
                 atr_factor: float= 0.0,
                 volume_ma_period: int=20,
                 atr_period: int=14,
                 min_distance_bars: int=0):
        
        self.df = df
        self.time_frame = time_frame
        self.left_bars= left_bars
        self.right_bars= right_bars
        self.volume_factor= volume_factor
        self.atr_factor= atr_factor
        self.volume_ma_period= volume_ma_period
        self.atr_period= atr_period
        self.min_distance_bars= min_distance_bars

        # Volume/ATR hazırlığı
        prepare_volume_ma(df, time_frame, volume_ma_period)
        if atr_factor > 0:
            prepare_atr(df, time_frame, atr_period)

    def find_pivots(self):
        close_col = get_col_name("Close", self.time_frame)
        if close_col not in self.df.columns:
            return []

        price = self.df[close_col]
        n = len(price)
        pivots = []
        for i in range(self.left_bars, n - self.right_bars):
            val = price.iloc[i]
            left_slice = price.iloc[i - self.left_bars: i]
            right_slice= price.iloc[i+1: i+1 + self.right_bars]
            
            is_local_max = (all(val > l for l in left_slice) and
                            all(val >= r for r in right_slice))
            is_local_min = (all(val < l for l in left_slice) and
                            all(val <= r for r in right_slice))
            if is_local_max:
                if self._pivot_ok(i, val, +1):
                    pivots.append((i, val, +1))
            elif is_local_min:
                if self._pivot_ok(i, val, -1):
                    pivots.append((i, val, -1))

        # min_distance_bars => pivotlar arasında en az X bar boşluk
        if self.min_distance_bars > 0 and len(pivots) > 1:
            filtered = [pivots[0]]
            for j in range(1, len(pivots)):
                if pivots[j][0] - filtered[-1][0] >= self.min_distance_bars:
                    filtered.append(pivots[j])
            pivots = filtered

        return pivots

    def _pivot_ok(self, idx, val, ptype):
        """
        volume_factor ve atr_factor kontrolleri.
        """
        volume_col= get_col_name("Volume", self.time_frame)
        vol_ma_col= f"Volume_MA_{self.volume_ma_period}_{self.time_frame}"
        atr_col   = get_col_name("ATR", self.time_frame)

        # Volume check
        if self.volume_factor>0 and (volume_col in self.df.columns) and (vol_ma_col in self.df.columns):
            vol_now= self.df[volume_col].iloc[idx]
            vol_ma= self.df[vol_ma_col].iloc[idx]
            if (not pd.isna(vol_now)) and (not pd.isna(vol_ma)):
                if vol_now < (self.volume_factor * vol_ma):
                    return False

        # ATR check
        if self.atr_factor>0 and (atr_col in self.df.columns):
            pivot_atr= self.df[atr_col].iloc[idx]
            if not pd.isna(pivot_atr):
                # basit bir check => pivot eşiği
                prev_close= self.df[get_col_name("Close", self.time_frame)].iloc[idx-1] if idx>0 else val
                diff= abs(val - prev_close)
                if diff < (self.atr_factor * pivot_atr):
                    return False

        return True

gann_best_params = {
    "use_ultra": True,
    "pivot_window": 400,
    "anchor_count": 3,
    "pivot_select_mode": "extremes_vol",

    "use_gann_ratios": True,
    "gann_ratios": [1.0, 2.0, 0.5],
    "angles": [45.0, 22.5, 67.5, 90.0, 135.0, 180.0],
    "additional_angle_shift": 180.0,

    "line_tolerance": 0.005,
    "min_line_respects": 3,

    "atr_filter": True,
    "atr_period": 14,
    "atr_factor": 0.5,

    "volume_filter": False,
    "volume_ratio": 1.3,

    "sq9_variant": "typeA",
    "sq9_steps": 5,
    "sq9_tolerance": 0.01,

    "w24_variant": "typeB",
    "w24_steps": 5,
    "w24_tolerance": 0.01,

    "cycles": [30, 90, 180],
    "astro_cycles": [90, 180, 360],
    "cycle_pivot_tolerance": 2,
    "pivot_left_bars": 3,
    "pivot_right_bars": 3,

    "debug": False,
    "check_retest": False,
    "retest_tolerance": 0.01
}

##############################################################################
# 7) PATTERN FN MAP + DETECT ALL
##############################################################################

pattern_fn_map = {
    "head_and_shoulders": detect_head_and_shoulders_advanced,
    "inverse_head_and_shoulders": detect_inverse_head_and_shoulders_advanced,
    "double_top": detect_double_top,
    "double_bottom": detect_double_bottom,
    "triple_top_advanced": detect_triple_top_advanced,
    "triple_bottom_advanced": detect_triple_bottom_advanced,
    "elliott": detect_elliott_5wave_strict,
    "wolfe": detect_wolfe_wave_advanced,
    "harmonic": detect_harmonic_pattern_advanced,
    "triangle": detect_triangle_advanced,
    "wedge": detect_wedge_advanced,
    "cup_handle": detect_cup_and_handle_advanced,
    "flag_pennant": detect_flag_pennant_advanced,
    "channel": detect_channel_advanced,
    "gann": detect_gann_pattern_ultra_final,
    "rectangle": detect_rectangle_range

}


async def detect_all_patterns_v2(
    df: pd.DataFrame,
    symbol: str,
    time_frame: str,
    filename: str = None,
    scoring_fn = None
):
    """
    Belirli bir sembol & time_frame için pivotları tespit eder;
    ardından pattern_fn_map'teki tüm pattern fonksiyonlarını uygun parametrelerle çalıştırır.
    Dönüş => {pattern_name: detection_result, ...}
    """
    if scoring_fn is None:
        scoring_fn = score_pattern_results

    # 1) Pivot stratejisi parametreleri (pivot_strategy) JSON'dan yüklemeyi dene
    pivot_params = load_best_params_from_json(symbol, time_frame, pattern_name=None, filename=filename)
    if pivot_params is None:
        print(f"[detect_all_patterns_v2] => pivot_strategy params not found => optimizing...")
        sys_opt_res = optimize_system_parameters(
            df=df,
            symbol=symbol,
            time_frame=time_frame
        )
        pivot_params = sys_opt_res["best_params"]
    else:
        print(f"[detect_all_patterns_v2] => pivot_strategy params LOADED: {pivot_params}")

    # 2) Pivotları bul
    scanner = PivotScanner(
        df=df,
        time_frame=time_frame,
        left_bars    = pivot_params.get("left_bars", 5),
        right_bars   = pivot_params.get("right_bars", 5),
        volume_factor= pivot_params.get("volume_factor", 1.2),
        atr_factor   = pivot_params.get("atr_factor", 0.0),
        # ... isterseniz pivot_params ekle ...
    )
    pivots = scanner.find_pivots()

    detection_results= {}
    # 3) Tüm pattern fonksiyonlarını sırayla
    for pattern_name, pattern_fn in pattern_fn_map.items():
        is_gann = (pattern_name=="gann")

        if pattern_name not in pattern_param_grids:
            print(f"[detect_all_patterns_v2] => pattern '{pattern_name}' param_grid yok => skip.")
            continue
        tf_dict= pattern_param_grids[pattern_name]
        if time_frame not in tf_dict:
            print(f"[detect_all_patterns_v2] => pattern '{pattern_name}', timeframe='{time_frame}' param yok => skip.")
            continue

        local_pattern_param_grid = tf_dict[time_frame]
        loaded_params= load_best_params_from_json(symbol, time_frame, pattern_name, filename)

        # GANN mi?
        if is_gann:
            if loaded_params:
                best_params= loaded_params
                print(f"[detect_all_patterns_v2] => GANN param LOADED => {best_params}")
            else:
                print("[detect_all_patterns_v2] => GANN param yok => default param kullanılıyor.")
                best_params= gann_best_params
           

        else:
            if loaded_params:
                best_params= loaded_params
            else:
                print(f"[detect_all_patterns_v2] => '{pattern_name}' no JSON => optimizing...")
                opt_res= await optimize_pattern_parameters(
                    df=df,
                    symbol=symbol,
                    time_frame=time_frame,
                    pattern_name=pattern_name,
                    pivots=pivots,
                    filename=filename,
                    scoring_fn=scoring_fn
                )
                best_params= opt_res["best_params"]

        # pattern çalıştır
        detection_result= pattern_fn(
            df=df,
            pivots=pivots,
            time_frame=time_frame,
            **best_params
        )
        # if pattern_name == "gann" and detection_result:
        #     print(detection_result)
        detection_results[pattern_name] = detection_result


    return detection_results


##############################################################################
# 8) JSON KAYIT / YÜKLEME
##############################################################################

def save_best_params_to_json(symbol: str,
                             timeframe: str,
                             best_params: dict,
                             best_score: float,
                             pattern_name: str = None,
                             filename: str = None):
    """
    Tek bir JSON dosyasında parametreleri saklar.
    pattern_name=None => "pivot_strategy"
    """
    if filename is None:
        filename = f"{symbol.lower()}.json"
    if not os.path.exists(filename):
        data = {
            "symbol": symbol,
            "time_frames": {}
        }
    else:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    if "time_frames" not in data:
        data["time_frames"] = {}
    if timeframe not in data["time_frames"]:
        data["time_frames"][timeframe] = {}

    store_key = pattern_name if pattern_name else "pivot_strategy"
    data["time_frames"][timeframe][store_key] = {
        "best_params": best_params,
        "best_score": best_score
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[save_best_params_to_json] => {filename} updated => time_frame={timeframe}, pattern={store_key}")


def load_best_params_from_json(symbol: str,
                               timeframe: str,
                               pattern_name: str = None,
                               filename: str = None) -> dict:
    if filename is None:
        filename = f"{symbol.lower()}.json"
    if not os.path.exists(filename):
        print(f"[load_best_params_from_json] => File '{filename}' not found.")
        return None

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    tf_data = data.get("time_frames", {}).get(timeframe, {})
    if not tf_data:
        return None

    store_key = pattern_name if pattern_name else "pivot_strategy"
    pat_data = tf_data.get(store_key, {})
    if not pat_data:
        return None
    return pat_data.get("best_params", None)


##############################################################################
# 9) OPTİMİZASYON ÖRNEKLERİ
##############################################################################

def calc_strategy_pnl(
    df: pd.DataFrame,
    pivots: list,
    time_frame: str = "1m",
    commission_rate: float = 0.0005,
    slippage: float = 0.0,
    use_stop_loss: bool = True,
    stop_loss_atr_factor: float = 2.0,
    allow_short: bool = True
)-> dict:
    """
    Demo pivot-based al-sat stratejisi => PnL metrikleri.
    """
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        return {
            "pnl": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "long_trades": 0,
            "short_trades": 0
        }

    sorted_piv= sorted(pivots, key=lambda x: x[0])
    position= 0
    position_price= 0.0
    realized_pnl= 0.0
    trade_count= 0
    long_trades= 0
    short_trades= 0

    equity=0.0
    peak_equity=0.0
    max_drawdown=0.0

    for (bar_idx, price, ptype) in sorted_piv:
        # basit mantık: -1 => long aç, +1 => short aç (ya da tam tersi)
        # Burada ptype=-1 "dip" => long, ptype=+1 "top" => short kuralını uygulayalım
        if ptype== -1:  # dip
            if position<=0:
                if position== -1:
                    trade_pnl= (position_price- price)
                    realized_pnl+= trade_pnl
                    equity+= trade_pnl
                    trade_count+=1
                    short_trades+=1
                    position=0
                position=1
                position_price= price
        elif ptype== +1: # top
            if allow_short and position>=0:
                if position==1:
                    trade_pnl= (price- position_price)
                    realized_pnl+= trade_pnl
                    equity+= trade_pnl
                    trade_count+=1
                    long_trades+=1
                    position=0
                position= -1
                position_price= price

        peak_equity= max(peak_equity,equity)
        dd= peak_equity- equity
        max_drawdown= max(max_drawdown, dd)

    # Dönem sonu pozisyon kapama
    if position!=0:
        last_price= df[close_col].iloc[-1]
        if position==1:
            trade_pnl= (last_price- position_price)
            realized_pnl+= trade_pnl
            equity+= trade_pnl
            trade_count+=1
            long_trades+=1
        else:
            trade_pnl= (position_price- last_price)
            realized_pnl+= trade_pnl
            equity+= trade_pnl
            trade_count+=1
            short_trades+=1
        position=0
        peak_equity= max(peak_equity,equity)
        dd= peak_equity- equity
        max_drawdown= max(max_drawdown, dd)

    return {
        "pnl": round(realized_pnl,4),
        "max_drawdown": round(max_drawdown,4),
        "trade_count": trade_count,
        "long_trades": long_trades,
        "short_trades": short_trades
    }

def score_pattern_results(pattern_output):
    """
    Basit bir skor örneği: found => +1, confirmed => +2
    Pattern çıktı dict/list olabilir.
    """
    if not pattern_output:
        return 0.0
    if isinstance(pattern_output, dict):
        items=[pattern_output]
    else:
        items= pattern_output
    total=0.0
    for it in items:
        if it.get("found",False):
            total+=1.0
        if it.get("confirmed",False):
            total+=2.0
    return total

def optimize_system_parameters(
    df,
    symbol: str="BTCUSDT",
    time_frame: str="1m"
)-> dict:
    """
    Pivot parametrelerini tarayarak en iyi PnL skorunu bulur.
    """
    from itertools import product
    param_grid={
        "left_bars": [5,10],
        "right_bars":[5,10],
        "volume_factor":[1.0,1.2],
        "atr_factor":[0.0,0.2]
    }
    best_score= float("-inf")
    best_params= None
    all_results= []
    for vals in product(*[param_grid[k] for k in param_grid.keys()]):
        params= dict(zip(param_grid.keys(), vals))
        scanner= PivotScanner(
            df, time_frame,
            left_bars= params["left_bars"],
            right_bars= params["right_bars"],
            volume_factor= params["volume_factor"],
            atr_factor= params["atr_factor"]
        )
        pivs= scanner.find_pivots()
        metrics= calc_strategy_pnl(df, pivs, time_frame=time_frame)
        score= metrics["pnl"]- 0.1* abs(metrics["max_drawdown"])
        result_item= {
            "params": params,
            "pnl": metrics["pnl"],
            "max_dd": metrics["max_drawdown"],
            "trade_count": metrics["trade_count"],
            "score": score,
            "pivot_count": len(pivs)
        }
        all_results.append(result_item)
        if score> best_score:
            best_score= score
            best_params= params

    save_best_params_to_json(
        symbol= symbol,
        timeframe= time_frame,
        best_params= best_params,
        best_score= best_score,
        filename= None
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results
    }


async def optimize_pattern_parameters(
    df: pd.DataFrame,
    symbol: str,
    time_frame: str,
    pattern_name: str,
    pivots,
    scoring_fn,
    filename: str= None
)-> dict:
    """
    Tek pattern özelinde param_grid deneyerek en iyi score'u bulur.
    """
    param_candidates= pattern_param_grids[pattern_name][time_frame]
    pattern_function= pattern_fn_map[pattern_name]

    best_score= float("-inf")
    best_params= None
    all_results= []

    param_keys= list(param_candidates.keys())
    param_value_lists= [param_candidates[k] for k in param_keys]

    for combo in product(*param_value_lists):
        params= dict(zip(param_keys, combo))
        detection_result= pattern_function(
            df=df,
            pivots=pivots,
            time_frame=time_frame,
            **params
        )
        score_val= scoring_fn(detection_result)
        all_results.append({
            "params": params,
            "score": score_val
        })
        if score_val> best_score:
            best_score= score_val
            best_params= params

    save_best_params_to_json(
        symbol=symbol,
        timeframe=time_frame,
        best_params=best_params,
        best_score=best_score,
        pattern_name=pattern_name,
        filename=filename
    )
    print(f"[optimize_pattern_parameters] => pattern={pattern_name}, best_score={best_score}, best_params={best_params}")
    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results
    }

import pandas as pd
import numpy as np

def detect_advanced_pivots(
    df: pd.DataFrame,
    price_col: str = "Close",
    left_bars: int = 2,
    right_bars: int = 2,
    amplitude_threshold: float = 0.005,
    min_bars_between_pivots: int = 5,
    volume_col: str = None,
    volume_factor: float = 1.2,  # Pivot bölgesinde ortalama hacme göre artış
    rsi_period: int = 14,
    rsi_divergence_check: bool = False,
    rsi_div_threshold: float = 5.0,   # RSI divergence için esnek eşik
    pivot_score_min: float = 50.0,    # Toplam skorun altında kalan pivotlar elensin
    overshadow_factor: float = 1.5,   # Bir pivot, yakında kendisinden 'overshadow_factor' kat büyük pivot varsa elensin
    merge_close_pivots: bool = True,
    merge_distance_bars: int = 3,     # Çok yakın pivotları birleştirme bar aralığı
    merge_distance_price: float = 0.003, # Fiyatta çok yakınsa
    max_lookback: int = 999999
) -> list:
    """
    Çok yönlü, gelişmiş pivot tespit fonksiyonu.
    
    DÖNEN DEĞER:
    [
      {
        'index': pivot_bar_index,
        'price': pivot_price,
        'pivot_type': +1 / -1 (tepe / dip),
        'score': hesaplanan pivot skor,
        'msgs': [ ... ]
      },
      ...
    ]
    
    ADIMLAR:
    1) Fraktal pivot tespiti (left_bars ve right_bars).
    2) Amplitude filtresi (yüksekliğin belli eşiğin üstünde olup olmadığı).
    3) min_bars_between_pivots ile pivotlar arasında en az X bar boşluk sağlama.
    4) Hacim teyidi (varsa volume_col).
    5) RSI divergences (opsiyonel).
    6) Pivot skor hesaplama (yukarıdaki unsurlar).
    7) overshadow_factor ile büyük pivot yakınındaki küçük pivotu atma.
    8) Yakın pivotları birleştirme (merge).
    """

    # ---------------------------
    # Hazırlık: Kolon Kontrolleri
    # ---------------------------
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame.")

    n = len(df)
    if n < (left_bars + right_bars + 1):
        return []  # yeterli veri yok

    # Hacim?
    if volume_col and volume_col not in df.columns:
        volume_col = None  # yoksa devre dışı

    # RSI?
    if rsi_period > 0 and rsi_divergence_check:
        if "rsi_col" not in df.columns:
            df["rsi_col"] = compute_rsi(df[price_col], period=rsi_period)

    # -----------------------------------------
    # 1) Basit Fraktal Pivot Adaylarını Bulma
    # -----------------------------------------
    pivot_candidates = []
    for i in range(left_bars, min(n - right_bars, max_lookback)):
        # Local window
        window_left = df[price_col].iloc[i - left_bars : i]
        window_right = df[price_col].iloc[i + 1 : i + right_bars + 1]
        curr_price = df[price_col].iloc[i]

        # Tepe (local max) => şimdiki fiyat, sol ve sağdaki tüm fiyatlardan büyük
        if all(curr_price > x for x in window_left) and all(curr_price > x for x in window_right):
            pivot_candidates.append({
                'index': i,
                'price': curr_price,
                'pivot_type': +1, # tepe
                'score': 0.0,
                'msgs': []
            })
        # Dip (local min) => şimdiki fiyat, sol ve sağdaki tüm fiyatlardan küçük
        elif all(curr_price < x for x in window_left) and all(curr_price < x for x in window_right):
            pivot_candidates.append({
                'index': i,
                'price': curr_price,
                'pivot_type': -1, # dip
                'score': 0.0,
                'msgs': []
            })

    # -----------------------------------------
    # 2) Amplitude Filtresi
    # -----------------------------------------
    # Pivotun etrafındaki fiyatlarla ortalama farkın amplitude_threshold'u geçip geçmediğini kontrol edebiliriz.
    # Örneğin basitçe pivot fiyatı ile local mean arasındaki fark.
    filtered = []
    for pivot in pivot_candidates:
        i = pivot['index']
        local_start = max(0, i - left_bars)
        local_end = min(n, i + right_bars + 1)
        local_prices = df[price_col].iloc[local_start:local_end]
        local_mean = local_prices.mean()
        diff = abs(pivot['price'] - local_mean) / (local_mean + 1e-9)
        if diff >= amplitude_threshold:
            pivot['score'] += (diff * 100)  # amplitude katkısı
            filtered.append(pivot)
        else:
            pivot['msgs'].append(f"Amplitude check failed: {diff:.4f} < {amplitude_threshold}")

    pivot_candidates = filtered

    # -----------------------------------------
    # 3) Min. bar sayısı (spacing)
    #    Aynı yönde çok sık pivotlar olmasın diye
    # -----------------------------------------
    final_list = []
    last_pivot_i = -999
    for pivot in pivot_candidates:
        i = pivot['index']
        if (i - last_pivot_i) < min_bars_between_pivots:
            pivot['msgs'].append(f"Too close to previous pivot. (bar distance={i - last_pivot_i})")
            continue
        final_list.append(pivot)
        last_pivot_i = i

    pivot_candidates = final_list

    # -----------------------------------------
    # 4) Hacim Teyidi (Var ise)
    # -----------------------------------------
    if volume_col:
        # Ortalama hacim
        avg_vol = df[volume_col].mean()
        updated = []
        for pivot in pivot_candidates:
            i = pivot['index']
            pivot_vol = df[volume_col].iloc[i]
            # Belirli bir oranda büyükse pivot'a ek skor
            if pivot_vol > volume_factor * avg_vol:
                pivot['score'] += 10  # örnek +10 puan
            else:
                pivot['msgs'].append(f"Volume check < factor: {pivot_vol:.2f} < {volume_factor}*{avg_vol:.2f}")
            updated.append(pivot)
        pivot_candidates = updated

    # -----------------------------------------
    # 5) RSI Divergence (Opsiyonel)
    #    Örnek: Tepe pivotunda RSI düşük tepe yapıyorsa => negatif uyumsuzluk.
    #           Dip pivotunda RSI yüksek dip yapıyorsa => pozitif uyumsuzluk.
    # -----------------------------------------
    if rsi_divergence_check and "rsi_col" in df.columns:
        updated = []
        for pivot in pivot_candidates:
            i = pivot['index']
            pivot_type = pivot['pivot_type']
            pivot_rsi = df["rsi_col"].iloc[i]
            # Etrafındaki son pivot RSI'si ile karşılaştırma yapabilirsiniz
            # Basit yaklaşımla, geriye doğru (left_bars) en son benzer pivot tipini bulur vs.
            # Aşağıda basit bir "yakındaki barların RSI ortalaması" örneği
            lookback_rsi_slice = df["rsi_col"].iloc[max(0, i-10):i]
            if len(lookback_rsi_slice) == 0:
                updated.append(pivot)
                continue

            prev_rsi_max = lookback_rsi_slice.max()
            prev_rsi_min = lookback_rsi_slice.min()

            if pivot_type == +1:  # tepe
                # eğer bu pivotun RSI'si önceki tepe RSI'sından belirgin şekilde düşükse => negatif uyumsuzluk
                if pivot_rsi < (prev_rsi_max - rsi_div_threshold):
                    pivot['score'] += 15  # ekstra puan
                    pivot['msgs'].append(f"RSI negative divergence detected. pivot_rsi={pivot_rsi:.2f} < prev_rsi_max={prev_rsi_max:.2f}")
            else:  # dip
                if pivot_rsi > (prev_rsi_min + rsi_div_threshold):
                    pivot['score'] += 15
                    pivot['msgs'].append(f"RSI positive divergence detected. pivot_rsi={pivot_rsi:.2f} > prev_rsi_min={prev_rsi_min:.2f}")

            updated.append(pivot)
        pivot_candidates = updated

    # -----------------------------------------
    # 6) Pivot Skor Eşiği
    # -----------------------------------------
    final_list = []
    for pivot in pivot_candidates:
        if pivot['score'] >= pivot_score_min:
            final_list.append(pivot)
        else:
            pivot['msgs'].append(f"Pivot score={pivot['score']:.2f} < pivot_score_min={pivot_score_min}")
    pivot_candidates = final_list

    # -----------------------------------------
    # 7) "Overshadow" Filtresi
    #    Aynı civarda, bir pivotun amplitude veya skorca çok daha büyük olması halinde 
    #    küçük pivotu atabiliriz.
    # -----------------------------------------
    overshadowed = set()
    for i, p1 in enumerate(pivot_candidates):
        for j, p2 in enumerate(pivot_candidates):
            if i == j:
                continue
            dist = abs(p1['index'] - p2['index'])
            if dist < (left_bars + right_bars)*2:  # Yakın pivotlar
                # Birinin skoru veya tepe/dip amplitude'ı çok yüksekse diğeri "gölge"de kalabilir
                # Burada basitçe "skor"u kullanıyoruz:
                if p2['score'] > p1['score'] * overshadow_factor:
                    overshadowed.add(i)  # p1 gölgelenmiş
                elif p1['score'] > p2['score'] * overshadow_factor:
                    overshadowed.add(j)  # p2 gölgelenmiş
    
    final_list = []
    for i, pivot in enumerate(pivot_candidates):
        if i not in overshadowed:
            final_list.append(pivot)
        else:
            pivot['msgs'].append("Overshadowed by stronger pivot nearby.")
    pivot_candidates = final_list

    # -----------------------------------------
    # 8) Yakın pivotları birleştirme (merge)
    #    Fiyatta ve barda çok yakın pivotlar tek pivot olarak kabul edilebilir.
    # -----------------------------------------
    if merge_close_pivots and len(pivot_candidates) > 1:
        pivot_candidates = sorted(pivot_candidates, key=lambda x: x['index'])
        merged_list = []
        skip_next = False
        for i in range(len(pivot_candidates)):
            if skip_next:
                skip_next = False
                continue
            if i < len(pivot_candidates) - 1:
                p1 = pivot_candidates[i]
                p2 = pivot_candidates[i+1]
                idx_diff = abs(p1['index'] - p2['index'])
                price_diff = abs(p1['price'] - p2['price']) / (p1['price'] + 1e-9)
                # Eğer yeterince yakınsa, birleştirme yap
                if idx_diff <= merge_distance_bars and price_diff <= merge_distance_price and p1['pivot_type'] == p2['pivot_type']:
                    # Daha yüksek skorlu pivotu "birleşik pivot" kabul edelim
                    if p1['score'] >= p2['score']:
                        merged_list.append(p1)
                    else:
                        merged_list.append(p2)
                    skip_next = True
                else:
                    merged_list.append(p1)
            else:
                # Son pivot
                merged_list.append(pivot_candidates[i])
        pivot_candidates = merged_list

    # -----------------------------------------
    # Sonuç
    # -----------------------------------------
    # index'e göre sıralayalım
    pivot_candidates = sorted(pivot_candidates, key=lambda x: x['index'])
    return pivot_candidates


# ---------------------------------------------------
# Yardımcı Fonksiyonlar
# ---------------------------------------------------

def compute_rsi(series: pd.Series, period: int = 14):
    """
    EMA tabanlı basit RSI hesabı.
    """
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

##############################################################################
# 10) PARALEL ÇALIŞMA ÖRNEĞİ (İsteğe Bağlı)
##############################################################################

def run_parallel_scans(symbols, time_frames, df_map: dict, config: dict):
    """
    Örnek: ThreadPoolExecutor ile paralel tarama.
    df_map => { (symbol, time_frame): DataFrame }
    """
    results= {}

    def process(sym, tf):
        df= df_map.get((sym,tf), None)
        if df is None:
            return (sym, tf, None)

        sc= PivotScanner(
            df, tf,
            left_bars= config["system_params"]["pivot_left_bars"],
            right_bars= config["system_params"]["pivot_right_bars"],
            volume_factor=1.2,
            atr_factor=0.0
        )
        pivots= sc.find_pivots()
        # pattern tespit
        # Burada detect_all_patterns_v2 kullanabilir veya tek tek pattern ler...
        # Örnek basit:
        wave= pivots
        # ...
        patterns= None  # tespit
        return (sym, tf, patterns)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map= {}
        for s in symbols:
            for tf in time_frames:
                f= executor.submit(process,s,tf)
                future_map[f]= (s,tf)
        for f in concurrent.futures.as_completed(future_map):
            s,tf= future_map[f]
            try:
                r= f.result()
                results[(s,tf)]= r[2]
            except Exception as e:
                results[(s,tf)]={"error": str(e)}

    return results

def post_pattern_filters(pattern_name,pattern_result: dict,
                         
                        df: pd.DataFrame,
                        time_frame: str = "1m",
                        volume_check: bool = True,
                        volume_multiplier: float = 1.5,
                        atr_check: bool = True,
                        min_atr_distance: float = 2.0,
                        rsi_macd_check: bool = True,
                        rsi_period: int = 14,
                        macd_fast: int = 12,
                        macd_slow: int = 26,
                        macd_signal: int = 9,
                        require_retest: bool = True,
                        min_pattern_percent: float = 0.02
                        ) -> bool:
    """
    Pattern fonksiyonundan gelen 'pattern_result' çıktılarını
    daha da sıkı filtrelere tabi tutar.
    
    Dönüş:
       True  => Filtreler olumlu => pattern sinyali güçlü
       False => Filtrelerden biri/birkaçı geçersiz => pattern elenir
    """
    if not pattern_result or not pattern_result.get("found", False):
        return False

    if "confirmed" in pattern_result and not pattern_result["confirmed"]:
        return False

    close_col = f"Close_{time_frame}"
    if close_col not in df.columns:
        return False

    # Retest
    if require_retest:
        if "retest_info" in pattern_result:
            rinfo = pattern_result["retest_info"]
            if not rinfo or not rinfo.get("retest_done", False):
                return False
        else:
            return False

    # ATR check
    if atr_check:
        atr_col = f"ATR_{time_frame}"
        if atr_col in df.columns:
            atr_now = df[atr_col].iloc[-1]
            # basit kural: pattern ile ATR karşılaştırması
            # Örn. (max_top - neckline) > min_atr_distance * ATR gibi
            # HeadShoulders/doubletop vs. => isterseniz pattern ayrımı
            if pattern_result.get("pattern") in ["head_and_shoulders","double_top","triple_top"]:
                neck = pattern_result.get("neckline", None)
                if neck and isinstance(neck, tuple):
                    neck_price = neck[1]
                    # tops
                    if "tops" in pattern_result:
                        all_tops = pattern_result["tops"]
                        max_top = max(t[1] for t in all_tops)
                        dist = abs(max_top - neck_price)
                        if dist < (min_atr_distance* atr_now):
                            return False

    # min_pattern_percent
    if min_pattern_percent> 0:
        avg_price = df[close_col].iloc[-100:].mean()
        # benzer mantık
        if "neckline" in pattern_result and isinstance(pattern_result["neckline"], tuple):
            nk_prc = pattern_result["neckline"][1]
            if pattern_result.get("pattern") in ["double_top","triple_top"]:
                if "tops" in pattern_result:
                    all_tops = pattern_result["tops"]
                    max_top = max(t[1] for t in all_tops)
                    dist_pct = abs(max_top - nk_prc)/(avg_price+1e-9)
                    if dist_pct< min_pattern_percent:
                        return False

    # Hacim
    if volume_check:
        vol_col = f"Volume_{time_frame}"
        ma_col  = f"Volume_MA_20_{time_frame}"
        df[ma_col] = df[vol_col].rolling(20).mean()

        if vol_col in df.columns and ma_col in df.columns:
            last_vol = df[vol_col].iloc[-1]
            vol_ma20 = df[ma_col].iloc[-1]
            if last_vol < (vol_ma20 * volume_multiplier):
                return False

    # RSI / MACD
    if rsi_macd_check:
        from math import isfinite
        idx = len(df)-1
        ind_res = indicator_checks(df, idx, time_frame=time_frame,
                                   rsi_check=True, macd_check=True,
                                   rsi_period=rsi_period,
                                   macd_fast=macd_fast,
                                   macd_slow=macd_slow,
                                   macd_signal=macd_signal)
        if not ind_res["verdict"]:
            return False

    return True

async def run_detection_with_filters(df, symbol, time_frame,filename):
    # 1) Normal pattern detection
    detection_results = await detect_all_patterns_v2(
        df=df,
        symbol=symbol,
        time_frame=time_frame, filename=filename
    )

    # 2) Filtre uygula
    final_results = {}
    for pattern_name, pat_output in detection_results.items():
        # Liste mi dict mi bak
        if isinstance(pat_output, list):
            filtered_list = []
            for item in pat_output:
                passed = post_pattern_filters(pattern_name,
                    item, df, time_frame,
                    volume_check=True,
                    volume_multiplier=1.5,
                    atr_check=True,
                    min_atr_distance=1.0,
                    rsi_macd_check=True,
                    require_retest=True,
                    min_pattern_percent=0.05
                )
                if passed:
                    filtered_list.append(item)
            if filtered_list:
                final_results[pattern_name] = filtered_list
        elif isinstance(pat_output, dict):
            passed = post_pattern_filters(
                pat_output, df, time_frame,
                volume_check=True,
                volume_multiplier=1.5,
                atr_check=True,
                min_atr_distance=2.0,
                rsi_macd_check=True,
                require_retest=True,
                min_pattern_percent=0.02
            )
            if passed:
                final_results[pattern_name] = pat_output

    return final_results
