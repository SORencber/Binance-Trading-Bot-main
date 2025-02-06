import math
import os
import json
import warnings
import pandas as pd
import numpy as np
import concurrent.futures
from itertools import product
from datetime import datetime, date

try:
    import swisseph as swe
except ImportError:
    swe = None
    # Astro hesapları devre dışı bırakabilirsiniz
    # ya da pip install pyswisseph gibi bir paket kurabilirsiniz

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
        },
            "30m": {
            "left_bars": [8, 15, 20],
            "right_bars": [8, 15, 20],
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
            "left_bars": [15, 20],
            "right_bars": [15, 20],
            "min_distance_bars": [20, 40],
            "shoulder_tolerance": [0.02, 0.04],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [80, 100],
            "atr_filter": [0.3, 0.6],
            "check_rsi_macd": [True, False],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
            "left_bars": [20, 30],
            "right_bars": [20, 30],
            "min_distance_bars": [30, 50],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [100, 120],
            "atr_filter": [0.5, 1.0],
            "check_rsi_macd": [True, False],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
            "left_bars": [25, 35],
            "right_bars": [25, 35],
            "min_distance_bars": [30, 50],
            "shoulder_tolerance": [0.02, 0.06],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [120, 150],
            "atr_filter": [1.0, 1.5],
            "check_rsi_macd": [True, False],
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
            "left_bars": [8, 15, 20],
            "right_bars": [8, 15, 20],
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
            "left_bars": [15, 20],
            "right_bars": [15, 20],
            "min_distance_bars": [20, 40],
            "shoulder_tolerance": [0.02, 0.04],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [80, 100],
            "atr_filter": [0.3, 0.6],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
            "left_bars": [20, 30],
            "right_bars": [20, 30],
            "min_distance_bars": [30, 50],
            "shoulder_tolerance": [0.02, 0.05],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [100, 120],
            "atr_filter": [0.5, 1.0],
            "check_rsi_macd": [False, True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
            "left_bars": [25, 35],
            "right_bars": [25, 35],
            "min_distance_bars": [30, 50],
            "shoulder_tolerance": [0.02, 0.06],
            "volume_decline": [True],
            "neckline_break": [True],
            "max_shoulder_width_bars": [120, 150],
            "atr_filter": [1.0, 1.5],
            "check_rsi_macd": [True],
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
            "tolerance": [0.01, 0.02],
            "min_distance_bars": [40, 60],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
            "tolerance": [0.015, 0.025],
            "min_distance_bars": [60, 80],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [80, 100],
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
            "tolerance": [0.01, 0.02],
            "min_distance_bars": [40, 60],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "4h": {
            "tolerance": [0.015, 0.025],
            "min_distance_bars": [60, 80],
            "triple_variation": [True],
            "volume_check": [True],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.015,0.02]
        },
        "1d": {
            "tolerance": [0.02, 0.03],
            "min_distance_bars": [80, 100],
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
            "tolerance": [0.02, 0.04],
            "min_distance_bars": [30, 50],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "tolerance": [0.03, 0.05],
            "min_distance_bars": [40, 60],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "tolerance": [0.04, 0.06],
            "min_distance_bars": [50, 80],
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
            "tolerance": [0.02, 0.04],
            "min_distance_bars": [30, 50],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "tolerance": [0.03, 0.05],
            "min_distance_bars": [40, 60],
            "volume_check": [True],
            "volume_col_factor": [0.8],
            "neckline_break": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "tolerance": [0.04, 0.06],
            "min_distance_bars": [50, 80],
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
            "wave_min_bars": [20, 30],
            "fib_tolerance": [0.15, 0.25],
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
            "wave_min_bars": [25, 40],
            "fib_tolerance": [0.15, 0.3],
            "rule_3rdwave_min_percent": [1.618],
            "rule_3rdwave_not_shortest": [True],
            "allow_4th_overlap": [False],
            "allow_wave2_above_wave1_start": [False],
            "wave2_fib_range": [(0.382, 0.618), (0.382, 0.786)],
            "wave4_fib_range": [(0.382, 0.618), (0.382, 0.786)],
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
            "fib_tolerance": [0.03, 0.04],
            "patterns": [["gartley","bat","crab"], ["shark","cipher"]],
            "check_volume": [True],
            "volume_factor": [1.3, 1.5],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "15m": {
            "fib_tolerance": [0.03, 0.05],
            "patterns": [["butterfly","shark","cipher"], ["gartley","bat","crab"]],
            "check_volume": [True],
            "volume_factor": [1.3],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
         "30m": {
            "fib_tolerance": [0.03, 0.05],
            "patterns": [["butterfly","shark","cipher"], ["gartley","bat","crab"]],
            "check_volume": [True],
            "volume_factor": [1.3],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1h": {
            "fib_tolerance": [0.02, 0.03],
            "patterns": [["bat","butterfly"], ["crab","shark","cipher"]],
            "check_volume": [True],
            "volume_factor": [1.5],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "4h": {
            "fib_tolerance": [0.02, 0.03],
            "patterns": [["gartley","shark"], ["bat","cipher"]],
            "check_volume": [True],
            "volume_factor": [1.5, 1.8],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        },
        "1d": {
            "fib_tolerance": [0.02, 0.04],
            "patterns": [["gartley","bat","crab","butterfly","shark","cipher"]],
            "check_volume": [True],
            "volume_factor": [1.5, 2.0],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02]
        }
    },

    "triangle": {
        "1m": {
            "triangle_tolerance": [0.01, 0.02],
            "check_breakout": [True, False],
            "check_retest": [False],
            "retest_tolerance": [0.001,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "5m": {
            "triangle_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True, False],
            "retest_tolerance": [0.001,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "15m": {
            "triangle_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
          "30m": {
            "triangle_tolerance": [0.02, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "1h": {
            "triangle_tolerance": [0.015, 0.03],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "4h": {
            "triangle_tolerance": [0.015, 0.025],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02],
            "triangle_types": [["ascending","descending","symmetrical"]]
        },
        "1d": {
            "triangle_tolerance": [0.01, 0.02],
            "check_breakout": [True],
            "check_retest": [True],
            "retest_tolerance": [0.001,0.02],
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
    }
}


##############################################################################
# 3) HELPER FONKSİYONLAR (Line eq., RSI, MACD vb.)
##############################################################################

def line_equation(x1, y1, x2, y2):
    """
    Returns slope (m) and intercept (b) of the line y = m*x + b
    If x2 == x1 => returns (None, None)
    """
    if x2 == x1:
        return None, None
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m*x1
    return m, b

def line_intersection(m1, b1, m2, b2):
    """
    Intersection (x, y) of lines y=m1*x+b1 and y=m2*x+b2.
    If parallel or invalid => returns (None, None).
    """
    if (m1 is None) or (m2 is None):
        return None, None
    if abs(m1 - m2) < 1e-15:  # parallel
        return None, None
    x = (b2 - b1) / (m1 - m2)
    y = m1*x + b1
    return x, y

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


##############################################################################
# 6) FARKLI PATTERN ALGILAMA FONKSİYONLARI
##############################################################################
# Aşağıda teker teker HEAD-SHOULDERS, DOUBLE BOTTOM vb. ekleniyor.
# (Her fonksiyon, istenilen parametreleri kwargs olarak alıp "results" döndürüyor.)

############################
# HEAD & SHOULDERS ADV
############################

def detect_head_and_shoulders_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_distance_bars: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0,
    check_rsi_macd: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> list:
    """
    Gelişmiş Head & Shoulders algılama. 
    returns: list of dict (bulunan formasyonlar)
    """
    high_col   = get_col_name("High",  time_frame)
    low_col    = get_col_name("Low",   time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume",time_frame)
    atr_col    = get_col_name("ATR",   time_frame)

    # ATR Filter
    if atr_filter > 0:
        prepare_atr(df, time_frame)

    top_pivots = [p for p in pivots if p[2] == +1]
    results=[]
    for i in range(len(top_pivots)-2):
        L= top_pivots[i]
        H= top_pivots[i+1]
        R= top_pivots[i+2]
        idxL, priceL,_= L
        idxH, priceH,_= H
        idxR, priceR,_= R

        # Sıra kontrolü
        if not (idxL < idxH < idxR):
            continue
        # Head en büyük olmalı
        if not (priceH > priceL and priceH > priceR):
            continue

        bars_LH= idxH - idxL
        bars_HR= idxR - idxH
        if bars_LH < min_distance_bars or bars_HR < min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue

        # Omuz yükseklik farkı
        diffShoulder = abs(priceL - priceR)/(priceH+ 1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        # Volume check
        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            if volH > (mean_shoulder_vol*0.8):
                vol_check= False

        # Boyun çizgisi => L-H ve H-R arasındaki min low
        segment_LH= df[low_col].iloc[idxL: idxH+1]
        segment_HR= df[low_col].iloc[idxH: idxR+1]
        if len(segment_LH)<1 or len(segment_HR)<1:
            continue
        dip1_idx= segment_LH.idxmin()
        dip2_idx= segment_HR.idxmin()
        dip1_val= df[low_col].iloc[dip1_idx]
        dip2_val= df[low_col].iloc[dip2_idx]

        confirmed= False
        confirmed_bar= None
        if neckline_break:
            if dip1_idx != dip2_idx:
                m_, b_= line_equation(dip1_idx, dip1_val, dip2_idx, dip2_val)
                if m_ is not None:
                    for test_i in range(idxR, len(df)):
                        c = df[close_col].iloc[test_i]
                        line_y = m_* test_i + b_
                        if c < line_y:
                            confirmed= True
                            confirmed_bar = test_i
                            break

        # RSI-MACD onayı
        indicator_res = None
        if check_rsi_macd and confirmed and confirmed_bar is not None:
            indicator_res = indicator_checks(df, confirmed_bar, time_frame=time_frame,
                                             rsi_check=True, macd_check=True)

        # Retest
        retest_info = None
        if check_retest and confirmed and (confirmed_bar is not None):
            retest_info = check_retest_levels(
                df, time_frame,
                neckline_points=((dip1_idx,dip1_val),(dip2_idx,dip2_val)),
                break_bar=confirmed_bar,
                tolerance=retest_tolerance
            )

        results.append({
            "pattern": "head_and_shoulders",
            "L": (idxL, priceL),
            "H": (idxH, priceH),
            "R": (idxR, priceR),
            "shoulder_diff": diffShoulder,
            "volume_check": vol_check,
            "neckline": ((dip1_idx, dip1_val),(dip2_idx, dip2_val)),
            "confirmed": confirmed,
            "confirmed_bar": confirmed_bar,
            "indicator_check": indicator_res,
            "retest_info": retest_info
        })

    return results

def check_retest_levels(df: pd.DataFrame,
                        time_frame: str,
                        neckline_points: tuple,
                        break_bar: int,
                        tolerance: float = 0.02) -> dict:
    """
    Neckline retest kontrolü.
    """
    if not neckline_points:
        return {"retest_done": False, "retest_bar": None}

    (x1, y1), (x2, y2) = neckline_points
    m_, b_ = line_equation(x1, y1, x2, y2)
    if m_ is None:
        return {"retest_done": False, "retest_bar": None}

    close_col = get_col_name("Close", time_frame)
    retest_done = False
    retest_bar = None
    for i in range(break_bar+1, len(df)):
        c = df[close_col].iloc[i]
        line_y = m_*i + b_
        diff_perc = abs(c - line_y)/(abs(line_y)+1e-9)
        if diff_perc <= tolerance:
            retest_done = True
            retest_bar = i
            break
    return {
        "retest_done": retest_done,
        "retest_bar": retest_bar
    }

############################
# INVERSE HEAD & SHOULDERS ADV
############################
def detect_inverse_head_and_shoulders_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    left_bars: int = 10,
    right_bars: int = 10,
    min_distance_bars: int = 10,
    shoulder_tolerance: float = 0.03,
    volume_decline: bool = True,
    neckline_break: bool = True,
    max_shoulder_width_bars: int = 50,
    atr_filter: float = 0.0,
    check_rsi_macd: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> list:
    """
    Inverse (Ters) Head & Shoulders.
    """
    low_col   = get_col_name("Low", time_frame)
    close_col = get_col_name("Close", time_frame)
    volume_col= get_col_name("Volume", time_frame)

    if atr_filter>0:
        prepare_atr(df, time_frame)

    dip_pivots= [p for p in pivots if p[2]== -1]
    results=[]
    for i in range(len(dip_pivots)-2):
        L= dip_pivots[i]
        H= dip_pivots[i+1]
        R= dip_pivots[i+2]
        idxL, priceL,_= L
        idxH, priceH,_= H
        idxR, priceR,_= R

        if not (idxL< idxH< idxR):
            continue
        if not (priceH< priceL and priceH< priceR):
            continue

        bars_LH= idxH- idxL
        bars_HR= idxR- idxH
        if bars_LH< min_distance_bars or bars_HR< min_distance_bars:
            continue
        if bars_LH> max_shoulder_width_bars or bars_HR> max_shoulder_width_bars:
            continue

        diffShoulder= abs(priceL- priceR)/ ((priceL+priceR)/2 +1e-9)
        if diffShoulder> shoulder_tolerance:
            continue

        vol_check= True
        if volume_decline and volume_col in df.columns:
            volL= df[volume_col].iloc[idxL]
            volH= df[volume_col].iloc[idxH]
            volR= df[volume_col].iloc[idxR]
            mean_shoulder_vol= (volL+ volR)/2
            if volH > (mean_shoulder_vol*0.8):
                vol_check= False

        # Boyun => local max
        seg_LH= df[close_col].iloc[idxL: idxH+1]
        seg_HR= df[close_col].iloc[idxH: idxR+1]
        if len(seg_LH)<1 or len(seg_HR)<1:
            continue
        T1_idx= seg_LH.idxmax()
        T2_idx= seg_HR.idxmax()
        T1_val= df[close_col].iloc[T1_idx]
        T2_val= df[close_col].iloc[T2_idx]

        confirmed= False
        confirmed_bar= None
        if neckline_break:
            m_, b_= line_equation(T1_idx, T1_val, T2_idx, T2_val)
            if m_ is not None:
                for test_i in range(idxR, len(df)):
                    c= df[close_col].iloc[test_i]
                    line_y= m_* test_i + b_
                    if c> line_y:
                        confirmed= True
                        confirmed_bar= test_i
                        break

        indicator_res = None
        if check_rsi_macd and confirmed_bar is not None:
            indicator_res = indicator_checks(df, confirmed_bar, time_frame=time_frame,
                                             rsi_check=True, macd_check=True)

        retest_info=None
        if check_retest and confirmed_bar is not None and confirmed:
            retest_info = check_retest_levels(
                df, time_frame,
                neckline_points=((T1_idx,T1_val),(T2_idx,T2_val)),
                break_bar=confirmed_bar,
                tolerance=retest_tolerance
            )

        results.append({
          "pattern":"inverse_head_and_shoulders",
          "L": (idxL, priceL),
          "H": (idxH, priceH),
          "R": (idxR, priceR),
          "shoulder_diff": diffShoulder,
          "volume_check": vol_check,
          "confirmed": confirmed,
          "confirmed_bar": confirmed_bar,
          "neckline": ((T1_idx,T1_val),(T2_idx,T2_val)),
          "indicator_check": indicator_res,
          "retest_info": retest_info
        })
    return results


############################
# DOUBLE TOP / BOTTOM
############################

def detect_double_top(
    df: pd.DataFrame,
    pivots,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Double veya opsiyonel triple-top varyasyonu.
    """
    top_pivots= [p for p in pivots if p[2]== +1]
    if len(top_pivots)<2:
        return []

    volume_col= get_col_name("Volume", time_frame)
    close_col = get_col_name("Close", time_frame)

    results=[]
    i=0
    while i< len(top_pivots)-1:
        t1= top_pivots[i]
        t2= top_pivots[i+1]
        idx1,price1= t1[0], t1[1]
        idx2,price2= t2[0], t2[1]

        avgp= (price1+ price2)/2
        pdiff= abs(price1- price2)/(avgp+1e-9)
        bar_diff= idx2- idx1

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
              "pattern": "double_top",
              "tops": [(idx1,price1),(idx2,price2)],
              "neckline": None,
              "confirmed": False,
              "start_bar": idx1,
              "end_bar": idx2,
              "retest_info": None
            }
            used_third= False
            if triple_variation and (i+2< len(top_pivots)):
                t3= top_pivots[i+2]
                idx3,price3= t3[0], t3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found["tops"]= [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found["end_bar"]= idx3
                    found["pattern"]= "triple_top"
                    used_third= True

            if volume_check and volume_col in df.columns:
                vol1= df[volume_col].iloc[idx1]
                vol2= df[volume_col].iloc[idx2]
                if vol2 > (vol1*0.8):
                    i+= (2 if used_third else 1)
                    continue

            if neckline_break and close_col in df.columns:
                seg_end= found["end_bar"]
                dips_for_neck = [pp for pp in pivots if pp[2]== -1 and (pp[0]> idx1 and pp[0]< seg_end)]
                if dips_for_neck:
                    dips_sorted= sorted(dips_for_neck, key=lambda x: x[1])  # ascending price
                    neck = dips_sorted[0]
                    found["neckline"]= (neck[0], neck[1])
                    last_close= df[close_col].iloc[-1]
                    if last_close< neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1
                        if check_retest:
                            retest_info= _check_retest_doubletop(
                                df, time_frame,
                                neckline_price= neck[1],
                                confirm_bar= len(df)-1,
                                retest_tolerance= retest_tolerance
                            )
                            found["retest_info"]= retest_info

            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results

def _check_retest_doubletop(df: pd.DataFrame,
                            time_frame: str,
                            neckline_price: float,
                            confirm_bar: int,
                            retest_tolerance: float=0.01) -> dict:
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        return {"retest_done": False, "retest_bar": None}

    n= len(df)
    if confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}


def detect_double_bottom(
    df: pd.DataFrame,
    pivots,
    time_frame:str="1m",
    tolerance: float=0.01,
    min_distance_bars: int=20,
    triple_variation: bool=True,
    volume_check: bool=False,
    neckline_break: bool=False,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    """
    Double veya opsiyonel triple-bottom varyasyonu.
    """
    bottom_pivots= [p for p in pivots if p[2]== -1]
    if len(bottom_pivots)<2:
        return []

    volume_col= get_col_name("Volume", time_frame)
    close_col = get_col_name("Close", time_frame)

    results=[]
    i=0
    while i< len(bottom_pivots)-1:
        b1= bottom_pivots[i]
        b2= bottom_pivots[i+1]
        idx1,price1= b1[0], b1[1]
        idx2,price2= b2[0], b2[1]

        avgp= (price1+ price2)/2
        pdiff= abs(price1- price2)/(avgp+1e-9)
        bar_diff= idx2- idx1

        if pdiff< tolerance and bar_diff>= min_distance_bars:
            found= {
              "pattern": "double_bottom",
              "bottoms": [(idx1,price1),(idx2,price2)],
              "neckline": None,
              "confirmed": False,
              "start_bar": idx1,
              "end_bar": idx2,
              "retest_info": None
            }
            used_third= False
            if triple_variation and (i+2< len(bottom_pivots)):
                b3= bottom_pivots[i+2]
                idx3,price3= b3[0], b3[1]
                pdiff3= abs(price2- price3)/ ((price2+price3)/2+1e-9)
                bar_diff3= idx3- idx2
                if pdiff3< tolerance and bar_diff3>= min_distance_bars:
                    found["bottoms"]= [(idx1,price1),(idx2,price2),(idx3,price3)]
                    found["end_bar"]= idx3
                    found["pattern"]= "triple_bottom"
                    used_third= True

            if volume_check and volume_col in df.columns:
                vol1= df[volume_col].iloc[idx1]
                vol2= df[volume_col].iloc[idx2]
                if vol2> (vol1*0.8):
                    i+=(2 if used_third else 1)
                    continue

            if neckline_break and close_col in df.columns:
                seg_end= found["end_bar"]
                top_pivs= [pp for pp in pivots if pp[2]== +1 and pp[0]> idx1 and pp[0]< seg_end]
                if top_pivs:
                    top_sorted= sorted(top_pivs, key=lambda x: x[1], reverse=True)
                    neck= top_sorted[0]
                    found["neckline"]= (neck[0], neck[1])
                    last_close= df[close_col].iloc[-1]
                    if last_close> neck[1]:
                        found["confirmed"]= True
                        found["end_bar"]= len(df)-1
                        if check_retest:
                            retest_info= _check_retest_dblbottom(
                                df, time_frame,
                                neckline_price= neck[1],
                                confirm_bar=len(df)-1,
                                retest_tolerance=retest_tolerance
                            )
                            found["retest_info"] = retest_info
            results.append(found)
            i+= (2 if used_third else 1)
        else:
            i+=1
    return results

def _check_retest_dblbottom(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float=0.01
) -> dict:
    close_col= get_col_name("Close", time_frame)
    n= len(df)
    if close_col not in df.columns or confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}


############################
# TRIPLE TOP / BOTTOM (ADV)
############################

def detect_triple_top_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    tolerance: float = 0.01,
    min_distance_bars: int = 20,
    volume_check: bool = False,
    volume_col_factor: float = 0.8,
    neckline_break: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
) -> list:
    """
    Triple Top => 3 tepe pivot, birbirine yakın (tolerance).
    """
    top_pivots = [p for p in pivots if p[2] == +1]
    if len(top_pivots) < 3:
        return []

    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume", time_frame)
    results = []
    i = 0
    while i < len(top_pivots) - 2:
        t1 = top_pivots[i]
        t2 = top_pivots[i+1]
        t3 = top_pivots[i+2]

        idx1, price1 = t1[0], t1[1]
        idx2, price2 = t2[0], t2[1]
        idx3, price3 = t3[0], t3[1]

        bar_diff_12 = idx2 - idx1
        bar_diff_23 = idx3 - idx2
        if bar_diff_12 < min_distance_bars or bar_diff_23 < min_distance_bars:
            i+=1
            continue

        avgp = (price1 + price2 + price3)/3
        pdiff_1 = abs(price1 - avgp)/(avgp+1e-9)
        pdiff_2 = abs(price2 - avgp)/(avgp+1e-9)
        pdiff_3 = abs(price3 - avgp)/(avgp+1e-9)
        if any(p > tolerance for p in [pdiff_1, pdiff_2, pdiff_3]):
            i+=1
            continue

        vol_ok = True
        msgs = []
        if volume_check and volume_col in df.columns:
            vol1 = df[volume_col].iloc[idx1]
            vol2 = df[volume_col].iloc[idx2]
            vol3 = df[volume_col].iloc[idx3]
            mean_top_vol = (vol1 + vol2)/2
            if vol3 > (mean_top_vol* volume_col_factor):
                vol_ok = False
                msgs.append(f"3rd top volume not lower => vol3={vol3:.2f}, mean12={mean_top_vol:.2f}")

        # Neckline
        seg_min_pivots = [p for p in pivots if p[2] == -1 and p[0]> idx1 and p[0]< idx3]
        neckline = None
        if seg_min_pivots:
            sorted_dips = sorted(seg_min_pivots, key=lambda x: x[1])
            neckline = (sorted_dips[0][0], sorted_dips[0][1])
        else:
            msgs.append("No local dip pivot found for neckline.")

        conf=False
        retest_data=None
        if neckline_break and neckline is not None and close_col in df.columns:
            neck_idx, neck_prc= neckline
            last_close = df[close_col].iloc[-1]
            if last_close< neck_prc:
                conf = True
                if check_retest:
                    retest_data= _check_retest_triple_top(
                        df, time_frame,
                        neckline_price= neck_prc,
                        confirm_bar= len(df)-1,
                        retest_tolerance= retest_tolerance
                    )
            else:
                msgs.append("Neckline not broken => not confirmed")

        pattern_info = {
            "pattern": "triple_top",
            "tops": [(idx1,price1),(idx2,price2),(idx3,price3)],
            "neckline": neckline,
            "confirmed": conf,
            "volume_check": vol_ok,
            "msgs": msgs,
            "retest_info": retest_data
        }
        if vol_ok:
            results.append(pattern_info)

        i+=1

    return results

def _check_retest_triple_top(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float = 0.02
):
    close_col = get_col_name("Close", time_frame)
    n= len(df)
    if close_col not in df.columns or confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}

    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c - neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}


def detect_triple_bottom_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    tolerance: float = 0.01,
    min_distance_bars: int = 20,
    volume_check: bool = False,
    volume_col_factor: float = 0.8,
    neckline_break: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
)-> list:
    bot_pivots= [p for p in pivots if p[2]== -1]
    if len(bot_pivots)< 3:
        return []

    close_col= get_col_name("Close", time_frame)
    volume_col= get_col_name("Volume", time_frame)
    results=[]
    i=0
    while i< len(bot_pivots)-2:
        b1= bot_pivots[i]
        b2= bot_pivots[i+1]
        b3= bot_pivots[i+2]

        idx1,price1= b1[0], b1[1]
        idx2,price2= b2[0], b2[1]
        idx3,price3= b3[0], b3[1]

        bar_diff_12= idx2- idx1
        bar_diff_23= idx3- idx2
        if bar_diff_12< min_distance_bars or bar_diff_23< min_distance_bars:
            i+=1
            continue

        avgp= (price1+ price2+ price3)/3
        pdiff_1= abs(price1- avgp)/(avgp+1e-9)
        pdiff_2= abs(price2- avgp)/(avgp+1e-9)
        pdiff_3= abs(price3- avgp)/(avgp+1e-9)
        if any(p> tolerance for p in [pdiff_1, pdiff_2, pdiff_3]):
            i+=1
            continue

        vol_ok= True
        msgs=[]
        if volume_check and volume_col in df.columns:
            vol1= df[volume_col].iloc[idx1]
            vol2= df[volume_col].iloc[idx2]
            vol3= df[volume_col].iloc[idx3]
            mean_bot_vol= (vol1+vol2)/2
            if vol3> (mean_bot_vol* volume_col_factor):
                vol_ok= False
                msgs.append(f"3rd bottom volume not lower => vol3={vol3:.2f}")

        seg_max_pivots= [p for p in pivots if p[2]== +1 and p[0]> idx1 and p[0]< idx3]
        neckline= None
        if seg_max_pivots:
            sorted_tops= sorted(seg_max_pivots, key=lambda x: x[1], reverse=True)
            neckline= (sorted_tops[0][0], sorted_tops[0][1])
        else:
            msgs.append("No local top pivot found for neckline.")

        conf=False
        retest_data=None
        if neckline_break and neckline and close_col in df.columns:
            neck_idx, neck_prc= neckline
            last_close= df[close_col].iloc[-1]
            if last_close> neck_prc:
                conf=True
                if check_retest:
                    retest_data= _check_retest_triple_bottom(
                        df, time_frame,
                        neckline_price= neck_prc,
                        confirm_bar= len(df)-1,
                        retest_tolerance= retest_tolerance
                    )
            else:
                msgs.append("Neckline not broken => not confirmed")

        pattern_info={
            "pattern":"triple_bottom",
            "bottoms":[(idx1,price1),(idx2,price2),(idx3,price3)],
            "neckline": neckline,
            "confirmed": conf,
            "volume_check": vol_ok,
            "msgs": msgs,
            "retest_info": retest_data
        }
        if vol_ok:
            results.append(pattern_info)
        i+=1
    return results

def _check_retest_triple_bottom(
    df: pd.DataFrame,
    time_frame: str,
    neckline_price: float,
    confirm_bar: int,
    retest_tolerance: float= 0.01
):
    close_col= get_col_name("Close", time_frame)
    n= len(df)
    if close_col not in df.columns or confirm_bar>= n-1:
        return {"retest_done": False, "retest_bar": None}
    for i in range(confirm_bar+1, n):
        c= df[close_col].iloc[i]
        dist_ratio= abs(c- neckline_price)/(abs(neckline_price)+1e-9)
        if dist_ratio<= retest_tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "retest_price": c,
                "dist_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None}


############################
# ZIGZAG HELPER
############################

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


############################
# WOLFE WAVE ADV
############################

def detect_wolfe_wave_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    price_tolerance: float = 0.03,
    strict_lines: bool = False,
    breakout_confirm: bool = True,
    line_projection_check: bool = True,
    check_2_4_slope: bool = True,
    check_1_4_intersection_time: bool = True,
    check_time_symmetry: bool = True,
    max_time_ratio: float = 0.3,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
):
    result= {
      "pattern": "wolfe",
      "found": False,
      "msgs": [],
      "breakout": False,
      "intersection": None,
      "time_symmetry_ok": True,
      "sweet_zone": None,
      "wolfe_line": None,
      "retest_info": None
    }
    wave = build_zigzag_wave(pivots)
    if len(wave)<5:
        result["msgs"].append("Not enough pivots (need 5).")
        return result
    w1= wave[-5]
    w2= wave[-4]
    w3= wave[-3]
    w4= wave[-2]
    w5= wave[-1]
    x1,y1,_= w1
    x2,y2,_= w2
    x3,y3,_= w3
    x4,y4,_= w4
    x5,y5,_= w5

    m13,b13= line_equation(x1,y1, x3,y3)
    m35,b35= line_equation(x3,y3, x5,y5)
    if (m13 is None) or (m35 is None):
        result["msgs"].append("Line(1->3) or (3->5) vertical => fail.")
        return result
    diff_slope= abs(m35- m13)/(abs(m13)+1e-9)
    if diff_slope> price_tolerance:
        result["msgs"].append(f"Slope difference too big => {diff_slope:.3f}")

    if check_2_4_slope:
        m24,b24= line_equation(x2,y2, x4,y4)
        if strict_lines and (m24 is not None):
            slope_diff= abs(m24- m13)/(abs(m13)+1e-9)
            if slope_diff>0.3:
                result["msgs"].append("Line(2->4) slope differs from line(1->3).")

    # sweet zone
    m24_,b24_= line_equation(x2,y2, x4,y4)
    if m24_ is not None:
        line13_y5= m13*x5+ b13
        line24_y5= m24_*x5+ b24_
        low_  = min(line13_y5, line24_y5)
        high_ = max(line13_y5, line24_y5)
        result["sweet_zone"]= (low_, high_)
        if not (low_<= y5<= high_):
            result["msgs"].append("W5 not in sweet zone")

    if check_time_symmetry:
        bars_23= x3- x2
        bars_34= x4- x3
        bars_45= x5- x4
        def ratio(a,b): return abs(a-b)/(abs(b)+1e-9)
        r1= ratio(bars_23,bars_34)
        r2= ratio(bars_34,bars_45)
        if (r1> max_time_ratio) or (r2> max_time_ratio):
            result["time_symmetry_ok"]= False
            result["msgs"].append(f"Time symmetry fail => r1={r1:.2f}, r2={r2:.2f}")

    if line_projection_check:
        m14,b14= line_equation(x1,y1,x4,y4)
        m23,b23= line_equation(x2,y2,x3,y3)
        if (m14 is not None) and (m23 is not None):
            ix,iy= line_intersection(m14,b14, m23,b23)
            if ix is not None:
                result["intersection"]= (ix, iy)
                if check_1_4_intersection_time and ix< x5:
                    result["msgs"].append("Intersection(1->4 & 2->3) < w5 => degrade")

    if breakout_confirm:
        close_col= get_col_name("Close", time_frame)
        if close_col in df.columns:
            last_close= df[close_col].iloc[-1]
            m14,b14= line_equation(x1,y1, x4,y4)  # (1->4) hattı
           
            if m14 is not None:
                last_i= len(df)-1
                line_y= m14* last_i + b14
                #print(f"Bearish check => last_close={last_close:.4f}, line_y={line_y:.4f}, slope={m14:.4f}, x1={x1}, x4={x4}")

                if m14 > 0:
                    # Bullish Wolfe => Üst tarafa kırılırsa breakout
                    if last_close > line_y:
                        result["breakout"] = True
                        result["wolfe_line"] = ((x1,y1),(x4,y4))
                        result["w5"] = (x5,y5)
                        result["direction"] = "LONG"  # <-- Ekleyebilirsiniz
                else:
                    # Bearish Wolfe => Alt tarafa kırılırsa breakout
                    if last_close < line_y:

                        result["breakout"] = True
                        result["wolfe_line"] = ((x1,y1),(x4,y4))
                        result["w5"] = (x5,y5)
                        result["direction"] = "SHORT"  # <-- Ekleyebilirsiniz

    result["found"]= True


    if check_retest and result["breakout"] and result["wolfe_line"]:
        (ixA,pxA),(ixB,pxB)= result["wolfe_line"]
        m_,b_= line_equation(ixA, pxA, ixB, pxB)
        if m_ is not None and df is not None:
            close_col= get_col_name("Close", time_frame)
            last_i= len(df)-1
            retest_done= False
            retest_bar= None
            for i in range(last_i+1, len(df)):
                c= df[close_col].iloc[i]
                line_val= m_* i + b_
                dist_perc= abs(c - line_val)/(abs(line_val)+1e-9)
                if dist_perc<= retest_tolerance:
                    retest_done= True
                    retest_bar= i
                    break
            result["retest_info"]= {
                "retest_done": retest_done,
                "retest_bar": retest_bar
            }

    return result

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
# HARMONIC PATTERNS
############################

def detect_harmonic_pattern_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str = "1m",
    fib_tolerance: float=0.02,
    patterns: list = None,
    check_volume: bool=False,
    volume_factor: float=1.3,
    check_retest: bool= False,
    retest_tolerance: float=0.01
)-> dict:
    """
    Harmonic Pattern (X,A,B,C,D).
    """
    if patterns is None:
        patterns= ["gartley","bat","crab","butterfly","shark","cipher"]
    result= {
      "pattern": "harmonic",
      "found": False,
      "pattern_name": None,
      "xabc": [],
      "msgs": [],
      "retest_info": None
    }
    wave= build_zigzag_wave(pivots)
    if len(wave)<5:
        result["msgs"].append("Not enough pivot for harmonic (need 5).")
        return result

    X= wave[-5]
    A= wave[-4]
    B= wave[-3]
    C= wave[-2]
    D= wave[-1]
    idxX, pxX,_= X
    idxA, pxA,_= A
    idxB, pxB,_= B
    idxC, pxC,_= C
    idxD, pxD,_= D
    result["xabc"]=[X,A,B,C,D]

    def length(a,b): return abs(b-a)
    XA= length(pxX, pxA)
    AB= length(pxA, pxB)
    BC= length(pxB, pxC)
    CD= length(pxC, pxD)

    harmonic_map= {
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
            "AB_XA": (0.886,1.13),
            "BC_AB": (1.13, 1.618),
            "CD_BC": (0.886,1.13)
        },
        "cipher": {
            "AB_XA": (0.382,0.618),
            "BC_AB": (1.27,2.0),
            "CD_BC": (1.13,1.414)
        }
    }
    def in_range(val, rng, tol):
        mn,mx= rng
        if abs(mn-mx)<1e-9:
            return abs(val- mn)<= abs(mn)*tol
        else:
            low_= mn- abs(mn)* tol
            high_= mx+ abs(mx)* tol
            return low_<= val<= high_

    AB_XA= AB/(XA+1e-9)
    BC_AB= BC/(AB+1e-9)
    CD_BC= CD/(BC+1e-9)

    found_any= False
    matched_pattern= None
    for pat in patterns:
        if pat not in harmonic_map:
            continue
        spec= harmonic_map[pat]
        rngAB_XA= spec["AB_XA"]
        rngBC_AB= spec["BC_AB"]
        rngCD_BC= spec["CD_BC"]

        ok1= in_range(AB_XA, rngAB_XA, fib_tolerance)
        ok2= in_range(BC_AB, rngBC_AB, fib_tolerance)
        ok3= in_range(CD_BC, rngCD_BC, fib_tolerance)
        if ok1 and ok2 and ok3:
            found_any= True
            matched_pattern= pat
            break
    if found_any:
        result["found"]= True
        result["pattern_name"]= matched_pattern

        volume_col= get_col_name("Volume", time_frame)
        if check_volume and volume_col in df.columns and idxD<len(df):
            vol_now= df[volume_col].iloc[idxD]
            prepare_volume_ma(df, time_frame, period=20)
            ma_col= f"Volume_MA_20_{time_frame}"
            if ma_col in df.columns:
                v_mean= df[ma_col].iloc[idxD]
                if (v_mean>0) and (vol_now> volume_factor*v_mean):
                    # "Güçlü hacim"
                    pass

        if check_retest:
            close_col= get_col_name("Close", time_frame)
            if close_col in df.columns:
                retest_done= False
                retest_bar = None
                for i in range(idxD+1, len(df)):
                    c= df[close_col].iloc[i]
                    dist_ratio = abs(c - pxD)/(abs(pxD)+1e-9)
                    if dist_ratio <= retest_tolerance:
                        retest_done= True
                        retest_bar= i
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


############################
# TRIANGLE (Asc, Desc, Sym)
############################
def detect_triangle_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str="1m",
    triangle_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01,
    triangle_types: list=None
):
    result={
      "pattern": "triangle",
      "found": False,
      "triangle_type": None,
      "breakout": False,
      "breakout_line": None,
      "retest_info": None,
      "msgs": []
    }
    wave= build_zigzag_wave(pivots)
    if triangle_types is None:
        triangle_types=["ascending","descending","symmetrical"]
    if len(wave)<4:
        result["msgs"].append("Not enough pivots for triangle (need >=4).")
        return result

    last4= wave[-4:]
    p1,p2,p3,p4= last4
    t_list=[p[2] for p in last4]
    up_zig= [+1,-1,+1,-1]
    down_zig= [-1,+1,-1,+1]
    if t_list not in [up_zig, down_zig]:
        result["msgs"].append("Zigzag pattern not matching triangle requirement.")
        return result

    if t_list==up_zig:
        x1,y1=p1[0],p1[1]
        x3,y3=p3[0],p3[1]
        x2,y2=p2[0],p2[1]
        x4,y4=p4[0],p4[1]
    else:
        # Ters
        x1,y1=p2[0],p2[1]
        x3,y3=p4[0],p4[1]
        x2,y2=p1[0],p1[1]
        x4,y4=p3[0],p3[1]

    m_top,b_top=line_equation(x1,y1,x3,y3)
    m_bot,b_bot=line_equation(x2,y2,x4,y4)
    if m_top is None or m_bot is None:
        result["msgs"].append("Line top/bot eq fail => vertical slope.")
        return result

    def is_flat(m):
        return abs(m)< triangle_tolerance

    top_type= None
    bot_type= None
    if is_flat(m_top):
        top_type="flat"
    elif m_top>0:
        top_type="rising"
    else:
        top_type="falling"

    if is_flat(m_bot):
        bot_type="flat"
    elif m_bot>0:
        bot_type="rising"
    else:
        bot_type="falling"

    tri_type=None
    if top_type=="flat" and bot_type=="rising" and ("ascending" in triangle_types):
        tri_type="ascending"
    elif top_type=="falling" and bot_type=="flat" and ("descending" in triangle_types):
        tri_type="descending"
    elif top_type=="falling" and bot_type=="rising" and ("symmetrical" in triangle_types):
        tri_type="symmetrical"

    if not tri_type:
        result["msgs"].append("No matching triangle type.")
        return result

    result["found"]=True
    result["triangle_type"]= tri_type

    breakout=False
    close_col= get_col_name("Close", time_frame)
    if check_breakout and close_col in df.columns:
        last_close= df[close_col].iloc[-1]
        last_i= len(df)-1
        line_y_top= m_top* last_i + b_top
        line_y_bot= m_bot* last_i + b_bot
        if tri_type=="ascending":
            if last_close> line_y_top:
                breakout= True
                result["breakout_line"]= ((x1,y1),(x3,y3))
        elif tri_type=="descending":
            if last_close< line_y_bot:
                breakout= True
                result["breakout_line"]= ((x2,y2),(x4,y4))
        else:
            if (last_close> line_y_top) or (last_close< line_y_bot):
                breakout= True
                result["breakout_line"]= ((x1,y1),(x3,y3))
    result["breakout"]= breakout

    if check_retest and breakout and result["breakout_line"]:
        (xA,pA),(xB,pB)= result["breakout_line"]
        m_,b_= line_equation(xA,pA,xB,pB)
        if m_ is not None:
            retest_done=False
            retest_bar=None
            for i in range(xB+1, len(df)):
                c= df[close_col].iloc[i]
                line_y= m_*i + b_
                diff_perc= abs(c - line_y)/(abs(line_y)+1e-9)
                if diff_perc<= retest_tolerance:
                    retest_done=True
                    retest_bar=i
                    break
            result["retest_info"]={
                "retest_done": retest_done,
                "retest_bar": retest_bar
            }
    return result


############################
# WEDGE (Rising, Falling)
############################

def detect_wedge_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame:str="1m",
    wedge_tolerance: float=0.02,
    check_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
)-> dict:
    result={
      "pattern":"wedge",
      "found":False,
      "wedge_type":None,
      "breakout":False,
      "breakout_line":None,
      "retest_info":None,
      "msgs":[]
    }
    wave= build_zigzag_wave(pivots)
    if len(wave)<5:
        result["msgs"].append("Not enough pivot for wedge (need>=5).")
        return result

    last5= wave[-5:]
    types=[p[2] for p in last5]
    rising_pat= [+1,-1,+1,-1,+1]
    falling_pat=[-1,+1,-1,+1,-1]
    if types==rising_pat:
        wedge_type="rising"
    elif types==falling_pat:
        wedge_type="falling"
    else:
        result["msgs"].append("Pivot pattern not matching rising/falling wedge.")
        return result

    x1,y1= last5[0][0], last5[0][1]
    x3,y3= last5[2][0], last5[2][1]
    x5,y5= last5[4][0], last5[4][1]
    slope_top= (y5-y1)/((x5-x1)+1e-9)

    x2,y2= last5[1][0], last5[1][1]
    x4,y4= last5[3][0], last5[3][1]
    slope_bot= (y4-y2)/((x4-x2)+1e-9)

    if wedge_type=="rising":
        if (slope_top<0) or (slope_bot<0):
            result["msgs"].append("Expected positive slopes for rising wedge.")
            return result
        if not (slope_bot> slope_top):
            result["msgs"].append("slope(2->4)<= slope(1->3)? => not wedge shape.")
            return result
    else:
        if (slope_top>0) or (slope_bot>0):
            result["msgs"].append("Expected negative slopes for falling wedge.")
            return result
        if not (slope_bot> slope_top):
            result["msgs"].append("Dip slope <= top slope => not wedge shape.")
            return result

    ratio= abs(slope_bot- slope_top)/(abs(slope_top)+1e-9)
    if ratio< wedge_tolerance:
        result["msgs"].append(f"Wedge slope difference ratio {ratio:.3f}< tolerance => might be channel.")

    df_len= len(df)
    brk=False
    close_col= get_col_name("Close", time_frame)
    if check_breakout and close_col in df.columns and df_len>0:
        last_close= df[close_col].iloc[-1]
        m_,b_= line_equation(x2,y2,x4,y4)  # alt çizgi
        if wedge_type=="rising":
            if m_ is not None:
                last_i= df_len-1
                line_y= m_* last_i + b_
                if last_close< line_y:
                    brk= True
        else:
            # falling => üst çizgi
            m2,b2= line_equation(x1,y1,x3,y3)
            if m2 is not None:
                last_i= df_len-1
                line_y2= m2* last_i+ b2
                if last_close> line_y2:
                    brk= True

    if brk:
        result["breakout"]=True
        if wedge_type=="rising":
            result["breakout_line"]= ((x2,y2),(x4,y4))
        else:
            result["breakout_line"]= ((x1,y1),(x3,y3))

    result["found"]= True
    result["wedge_type"]= wedge_type

    if check_retest and brk and result["breakout_line"]:
        (ixA,pxA),(ixB,pxB)= result["breakout_line"]
        mW,bW= line_equation(ixA,pxA,ixB,pxB)
        if mW is not None:
            retest_done=False
            retest_bar=None
            for i in range(ixB+1, df_len):
                c= df[close_col].iloc[i]
                line_y= mW*i + bW
                diff_perc= abs(c-line_y)/(abs(line_y)+1e-9)
                if diff_perc<= retest_tolerance:
                    retest_done= True
                    retest_bar= i
                    break
            result["retest_info"]= {
                "retest_done": retest_done,
                "retest_bar": retest_bar
            }

    return result


############################
# CUP & HANDLE (ADV)
############################

def detect_cup_and_handle_advanced(
    df: pd.DataFrame,
    pivots=None,
    time_frame: str="1m",
    tolerance: float=0.02,
    volume_drop_check: bool=True,
    volume_drop_ratio: float=0.2,
    cup_min_bars: int=20,
    cup_max_bars: int=300,
    handle_ratio: float=0.3,
    handle_max_bars: int=50,
    close_above_rim: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
):
    result = {
        "pattern": "cup_handle",
        "found": False,
        "cup_left_top": None,
        "cup_bottom": None,
        "cup_right_top": None,
        "cup_bars": 0,
        "cup_volume_drop": None,
        "handle_found": False,
        "handle_top": None,
        "handle_bars": 0,
        "confirmed": False,
        "rim_line": None,
        "msgs": [],
        "retest_info": None
    }
    close_col= get_col_name("Close", time_frame)
    volume_col= get_col_name("Volume", time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"Missing col: {close_col}")
        return result

    if pivots is None:
        # PivotScanner vs. => opsiyonel
        pass

    top_pivots= [p for p in pivots if p[2]== +1]
    bot_pivots= [p for p in pivots if p[2]== -1]
    if len(top_pivots)<2 or len(bot_pivots)<1:
        result["msgs"].append("Not enough top/dip pivots for Cup&Handle.")
        return result

    sorted_p= sorted(pivots, key=lambda x: x[0])
    best_cup= None
    for i in range(1, len(sorted_p)-1):
        if sorted_p[i][2]== -1:  # dip
            idxDip, pxDip= sorted_p[i][0], sorted_p[i][1]
            left_candidates= [tp for tp in sorted_p[:i] if tp[2]== +1]
            right_candidates= [tp for tp in sorted_p[i+1:] if tp[2]== +1]
            if (not left_candidates) or (not right_candidates):
                continue
            left_top= left_candidates[-1]
            right_top= right_candidates[0]
            bars_cup= right_top[0]- left_top[0]
            if bars_cup< cup_min_bars or bars_cup> cup_max_bars:
                continue

            avg_top= (left_top[1]+ right_top[1])/2
            top_diff= abs(left_top[1]- right_top[1])/(avg_top+1e-9)
            if top_diff> tolerance:
                continue
            if pxDip> avg_top:
                continue
            best_cup= (left_top, (idxDip,pxDip), right_top, bars_cup)
            break

    if not best_cup:
        result["msgs"].append("No valid cup found.")
        return result

    l_top, cup_dip, r_top, cup_bars= best_cup
    result["found"]= True
    result["cup_left_top"]= l_top
    result["cup_bottom"]= cup_dip
    result["cup_right_top"]= r_top
    result["cup_bars"]= cup_bars

    if volume_drop_check and volume_col in df.columns:
        idxL, pxL= l_top[0], l_top[1]
        idxR, pxR= r_top[0], r_top[1]
        cup_vol_series= df[volume_col].iloc[idxL : idxR+1]
        if len(cup_vol_series)>5:
            start_vol= cup_vol_series.iloc[0]
            min_vol= cup_vol_series.min()
            drop_percent= (start_vol- min_vol)/(start_vol+1e-9)
            result["cup_volume_drop"]= drop_percent
            if drop_percent< volume_drop_ratio:
                result["msgs"].append(f"Cup volume drop {drop_percent:.2f} < {volume_drop_ratio:.2f}")

    rim_idxL, rim_pxL= l_top[0], l_top[1]
    rim_idxR, rim_pxR= r_top[0], r_top[1]
    slope_rim= (rim_pxR- rim_pxL)/(rim_idxR- rim_idxL+1e-9)
    intercept= rim_pxL - slope_rim* rim_idxL

    dip_price= cup_dip[1]
    cup_height= ((l_top[1] + r_top[1])/2) - dip_price
    if cup_height<=0:
        return result

    handle_start= rim_idxR
    handle_end= min(rim_idxR+ handle_max_bars, len(df)-1)
    handle_found= False
    handle_top= None
    handle_bars= 0

    if handle_start< handle_end:
        seg= df[close_col].iloc[handle_start: handle_end+1]
        loc_max_val= seg.max()
        loc_max_idx= seg.idxmax()
        handle_bars= handle_end- handle_start
        handle_depth= ((r_top[1]+ l_top[1])/2)- loc_max_val
        if handle_depth>0:
            ratio= handle_depth/cup_height
            if ratio<= handle_ratio:
                handle_found= True
                handle_top= (loc_max_idx, loc_max_val)

    result["handle_found"]= handle_found
    result["handle_top"]= handle_top
    result["handle_bars"]= handle_bars

    # Cup&Handle onayı => rim break
    last_price= df[close_col].iloc[-1]
    last_i= len(df)-1
    rim_line_val= slope_rim* last_i + intercept
    if close_above_rim:
        if last_price> rim_line_val:
            result["confirmed"]= True
            result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))
    else:
        high_col= get_col_name("High", time_frame)
        if high_col in df.columns:
            last_high= df[high_col].iloc[-1]
            if last_high> rim_line_val:
                result["confirmed"]= True
                result["rim_line"]= ((rim_idxL, rim_pxL), (rim_idxR, rim_pxR))

    # Retest
    if check_retest and result["confirmed"] and result["rim_line"]:
        retest_info= _check_retest_cup_handle(
            df, time_frame,
            rim_line= result["rim_line"],
            break_bar= last_i,
            tolerance= retest_tolerance
        )
        result["retest_info"]= retest_info

    return result

def _check_retest_cup_handle(
    df: pd.DataFrame,
    time_frame: str,
    rim_line: tuple,
    break_bar: int,
    tolerance: float=0.01
):
    (xL, pL), (xR, pR)= rim_line
    m,b= line_equation(xL, pL, xR, pR)
    if m is None:
        return {"retest_done": False, "retest_bar": None, "distance_ratio": None}

    close_col= get_col_name("Close", time_frame)
    for i in range(break_bar+1, len(df)):
        c= df[close_col].iloc[i]
        line_y= m*i + b
        dist_ratio= abs(c- line_y)/(abs(line_y)+1e-9)
        if dist_ratio<= tolerance:
            return {
                "retest_done": True,
                "retest_bar": i,
                "distance_ratio": dist_ratio
            }
    return {"retest_done": False, "retest_bar": None, "distance_ratio": None}


############################
# FLAG / PENNANT
############################

def detect_flag_pennant_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str="1m",
    min_flagpole_bars: int=15,
    impulse_pct: float=0.05,
    max_cons_bars: int=40,
    pivot_channel_tolerance: float=0.02,
    pivot_triangle_tolerance: float=0.02,
    require_breakout: bool=True,
    check_retest: bool=False,
    retest_tolerance: float=0.01
) -> dict:
    result={
        "pattern":"flag_pennant",
        "found":False,
        "direction":None,
        "pattern_type":None,
        "consolidation_pivots":[],
        "upper_line":None,
        "lower_line":None,
        "confirmed":False,
        "breakout_bar":None,
        "breakout_line":None,
        "retest_info":None,
        "msgs":[]
    }
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        result["msgs"].append(f"Missing {close_col}")
        return result
    n=len(df)
    if n< min_flagpole_bars:
        result["msgs"].append("Not enough bars for flagpole check.")
        return result

    start_i= n- min_flagpole_bars
    price_start= df[close_col].iloc[start_i]
    price_end= df[close_col].iloc[-1]
    pct_chg= (price_end- price_start)/(price_start+1e-9)
    if abs(pct_chg)< impulse_pct:
        result["msgs"].append(f"No strong impulse (< {impulse_pct*100}%).")
        return result

    direction= "bull" if (pct_chg>0) else "bear"
    result["direction"]= direction

    cons_start= n - min_flagpole_bars
    cons_end= min(n-1, cons_start+ max_cons_bars)
    if cons_end<= cons_start:
        result["msgs"].append("Consolidation not enough bars.")
        return result

    cons_piv= [p for p in pivots if (p[0]>= cons_start and p[0]<= cons_end)]
    result["consolidation_pivots"]= cons_piv

    top_pivs= [p for p in cons_piv if p[2]==+1]
    bot_pivs= [p for p in cons_piv if p[2]==-1]
    if len(top_pivs)<2 or len(bot_pivs)<2:
        result["msgs"].append("Not enough top/bottom pivots => can't form mini-channel or triangle.")
        return result

    top_sorted= sorted(top_pivs, key=lambda x: x[0])
    bot_sorted= sorted(bot_pivs, key=lambda x: x[0])
    up1,up2= top_sorted[0], top_sorted[1]
    dn1,dn2= bot_sorted[0], bot_sorted[1]

    def slope(x1,y1,x2,y2):
        if (x2-x1)==0: return None
        return (y2-y1)/(x2-x1)
    s_up= slope(up1[0], up1[1], up2[0], up2[1])
    s_dn= slope(dn1[0], dn1[1], dn2[0], dn2[1])
    if (s_up is None) or (s_dn is None):
        result["msgs"].append("Channel lines vertical => cannot form slope.")
        return result

    slope_diff= abs(s_up- s_dn)/(abs(s_up)+1e-9)
    is_parallel= (slope_diff< pivot_channel_tolerance)
    is_opposite_sign= (s_up* s_dn< 0)

    upper_line= ((up1[0], up1[1]),(up2[0], up2[1]))
    lower_line= ((dn1[0], dn1[1]),(dn2[0], dn2[1]))
    result["upper_line"]= upper_line
    result["lower_line"]= lower_line

    pattern_type=None
    if is_parallel:
        pattern_type= "flag"
    elif is_opposite_sign and slope_diff> pivot_triangle_tolerance:
        pattern_type= "pennant"

    if not pattern_type:
        result["msgs"].append("No definitive mini-flag or mini-pennant.")
        return result

    result["pattern_type"]= pattern_type
    result["found"]= True

    if not require_breakout:
        return result

    last_i= n-1
    last_close= df[close_col].iloc[-1]
    def line_val(p1,p2,x):
        if (p2[0]- p1[0])==0:
            return p1[1]
        m= (p2[1]- p1[1])/(p2[0]- p1[0])
        b= p1[1] - m*p1[0]
        return m*x+ b

    up_line_last= line_val(up1, up2, last_i)
    dn_line_last= line_val(dn1, dn2, last_i)
    conf= False
    brk_bar= None
    if direction=="bull":
        if last_close> up_line_last:
            conf= True
            brk_bar= last_i
    else:
        if last_close< dn_line_last:
            conf= True
            brk_bar= last_i

    result["confirmed"]= conf
    result["breakout_bar"]= brk_bar
    if conf:
        if direction=="bull":
            result["breakout_line"]= upper_line
        else:
            result["breakout_line"]= lower_line
        if check_retest and result["breakout_line"]:
            (ixA,pxA),(ixB,pxB)= result["breakout_line"]
            mF,bF= line_equation(ixA,pxA,ixB,pxB)
            if mF is not None:
                retest_done=False
                retest_bar=None
                for i in range(ixB+1, n):
                    c= df[close_col].iloc[i]
                    line_y= mF*i+ bF
                    diff_perc= abs(c- line_y)/(abs(line_y)+1e-9)
                    if diff_perc<= retest_tolerance:
                        retest_done=True
                        retest_bar= i
                        break
                result["retest_info"]={
                    "retest_done": retest_done,
                    "retest_bar": retest_bar
                }
    return result


############################
# CHANNEL (Advanced)
############################

def detect_channel_advanced(
    df: pd.DataFrame,
    pivots,
    time_frame: str="1m",
    parallel_thresh: float=0.02,
    min_top_pivots: int=3,
    min_bot_pivots: int=3,
    max_iter: int=10,
    check_retest: bool=False,
    retest_tolerance: float=0.01
)-> dict:
    import numpy as np
    result={
        "pattern":"channel",
        "found":False,
        "channel_type":None,
        "upper_line_points":[],
        "lower_line_points":[],
        "upper_line_eq":None,
        "lower_line_eq":None,
        "breakout":False,
        "breakout_line":None,
        "retest_info":None,
        "msgs":[]
    }
    close_col= get_col_name("Close", time_frame)
    if close_col not in df.columns:
        result["msgs"].append("No close col found.")
        return result
    if not pivots or len(pivots)==0:
        result["msgs"].append("No pivots given.")
        return result

    top_piv= [p for p in pivots if p[2]== +1]
    bot_piv= [p for p in pivots if p[2]== -1]
    if len(top_piv)< min_top_pivots or len(bot_piv)< min_bot_pivots:
        result["msgs"].append("Not enough top/bottom pivots.")
        return result

    def best_fit_line(pivot_list):
        xs= np.array([p[0] for p in pivot_list], dtype=float)
        ys= np.array([p[1] for p in pivot_list], dtype=float)
        if len(xs)<2:
            return (0.0, float(ys.mean()))
        m= (np.mean(xs*ys)- np.mean(xs)* np.mean(ys)) / \
           (np.mean(xs**2)- (np.mean(xs))**2+1e-9)
        b= np.mean(ys)- m*np.mean(xs)
        return (m,b)

    m_top,b_top= best_fit_line(top_piv)
    m_bot,b_bot= best_fit_line(bot_piv)
    slope_diff= abs(m_top- m_bot)/(abs(m_top)+1e-9)
    if slope_diff> parallel_thresh:
        msg= f"Slope diff {slope_diff:.3f}>threshold => not channel."
        result["msgs"].append(msg)
        return result

    result["found"]= True
    result["upper_line_points"]= top_piv
    result["lower_line_points"]= bot_piv
    result["upper_line_eq"]= (m_top,b_top)
    result["lower_line_eq"]= (m_bot,b_bot)

    avg_slope= (m_top+ m_bot)/2
    if abs(avg_slope)<0.01:
        result["channel_type"]="horizontal"
    elif avg_slope>0:
        result["channel_type"]="ascending"
    else:
        result["channel_type"]="descending"

    last_i= len(df)-1
    last_close= df[close_col].iloc[-1]
    top_line_val= m_top* last_i+ b_top
    bot_line_val= m_bot* last_i+ b_bot
    breakout_up= (last_close> top_line_val)
    breakout_down= (last_close< bot_line_val)
    if breakout_up or breakout_down:
        result["breakout"]= True

        def line_points_from_regression(m,b, pivot_list):
            xvals=[p[0] for p in pivot_list]
            x_min,x_max= min(xvals), max(xvals)
            y_min= m*x_min+ b
            y_max= m*x_max+ b
            return ((x_min,y_min),(x_max,y_max))

        if breakout_up:
            line2d= line_points_from_regression(m_top,b_top, top_piv)
            result["breakout_line"]= line2d
        else:
            line2d= line_points_from_regression(m_bot,b_bot, bot_piv)
            result["breakout_line"]= line2d

        if check_retest and result["breakout_line"]:
            (ixA,pxA),(ixB,pxB)= result["breakout_line"]
            mC,bC= line_equation(ixA,pxA,ixB,pxB)
            if mC is not None:
                retest_done=False
                retest_bar=None
                for i in range(ixB+1, len(df)):
                    c= df[close_col].iloc[i]
                    line_y= mC*i+ bC
                    diff_perc= abs(c-line_y)/(abs(line_y)+1e-9)
                    if diff_perc<= retest_tolerance:
                        retest_done= True
                        retest_bar= i
                        break
                result["retest_info"]={
                    "retest_done": retest_done,
                    "retest_bar": retest_bar
                }
    return result


############################
# GANN ULTRA FINAL
############################

def get_planet_angle(dt, planet_name="SUN"):
    if swe is None:
        return 0
    if not isinstance(dt, (datetime, date)):
        dt= pd.to_datetime(dt)
    jd= swe.julday(dt.year, dt.month, dt.day)
    planet_codes= {
        "SUN": swe.SUN, "MOON": swe.MOON, "MERCURY": swe.MERCURY, "VENUS": swe.VENUS,
        "MARS": swe.MARS, "JUPITER": swe.JUPITER, "SATURN": swe.SATURN,
        "URANUS": swe.URANUS, "NEPTUNE": swe.NEPTUNE, "PLUTO": swe.PLUTO
    }
    pcode= planet_codes.get(planet_name.upper(), swe.SUN)
    flag= swe.FLG_SWIEPH | swe.FLG_SPEED
    pos, ret= swe.calc(jd, pcode, flag)
    return pos[0]  # 0-360

def advanced_wheel_of_24_variants(anchor_price: float, variant: str = "typeA", steps: int = 5):
    """
    Basit Wheel-of-24 türetmesi. 
    """
    levels = []
    if anchor_price <= 0:
        return levels
    if variant == "typeA":
        for n in range(1, steps+1):
            uv = anchor_price*(1+ n*(24/100))
            dv = anchor_price*(1- n*(24/100))
            if uv>0: levels.append(uv)
            if dv>0: levels.append(dv)
    elif variant == "typeB":
        base= math.sqrt(24)
        anc_sqrt= math.sqrt(anchor_price)
        for n in range(1, steps+1):
            upv= (anc_sqrt+ n*base)**2
            dnv= None
            if anc_sqrt> n*base:
                dnv= (anc_sqrt- n*base)**2
            if upv>0: levels.append(upv)
            if dnv and dnv>0: levels.append(dnv)
    else:
        for n in range(1, steps+1):
            uv= anchor_price+ n*15
            dv= anchor_price- n*15
            if uv>0: levels.append(uv)
            if dv>0: levels.append(dv)
    return sorted(list(set(levels)))

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

def detect_gann_pattern_ultra_final(
    df: pd.DataFrame,
    pivots,
    use_ultra: bool=True,
    time_frame: str = "1m",
    # Gann Ratios veya Angle
    use_gann_ratios: bool = True,
    gann_ratios = [1.0, 2.0, 0.5],
    angles = [45.0, 22.5, 67.5, 90.0, 135.0, 180.0],
    additional_angle_shift: float = 180.0,

    # Anchor / Pivot param
    pivot_window: int = 200,
    anchor_count: int = 3,
    pivot_select_mode: str = "extremes_vol",

    # Fan param
    line_tolerance: float = 0.005,
    min_line_respects: int = 3,

    # ATR/Volume filtreleri
    atr_filter: bool = True,
    atr_period: int = 14,
    atr_factor: float = 0.5,
    volume_filter: bool = False,
    volume_ratio: float = 1.3,

    # Square of 9 & Wheel of 24
    sq9_variant: str = "typeA",
    sq9_steps: int = 5,
    sq9_tolerance: float = 0.01,
    w24_variant: str = "typeB",
    w24_steps: int = 5,
    w24_tolerance: float = 0.01,

    # Time / Astro cycles
    cycles = None,
    astro_cycles = None,
    cycle_pivot_tolerance: int = 2,
    pivot_left_bars: int = 3,
    pivot_right_bars: int = 3,

    debug: bool = False,
    check_retest: bool = False,
    retest_tolerance: float = 0.01
)-> dict:
    """
    Gann Patterns hepsi bir arada. 
    """
    import numpy as np

    result={
        "pattern": "gann",
        "found": False,
        "best_anchor": None,
        "anchors": [],
        "gann_line": None,
        "retest_info": None,
        "msgs": []
    }

    close_col= f"Close_{time_frame}"
    if cycles is None:
        cycles= [30,90,180]
    if astro_cycles is None:
        astro_cycles= [90,180,360]

    if close_col not in df.columns or len(df)< pivot_window:
        result["msgs"].append("Not enough data or missing close_col.")
        return result

    if atr_filter:
        prepare_atr(df, time_frame, period=atr_period)

    # 1) Anchor pivot seçimi
    anchor_pivots= []
    seg= df[close_col].iloc[-pivot_window:]
    smin= seg.min()
    smax= seg.max()
    i_min= seg.idxmin()
    i_max= seg.idxmax()
    anchor_pivots.append((i_min, smin))
    anchor_pivots.append((i_max, smax))

    if pivot_select_mode=="extremes_vol":
        vol_col= get_col_name("Volume", time_frame)
        if vol_col in df.columns:
            vseg= df[vol_col].iloc[-pivot_window:]
            iv= vseg.idxmax()
            if iv not in [i_min, i_max]:
                pv= df[close_col].loc[iv]
                anchor_pivots.append((iv, pv))

    anchor_pivots= list(dict.fromkeys(anchor_pivots))
    if len(anchor_pivots)> anchor_count:
        anchor_pivots= anchor_pivots[:anchor_count]

    def slope_from_gann_ratio(ratio: float)-> float:
        return ratio

    def slope_from_angle(deg: float)-> float:
        return math.tan(math.radians(deg))

    def build_fan_lines(anc_idx, anc_val):
        fan=[]
        if use_gann_ratios:
            for r in gann_ratios:
                m_pos= slope_from_gann_ratio(r)
                m_neg= -m_pos
                fan.append({
                    "label": f"{r}x1(+)",
                    "ratio": r,
                    "angle_deg": None,
                    "slope": m_pos,
                    "respects": 0,
                    "confidence": 0.0,
                    "points":[]
                })
                fan.append({
                    "label": f"{r}x1(-)",
                    "ratio": -r,
                    "angle_deg": None,
                    "slope": m_neg,
                    "respects": 0,
                    "confidence": 0.0,
                    "points":[]
                })
        else:
            expanded_angles= angles[:]
            if additional_angle_shift>0:
                for ag in angles:
                    shifted= ag+ additional_angle_shift
                    if shifted not in expanded_angles:
                        expanded_angles.append(shifted)
            expanded_angles= sorted(list(set(expanded_angles)))
            for ag in expanded_angles:
                m= slope_from_angle(ag)
                fan.append({
                    "label": f"{ag}°",
                    "ratio": None,
                    "angle_deg": ag,
                    "slope": m,
                    "respects": 0,
                    "confidence":0.0,
                    "points":[]
                })
        return fan

    def check_fan_respects(fan_lines, anc_idx, anc_val):
        for b_i in range(len(df)):
            px= df[close_col].iloc[b_i]
            # ATR Filter
            if atr_filter:
                atr_col= get_col_name("ATR", time_frame)
                if atr_col in df.columns:
                    av= df[atr_col].iloc[b_i]
                    if not math.isnan(av):
                        rng= df[get_col_name("High", time_frame)].iloc[b_i]- df[get_col_name("Low", time_frame)].iloc[b_i]
                        if rng< (av* atr_factor):
                            continue
            xdiff= b_i- anc_idx
            for fl in fan_lines:
                line_y= fl["slope"]* xdiff+ anc_val
                dist_rel= abs(px- line_y)/(abs(line_y)+1e-9)
                if dist_rel< line_tolerance:
                    # pivot?
                    ptype= None
                    # local min check
                    # buraya pivot_left_bars vs. entegre edebilirsiniz
                    fl["respects"]+=1
                    fl["points"].append((b_i, line_y, px, ptype))
        for fl in fan_lines:
            c=0.0
            pivot_count= sum(1 for pt in fl["points"] if pt[3] is not None)
            if fl["respects"]>= min_line_respects:
                c= 0.5+ min(0.5, 0.05* (fl["respects"]- min_line_respects))
            c+= pivot_count*0.1
            if c>1.0: c=1.0
            fl["confidence"]= round(c,2)

    def compute_sq9_levels(anchor_price: float):
        return advanced_wheel_of_24_variants(anchor_price, variant=sq9_variant, steps=sq9_steps)

    def check_levels_respects(level_list, tolerance):
        out=[]
        for lv in level_list:
            res_count=0
            plist=[]
            for b_i in range(len(df)):
                px_b= df[close_col].iloc[b_i]
                dist= abs(px_b- lv)/(abs(lv)+1e-9)
                if dist< tolerance:
                    res_count+=1
                    plist.append((b_i, px_b))
            conf_= min(1.0, res_count/10)
            out.append((lv, res_count, conf_, plist))
        return out

    def compute_wheel24(anchor_price: float):
        return advanced_wheel_of_24_variants(anchor_price, variant=w24_variant, steps=w24_steps)

    def build_time_cycles(anchor_idx):
        cyc_data=[]
        # sabit bar cycles
        for cyc in cycles:
            tbar= anchor_idx+ cyc
            cyc_date= df.index[tbar] if (0<= tbar< len(df)) else None
            cyc_data.append({
                "bars": cyc,
                "astro": None,
                "target_bar": tbar,
                "approx_date": str(cyc_date) if cyc_date else None,
                "cycle_confidence": 0.0,
                "pivot_detected": None
            })
        # astro
        anchor_date= df.index[anchor_idx] if (0<= anchor_idx< len(df)) else None
        if anchor_date:
            anchor_astro_angle= get_planet_angle(anchor_date, "SUN")
        else:
            anchor_astro_angle= 0
        for deg in astro_cycles:
            target_bar= anchor_idx+ deg
            cyc_date= df.index[target_bar] if (0<= target_bar< len(df)) else None
            cyc_data.append({
                "bars": None,
                "astro": (anchor_astro_angle+ deg)%360,
                "target_bar": target_bar,
                "approx_date": str(cyc_date) if cyc_date else None,
                "cycle_confidence":0.0,
                "pivot_detected": None
            })
        return cyc_data

    def is_local_min(df, bar_i: int, close_col: str, left_bars: int, right_bars: int)-> bool:
        if bar_i< left_bars or bar_i> (len(df)- right_bars-1):
            return False
        val= df[close_col].iloc[bar_i]
        left_slice= df[close_col].iloc[bar_i- left_bars: bar_i]
        right_slice= df[close_col].iloc[bar_i+1: bar_i+1+ right_bars]
        return (all(val< x for x in left_slice) and all(val<= x for x in right_slice))

    def is_local_max(df, bar_i: int, close_col: str, left_bars: int, right_bars: int)-> bool:
        if bar_i< left_bars or bar_i> (len(df)- right_bars-1):
            return False
        val= df[close_col].iloc[bar_i]
        left_slice= df[close_col].iloc[bar_i- left_bars: bar_i]
        right_slice= df[close_col].iloc[bar_i+1: bar_i+1+ right_bars]
        return (all(val> x for x in left_slice) and all(val>= x for x in right_slice))

    def check_cycle_pivots(cyc_data):
        for ci in cyc_data:
            tb= ci["target_bar"]
            if (tb is not None) and (0<= tb< len(df)):
                lb= max(0, tb- cycle_pivot_tolerance)
                rb= min(len(df)-1, tb+ cycle_pivot_tolerance)
                found_piv= None
                for b_ in range(lb, rb+1):
                    if is_local_min(df, b_, close_col, pivot_left_bars, pivot_right_bars):
                        found_piv= (b_, df[close_col].iloc[b_], "min")
                        break
                    elif is_local_max(df, b_, close_col, pivot_left_bars, pivot_right_bars):
                        found_piv= (b_, df[close_col].iloc[b_], "max")
                        break
                if found_piv:
                    ci["pivot_detected"]= found_piv
                    ci["cycle_confidence"]= 1.0
                else:
                    ci["cycle_confidence"]= 0.0

    def build_confluence_points(fan_lines, sq9_data, w24_data, cyc_data, anc_idx, anc_val):
        conf=[]
        for fl in fan_lines:
            if fl["confidence"]<=0:
                continue
            for (b_i, line_y, px, ptype) in fl["points"]:
                sq9_match= None
                for (lvl, rescount, conf_, plist) in sq9_data:
                    dist= abs(px- lvl)/(abs(lvl)+1e-9)
                    if dist< (sq9_tolerance*2):
                        sq9_match= lvl
                        break
                w24_match= None
                for (wl, wres, wconf, wpl) in w24_data:
                    dist= abs(px- wl)/(abs(wl)+1e-9)
                    if dist< (w24_tolerance*2):
                        w24_match= wl
                        break
                cyc_found= None
                for ci in cyc_data:
                    if ci["cycle_confidence"]>0 and ci["pivot_detected"]:
                        if abs(ci["pivot_detected"][0]- b_i)<= cycle_pivot_tolerance:
                            cyc_found= ci
                            break
                if (sq9_match or w24_match or cyc_found):
                    cboost= fl["confidence"]
                    if sq9_match: cboost+= 0.3
                    if w24_match: cboost+= 0.2
                    if cyc_found: cboost+= 0.4
                    if cboost> 2.0: cboost= 2.0
                    conf.append({
                        "bar_index": b_i,
                        "price": px,
                        "fan_line_label": fl["label"],
                        "ptype": ptype,
                        "sq9_level": sq9_match,
                        "w24_level": w24_match,
                        "cycle_bar": cyc_found["target_bar"] if cyc_found else None,
                        "confidence_boost": round(cboost,2)
                    })
        return conf

    anchor_list=[]
    for (anc_idx, anc_val) in anchor_pivots:
        item={
            "anchor_idx": anc_idx,
            "anchor_price": anc_val,
            "fan_lines":[],
            "sq9_levels":[],
            "wheel24_levels":[],
            "time_cycles":[],
            "confluence_points":[],
            "score":0.0
        }
        fl= build_fan_lines(anc_idx, anc_val)
        check_fan_respects(fl, anc_idx, anc_val)
        item["fan_lines"]= fl

        sq9_lvls= compute_sq9_levels(anc_val)
        sq9_data= check_levels_respects(sq9_lvls, sq9_tolerance)
        item["sq9_levels"]= sq9_data

        w24_lvls= compute_wheel24(anc_val)
        w24_data= check_levels_respects(w24_lvls, w24_tolerance)
        item["wheel24_levels"]= w24_data

        cyc_data= build_time_cycles(anc_idx)
        check_cycle_pivots(cyc_data)
        item["time_cycles"]= cyc_data

        conf_pts= build_confluence_points(fl, sq9_data, w24_data, cyc_data, anc_idx, anc_val)
        item["confluence_points"]= conf_pts

        best_fan_conf= max([f["confidence"] for f in fl]) if fl else 0
        ccount= len(conf_pts)
        item["score"]= round(best_fan_conf+ ccount*0.2,2)
        anchor_list.append(item)

    if not anchor_list:
        return result
    best_anchor= max(anchor_list, key=lambda x: x["score"])
    result["best_anchor"]= best_anchor
    result["anchors"]= anchor_list
    if best_anchor["score"]<= 0:
        return result

    result["found"]= True
    best_fan_line= None
    best_conf= -999
    for fl in best_anchor["fan_lines"]:
        if fl["confidence"]> best_conf:
            best_conf= fl["confidence"]
            best_fan_line= fl
    if best_fan_line and best_fan_line["respects"]>=3:
        anc_idx= best_anchor["anchor_idx"]
        anc_val= best_anchor["anchor_price"]
        pivot_points= [pt for pt in best_fan_line["points"] if pt[3] is not None]
        if len(pivot_points)>=2:
            x_vals= [pp[0] for pp in pivot_points]
            y_vals= [pp[2] for pp in pivot_points]
            m_ref,b_ref= np.polyfit(x_vals, y_vals,1)
            x2= anc_idx+ 100
            y2= m_ref*x2+ b_ref
            result["gann_line"]= ((anc_idx, anc_val),(x2,y2))
        else:
            x2= anc_idx+ 100
            y2= anc_val+ best_fan_line["slope"]*100
            result["gann_line"]= ((anc_idx,anc_val),(x2,y2))

    if check_retest and result["gann_line"]:
        (ixA, pxA),(ixB, pxB)= result["gann_line"]
        m_line, b_line= line_equation(ixA, pxA, ixB, pxB)
        if m_line is not None:
            retest_done=False
            retest_bar=None
            start_bar= int(max(ixA, ixB))
            for i in range(start_bar, len(df)):
                c= df[close_col].iloc[i]
                line_y= m_line* i+ b_line
                dist_perc= abs(c- line_y)/(abs(line_y)+1e-9)
                if dist_perc<= retest_tolerance:
                    retest_done= True
                    retest_bar= i
                    break
            result["retest_info"]= {
                "retest_done": retest_done,
                "retest_bar": retest_bar
            }
    return result


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
    "gann": detect_gann_pattern_ultra_final
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

##############################################################################
# 11) ÖRNEK TEST / KULLANIM
##############################################################################

if __name__=="__main__":
    """
    Basit bir test senaryosu. Elinizde bir DataFrame olsun:
    Kolon adları: DatetimeIndex, 
      "Open_1m","High_1m","Low_1m","Close_1m","Volume_1m" vs...
    """
    # Bu kısımda test için sahte data oluşturabilirsiniz veya gerçekte CSV okuyabilirsiniz.
    # Örnek:
    data_size = 500
    dates = pd.date_range("2023-01-01", periods=data_size, freq="T")  # 1m
    np.random.seed(42)
    price = np.cumsum(np.random.randn(data_size))*0.1 + 100
    high_ = price + np.random.rand(data_size)*0.5
    low_  = price - np.random.rand(data_size)*0.5
    volume= np.random.randint(100,1000, size=data_size)

    df_test= pd.DataFrame({
        "Open_1m": price,
        "High_1m": high_,
        "Low_1m": low_,
        "Close_1m": price,
        "Volume_1m": volume
    }, index=dates)

    symbol_test= "TESTCOIN"
    tf_test= "1m"

    # 1) pivot param optimize => kaydet
    # sys_res= optimize_system_parameters(df_test, symbol_test, tf_test)
    # print("Pivot optimize result:", sys_res)

    # 2) detect_all_patterns
    import asyncio
    loop = asyncio.get_event_loop()
    det_res = loop.run_until_complete(
        detect_all_patterns_v2(
            df= df_test,
            symbol= symbol_test,
            time_frame= tf_test
        )
    )

    # 3) İnceleme
    for pname, pres in det_res.items():
        print(f"Pattern: {pname} =>", pres)
        # “pres” pattern fonksiyonunun döndürdüğü dict/list
        # Bazıları list (örn. double_top), bazıları dict (wolfe, gann vs.)

    print("Done.")