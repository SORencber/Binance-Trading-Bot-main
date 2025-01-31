import os
import math
from typing import Optional, Callable, List, Dict

import  pandas as pd

import numpy as np
from core.logging_setup import log

# sklearn / joblib sadece örnek, her zaman gerekmeyebilir
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import sqlite3  # örnek DB için


##############################################################################
# 0.1) ENV / CONFIG
##############################################################################
# Bazı parametreleri environment variable üzerinden çekme örneği:
DB_PATH        = os.getenv("DB_PATH", "trades.db")

##############################################################################
# config.py benzeri:
##############################################################################
TIMEFRAME_CONFIGS = {
    "1m": {
        "system_params": {
            # Daha hızlı tepki için azaltıldı
            "pivot_left_bars": 5,
            "pivot_right_bars": 5,
            "volume_filter": True,
            "min_atr_factor": 0.3
        },
        "pattern_config": {
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
                "left_bars": 5,
                "right_bars": 5,
                "check_volume": False
            },
            "headshoulders": {
                "left_bars": 10,
                "right_bars": 10,
                "min_distance_bars": 10,
                "shoulder_tolerance": 0.03,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 50,
                "atr_filter": 0.2
            },
            "inverse_headshoulders": {
                "left_bars": 10,
                "right_bars": 10,
                "min_distance_bars": 10,
                "shoulder_tolerance": 0.03,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 50,
                "atr_filter": 0.2
            },
            "doubletriple": {
                "tolerance": 0.015,
                "min_distance_bars": 20,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    },

    "5m": {
        "system_params": {
            # 5m için orta seviye pivot
            "pivot_left_bars": 10,
            "pivot_right_bars": 10,
            "volume_filter": True,
            "min_atr_factor": 0.5
        },
        "pattern_config": {
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
                # Tüm TF'lerde aynı pattern seti
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 8,
                "right_bars": 8,
                "check_volume": True
            },
            "headshoulders": {
                "left_bars": 15,
                "right_bars": 15,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3
            },
            "inverse_headshoulders": {
                "left_bars": 15,
                "right_bars": 15,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 25,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    },

    "15m": {
        "system_params": {
            # 15 bar pivot
            "pivot_left_bars": 15,
            "pivot_right_bars": 15,
            "volume_filter": True,
            "min_atr_factor": 0.7
        },
        "pattern_config": {
            "elliott": {
                "fib_tolerance": 0.07,
                "wave_min_bars": 30,
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
                "check_volume": True
            },
            "headshoulders": {
                "left_bars": 20,
                "right_bars": 20,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3
            },
            "inverse_headshoulders": {
                "left_bars": 20,
                "right_bars": 20,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 60,
                "atr_filter": 0.3
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 30,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    },

    "30m": {
        "system_params": {
            # 20 bar pivot
            "pivot_left_bars": 20,
            "pivot_right_bars": 20,
            "volume_filter": True,
            "min_atr_factor": 0.8
        },
        "pattern_config": {
            "elliott": {
                "fib_tolerance": 0.08,
                "wave_min_bars": 40,
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
                "check_volume": True
            },
            "headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.3
            },
            "inverse_headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 20,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.3
            },
            "doubletriple": {
                # Artık 30m'de de hacim/neckline aktif
                "tolerance": 0.01,
                "min_distance_bars": 40,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    },

    "1h": {
        "system_params": {
            # 30 bar pivot
            "pivot_left_bars": 30,
            "pivot_right_bars": 30,
            "volume_filter": True,
            "min_atr_factor": 1.0
        },
        "pattern_config": {
            "elliott": {
                "fib_tolerance": 0.06,
                "wave_min_bars": 40,
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
                "fib_tolerance": 0.02,
                "patterns": ["gartley","bat","crab","butterfly","shark","cipher"],
                "left_bars": 12,
                "right_bars": 12,
                "check_volume": True
            },
            "headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.5
            },
            "inverse_headshoulders": {
                "left_bars": 30,
                "right_bars": 30,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 70,
                "atr_filter": 0.5
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 40,
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    },

    "4h": {
        "system_params": {
            # 40 bar pivot
            "pivot_left_bars": 40,
            "pivot_right_bars": 40,
            "volume_filter": True,
            "min_atr_factor": 1.2
        },
        "pattern_config": {
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
                "check_volume": True
            },
            "headshoulders": {
                "left_bars": 35,
                "right_bars": 35,
                "min_distance_bars": 35,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 0.5
            },
            "inverse_headshoulders": {
                "left_bars": 35,
                "right_bars": 35,
                "min_distance_bars": 35,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 0.5
            },
            "doubletriple": {
                "tolerance": 0.01,
                "min_distance_bars": 45,  # 4h
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    },

    "1d": {
        "system_params": {
            # 50 bar pivot (daha uzun dönem)
            "pivot_left_bars": 50,
            "pivot_right_bars": 50,
            "volume_filter": True,
            "min_atr_factor": 1.5
        },
        "pattern_config": {
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
                "check_volume": True
            },
            "headshoulders": {
                "left_bars": 25,
                "right_bars": 25,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 1.0
            },
            "inverse_headshoulders": {
                "left_bars": 25,
                "right_bars": 25,
                "min_distance_bars": 30,
                "shoulder_tolerance": 0.02,
                "volume_decline": True,
                "neckline_break": True,
                "max_shoulder_width_bars": 80,
                "atr_filter": 1.0
            },
            "doubletriple": {
                "tolerance": 0.008,
                "min_distance_bars": 50,  # 1d için artırıldı
                "triple_variation": True,
                "volume_check": True,
                "neckline_break": True
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
            }
        }
    }
}

##############################################################################
# 0.2) TRADE DATABASE
##############################################################################
class TradeDatabase:
    """
    Basit bir SQLite veritabanı örneği.
    trade_logs tablosuna sinyal veya pozisyon bilgilerini kaydedebilir.
    Production'da PostgreSQL vs. olabilir.
    """
    def __init__(self, db_path="trades.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            action TEXT,
            price REAL,
            qty REAL,
            pnl REAL,
            note TEXT
        )
        """)
        con.commit()
        con.close()

    def log_trade(self, timestamp, symbol, action, price, qty, pnl, note=""):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute("""
        INSERT INTO trade_logs (timestamp, symbol, action, price, qty, pnl, note)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, symbol, action, price, qty, pnl, note))
        con.commit()
        con.close()

trade_db = TradeDatabase(db_path=DB_PATH)

def get_col_name(base_col: str, time_frame: str) -> str:
    """
    Örnek: get_col_name("High", "5m") -> "High_5m"
    """
    #return f"{base_col}_{time_frame}"
    return f"{base_col}_{time_frame}"

class AdvancedPivotScanner:
    """
    Gelişmiş pivot tarayıcı. ATR hesaplama, hacim filtresi vb. içeriyor.
    """
    def __init__(self,
                 left_bars=10, 
                 right_bars=10,
                 volume_filter=False,
                 min_atr_factor=0.0,
                 df: pd.DataFrame = None,
                 time_frame: str = "1m"):  
        """
        Artık time_frame de alıyoruz ki _ensure_atr() içinde hangi kolonları kullanacağımızı bilelim.
        """
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.volume_filter = volume_filter
        self.min_atr_factor = min_atr_factor
        self.df = df.copy() if df is not None else None
        self.time_frame = time_frame  # kritik

        if self.df is not None:
            self._ensure_atr()

    def _ensure_atr(self):
        """
        ATR hesaplaması: "High_{time_frame}", "Low_{time_frame}", "Close_{time_frame}" 
        üzerinden "ATR_{time_frame}" oluşturur.
        """
        high_col  = get_col_name("High",  self.time_frame)
        low_col   = get_col_name("Low",   self.time_frame)
        close_col = get_col_name("Close", self.time_frame)
        atr_col   = get_col_name("ATR",   self.time_frame)

        # Eğer ATR kolonu yoksa oluştur
        if atr_col not in self.df.columns:
            # Geçici TR hesap kolonları
            hl_col  = f"H-L_{self.time_frame}"
            hpc_col = f"H-PC_{self.time_frame}"
            lpc_col = f"L-PC_{self.time_frame}"
            tr_col  = f"TR_{self.time_frame}"

            self.df[hl_col]  = self.df[high_col] - self.df[low_col]
            self.df[hpc_col] = (self.df[high_col] - self.df[close_col].shift(1)).abs()
            self.df[lpc_col] = (self.df[low_col]  - self.df[close_col].shift(1)).abs()
            self.df[tr_col]  = self.df[[hl_col,hpc_col,lpc_col]].max(axis=1)

            # 14 peryotluk ATR
            self.df[atr_col] = self.df[tr_col].rolling(14).mean()

    def find_pivots(self):
        """
        Piyasa fiyatını (Close_{time_frame} sütununu) kullanarak pivot arar.
        """
        price_col = get_col_name("Close", self.time_frame)
        if price_col not in self.df.columns:
            raise ValueError(f"DataFrame does not have column {price_col} for {self.time_frame}")

        price_series = self.df[price_col]
        pivots = []
        n = len(price_series)

        for i in range(self.left_bars, n - self.right_bars):
            val = price_series.iloc[i]
            left_slice  = price_series.iloc[i - self.left_bars : i]
            right_slice = price_series.iloc[i+1 : i+1 + self.right_bars]

            # TEPE (local max)
            if all(val > x for x in left_slice) and all(val >= x for x in right_slice):
                if self._pivot_ok(i, val, +1):
                    pivots.append((i, val, +1))
            # DIP (local min)
            elif all(val < x for x in left_slice) and all(val <= x for x in right_slice):
                if self._pivot_ok(i, val, -1):
                    pivots.append((i, val, -1))

        return pivots

    def _pivot_ok(self, idx, val, ptype):
        """
        Ek filtreler: hacim, ATR vs.
        """
        if self.df is None:
            return True

        # Hacim filtresi
        if self.volume_filter:
            vol_col = get_col_name("Volume", self.time_frame)
            if vol_col in self.df.columns:
                vol_now = self.df[vol_col].iloc[idx]
                vol_mean = self.df[vol_col].iloc[max(0, idx-20): idx].mean()
                if vol_now < 1.2 * vol_mean:
                    return False

        # ATR faktörü
        if self.min_atr_factor > 0:
            atr_col = get_col_name("ATR", self.time_frame)
            if atr_col in self.df.columns:
                atr_now = self.df[atr_col].iloc[idx]
                if pd.isna(atr_now):
                    return True
                # Örnek bir kontrol
                if abs(val - self.df[atr_col].iloc[idx-1]) < (self.min_atr_factor * atr_now):
                    return False

        return True


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
# 4) ML MODEL
##############################################################################
class PatternEnsembleModel:
    """
    Örnek ML modeli (RandomForest) pipeline.
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



def check_breakout_volume(
    df: pd.DataFrame, 
    time_frame: str = "1m",
    atr_window: int = 14,
    vol_window: int = 20
) -> tuple:
    """
    Basit breakout + hacim spike kontrolü, time_frame'e göre kolonları okur:
      breakout_up, breakout_down, volume_spike döner.
    """
    high_col   = get_col_name("High", time_frame)
    low_col    = get_col_name("Low",  time_frame)
    close_col  = get_col_name("Close", time_frame)
    volume_col = get_col_name("Volume", time_frame)
    atr_col    = get_col_name("ATR", time_frame)

    # ATR hesaplaması yoksa ekle
    if atr_col not in df.columns:
        hl_  = f"H-L_{time_frame}"
        hpc_ = f"H-PC_{time_frame}"
        lpc_ = f"L-PC_{time_frame}"
        tr_  = f"TR_{time_frame}"

        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
            return (False,False,False)

        df[hl_]  = df[high_col] - df[low_col]
        df[hpc_] = (df[high_col] - df[close_col].shift(1)).abs()
        df[lpc_] = (df[low_col]  - df[close_col].shift(1)).abs()
        df[tr_]  = df[[hl_, hpc_, lpc_]].max(axis=1)
        df[atr_col] = df[tr_].rolling(atr_window).mean()

    if len(df)<2:
        return (False,False,False)

    last_close = df[close_col].iloc[-1]
    prev_close = df[close_col].iloc[-2]
    last_atr   = df[atr_col].iloc[-1]
    if pd.isna(last_atr):
        last_atr=0

    # breakout up/down
    breakout_up   = (last_close - prev_close)> last_atr
    breakout_down = (prev_close - last_close)> last_atr

    volume_spike= False
    if volume_col in df.columns and len(df)> vol_window:
        v_now  = df[volume_col].iloc[-1]
        v_mean = df[volume_col].rolling(vol_window).mean().iloc[-2]
        volume_spike= (v_now> 1.5* v_mean)

    return (breakout_up, breakout_down, volume_spike)


   ##############################################################################
# 6) SIGNAL ENGINE (generate_signals)
##############################################################################

def generate_signals(
    df: pd.DataFrame, 
    time_frame: str = "1m", 
    ml_model=None,
    max_bars_ago: int = 300,         # Son X bar içinde biten pattern'leri dikkate almak
    require_confirmed: bool = False  # True => confirmed=False pattern’leri es geç
) -> dict:
    """
    Gelişmiş Pattern + ML + Breakout & Hacim analizi => final sinyal üretimi.
    
    Dönüş formatı:
      {
        "signal": "BUY"/"SELL"/"HOLD",
        "score": <int>,
        "reason": "<metin>",
        "patterns": {...},             # detect_all_patterns çıktısı (ham)
        "triggered_patterns": {...},   # sinyal oluşturmada katkısı olan pattern'lerin detayları
        "ml_label": 0/1/2,
        "breakout_up": bool,
        "breakout_down": bool,
        "volume_spike": bool,
        "time_frame": "1m"
      }
    """
    if time_frame not in TIMEFRAME_CONFIGS:
        raise ValueError(f"Invalid time_frame='{time_frame}'")

    tf_settings = TIMEFRAME_CONFIGS[time_frame]
    system_params = tf_settings["system_params"]
    pattern_conf  = tf_settings["pattern_config"]

    # 1) Pivot & ZigZag
    scanner = AdvancedPivotScanner(
        left_bars= system_params["pivot_left_bars"],
        right_bars= system_params["pivot_right_bars"],
        volume_filter= system_params["volume_filter"],
        min_atr_factor= system_params["min_atr_factor"],
        df= df,
        time_frame= time_frame
    )
    pivots = scanner.find_pivots()
    wave   = build_zigzag_wave(pivots)

    # 2) Tüm pattern tespiti (all_patterns)
    from trading_view.patterns.all_patterns import detect_all_patterns
    patterns = detect_all_patterns(
        pivots, 
        wave, 
        df=df, 
        time_frame=time_frame, 
        config=pattern_conf
    )

    # 3) ML tahmini (opsiyonel)
    ml_label= None
    if ml_model is not None:
        feats = ml_model.extract_features(wave)
        ml_label = ml_model.predict(feats)[0]  # 0=HOLD,1=BUY,2=SELL

    # 4) Basit Breakout + Hacim kontrolü
    b_up, b_down, v_spike = check_breakout_volume(df, time_frame=time_frame)

    ###################################################################
    # Yardımcı fonksiyon: pattern listesini son bar filtresi ile daralt
    ###################################################################
     
    def filter_patterns(pat_list):
        """
        Sadece 'end_bar' >= (len(df)-max_bars_ago) 
        ve eğer 'require_confirmed'==True ise 'confirmed'=True patternleri döndürür.
        """
        #cutoff = len(df) - max_bars_ago
        filtered = []
        for p in pat_list:
            #if p["end_bar"] >= cutoff:
                if require_confirmed:
                    if p.get("confirmed", False):
                        filtered.append(p)
                else:
                    filtered.append(p)
        return filtered

    # 6) Pattern skorunu hesaplayacak basit kural seti:
    pattern_score = 0
    reasons = []

    # Aşağıdaki bloklar, "en güncel" pattern’lere bakacak şekilde 'filter_patterns' ile filtrelenir

    # --- HEAD & SHOULDERS ---
    hs_list = filter_patterns(patterns["headshoulders"])
    for hs in hs_list:
        val = -3
        if hs["confirmed"] and hs["volume_check"]:
            val = -4
        pattern_score += val
        reasons.append(f"headshoulders({val})")

    inv_hs_list = filter_patterns(patterns["inverse_headshoulders"])
    for inv in inv_hs_list:
        val = +3
        if inv["confirmed"] and inv["volume_check"]:
            val = +4
        pattern_score += val
        reasons.append(f"inverseHS({val})")

    # --- DOUBLE / TRIPLE TOP-BOTTOM ---
    dtops = filter_patterns(patterns["double_top"])
    for dt in dtops:
        val = -2
        if dt.get("confirmed"):
            val -= 1  # -3
        pattern_score += val
        reasons.append(f"double_top({val})")

    dbots = filter_patterns(patterns["double_bottom"])
    for db in dbots:
        val = +2
        if db.get("confirmed"):
            val += 1  # +3
        pattern_score += val
        reasons.append(f"double_bottom({val})")

    # --- ELLIOTT ---
    ell = patterns["elliott"]
    if ell["found"]:
        # Elliott "end_bar" => son pivot p4i
        # Basit: son pivot -> wave[-1][0]
        # Kontrol edelim, son pivot bar'ı cutoff'tan büyük mü
        if wave and wave[-1][0] >= (len(df)-max_bars_ago):
            if ell["trend"] == "UP":
                pattern_score += 3
                reasons.append("elliott_up")
            else:
                pattern_score -= 3
                reasons.append("elliott_down")

    # --- WOLFE ---
    wol = patterns["wolfe"]
    if wol["found"] and wave:
        # wave son pivot bar => wave[-1][0]
        if wave[-1][0] >= (len(df)-max_bars_ago):
            wol_val = +2
            if wol["breakout"]:
                wol_val += 1
            pattern_score += wol_val
            reasons.append(f"wolfe({wol_val})")

    # --- HARMONIC ---
    harm = patterns["harmonic"]
    if harm["found"] and wave:
        if wave[-1][0] >= (len(df)-max_bars_ago):
            pattern_score -= 1
            reasons.append("harmonic(-1)")

    # --- TRIANGLE ---
    tri = patterns["triangle"]
    if tri["found"] and tri["breakout"]:
        # son pivot bar / wave check => basit yaklaşımla 
        # wave[-1] vs. df len
        if wave and wave[-1][0] >= (len(df)-max_bars_ago):
            if tri["triangle_type"] == "ascending":
                pattern_score += 1
                reasons.append("triangle_asc(+1)")
            elif tri["triangle_type"] == "descending":
                pattern_score -= 1
                reasons.append("triangle_desc(-1)")
            else:
                pattern_score += 1
                reasons.append("triangle_sym(+1)")

    # --- WEDGE ---
    wd = patterns["wedge"]
    if wd["found"] and wd["breakout"]:
        if wave and wave[-1][0] >= (len(df)-max_bars_ago):
            if wd["wedge_type"] == "rising":
                pattern_score -= 1
                reasons.append("wedge_rising(-1)")
            else:
                pattern_score += 1
                reasons.append("wedge_falling(+1)")

    # --- ML LABEL ---
    if ml_label == 1:
        pattern_score += 3
        reasons.append("ml_buy")
    elif ml_label == 2:
        pattern_score -= 3
        reasons.append("ml_sell")

    # 7) Ek breakout/hacim -> skor
    final_score = pattern_score
    if final_score > 0:  # potansiyel alıcı
        if b_up:
            final_score += 1
            reasons.append("breakout_up")
            if v_spike:
                final_score += 1
                reasons.append("vol_spike_up")
    elif final_score < 0:  # potansiyel satıcı
        if b_down:
            final_score -= 1
            reasons.append("breakout_down")
            if v_spike:
                final_score -= 1
                reasons.append("vol_spike_down")

    # 8) Son Karar (örnek eşikler: >=+2 => BUY, <=-2 => SELL, aksi => HOLD)
    final_signal = "HOLD"
    if final_score >= 2:
        final_signal = "BUY"
    elif final_score <= -2:
        final_signal = "SELL"

    reason_str = ",".join(reasons) if reasons else "NONE"
    current_price = df[get_col_name("Close", time_frame)].iloc[-1]
    

    # Örnek "extract_pattern_trade_levels" fonksiyonu
    pattern_trade_levels = extract_pattern_trade_levels(patterns, current_price)
    return {
        "signal": final_signal,
        "score": final_score,
        "reason": reason_str,
        "patterns": patterns,
        # "triggered_patterns": triggered_patterns, 
        "ml_label": ml_label,
        "breakout_up": b_up,
        "breakout_down": b_down,
        "volume_spike": v_spike,
        "time_frame": time_frame,

        # YENİ ALAN => Hangi pattern, hangi entry/stop/target/direction gibi
        "pattern_trade_levels": pattern_trade_levels
    }


import math

def extract_pattern_trade_levels(patterns_dict, current_price):
    """
    Her pattern için (entry, stop, target, direction) gibi detayları 
    toplu halde döndüren örnek fonksiyon.

    Dönüş örneği:
    {
      "inverse_headshoulders": [
         {
           "entry_price": ...,
           "stop_price":  ...,
           "target_price": ...,
           "direction": "LONG",
           "confidence": ... (opsiyonel),
           ... başka alanlar
         },
         ...
      ],
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
        "harmonic": []
    }

    ############################################################################
    # 1) Inverse HeadShoulders => Bullish
    ############################################################################
    inv_list = patterns_dict.get("inverse_headshoulders", [])
    # Eğer tek pattern dict dönerse list'e çevir
    if isinstance(inv_list, dict):
        inv_list = [inv_list]
    for inv in inv_list:
        if inv.get("confirmed", False):
            # Head Price
            head_price = inv["H"][1]
            # Boyun çizgisi ortalama
            if inv.get("neckline"):
                (nx1, px1), (nx2, px2) = inv["neckline"]
                neckline_avg = (px1 + px2) / 2
            else:
                neckline_avg = None

            entry_price  = neckline_avg if neckline_avg else current_price
            stop_price   = head_price * 0.98
            if neckline_avg:
                # "measured move" => neckline_avg - head_price
                mm = neckline_avg - head_price
                target_price = neckline_avg + mm
            else:
                # Emniyet varsayım
                target_price = current_price * 1.1

            trade_info = {
                "entry_price":  entry_price,
                "stop_price":   stop_price,
                "target_price": target_price,
                "direction": "LONG",
                "pattern_raw": inv  # opsiyonel: ham pattern
            }
            results["inverse_headshoulders"].append(trade_info)

    ############################################################################
    # 2) Double Bottom => Bullish
    ############################################################################
    db_list = patterns_dict.get("double_bottom", [])
    for db in db_list:
        if db.get("confirmed", False):
            # dip
            dip_price = min(db["bottoms"][0][1], db["bottoms"][-1][1])
            stop_price = dip_price * 0.97

            neck = db.get("neckline", None)
            if neck:
                # (idx, price)
                neck_price = neck[1]
                mm = neck_price - dip_price
                target_price = neck_price + mm
                entry_price  = neck_price  # ya da current_price
            else:
                target_price = current_price * 1.1
                entry_price  = current_price

            trade_info = {
                "entry_price":  entry_price,
                "stop_price":   stop_price,
                "target_price": target_price,
                "direction": "LONG",
                "pattern_raw": db
            }
            results["double_bottom"].append(trade_info)

    ############################################################################
    # 3) Double Top => Bearish
    ############################################################################
    dt_list = patterns_dict.get("double_top", [])
    for dt in dt_list:
        if dt.get("confirmed", False):
            peak_price = max(dt["tops"][0][1], dt["tops"][-1][1])
            stop_price = peak_price * 1.03

            # neckline var mı?
            neck = dt.get("neckline", None)
            if neck:
                neck_price = neck[1]
                # Measured move => peak - neckline => target= neckline - (peak - neckline)
                mm = peak_price - neck_price
                target_price = neck_price - mm
                entry_price  = neck_price  # short entry or current_price
            else:
                target_price = current_price * 0.9
                entry_price  = current_price

            trade_info = {
                "entry_price":  entry_price,
                "stop_price":   stop_price,
                "target_price": target_price,
                "direction": "SHORT",
                "pattern_raw": dt
            }
            results["double_top"].append(trade_info)

    ############################################################################
    # 4) Elliott Wave
    #    - trend == "UP" => bullish
    #    - trend == "DOWN" => bearish
    ############################################################################
    ell = patterns_dict.get("elliott", {})
    if ell.get("found", False):
        wave_pivots = ell.get("pivots", [])  # (idx,price)
        trend = ell.get("trend", None)
        if len(wave_pivots) == 5:
            # p4 => wave_pivots[3], p5 => wave_pivots[4]
            p4_price = wave_pivots[3][1]
            p5_price = wave_pivots[4][1]

            if trend == "UP":
                entry_price  = current_price  # veya p4_price
                stop_price   = p4_price * 0.98
                target_price = p5_price
                direction    = "LONG"
            else:
                # DOWN senaryosu
                entry_price  = current_price
                stop_price   = p4_price * 1.02
                # target => p5 nin altı
                target_price = p5_price
                direction    = "SHORT"

            trade_info = {
                "entry_price":  entry_price,
                "stop_price":   stop_price,
                "target_price": target_price,
                "direction":    direction,
                "pattern_raw":  ell
            }
            results["elliott"].append(trade_info)

    ############################################################################
    # 5) Wolfe Wave
    ############################################################################
    wol = patterns_dict.get("wolfe", {})
    if wol.get("found", False) and wol.get("breakout", False):
        # w5 => (idx, price, pivot_type)
        w5_data = wol.get("w5", None)
        if w5_data:
            w5_price = w5_data[1]
        else:
            w5_price = current_price

        # long mu short mu?
        # Wolfe bazen bullish/bearish olabilir. Örnek basit: breakout => bullish farzet.
        direction = "LONG"
        stop_price = w5_price * 0.98
        target_price = wol.get("wolfe_target", current_price * 1.1)

        trade_info = {
            "entry_price":  current_price,
            "stop_price":   stop_price,
            "target_price": target_price,
            "direction":    direction,
            "pattern_raw":  wol
        }
        results["wolfe"].append(trade_info)

    ############################################################################
    # 6) Triangle => ascending => bullish, descending => bearish
    ############################################################################
    tri = patterns_dict.get("triangle", {})
    if tri.get("found", False):
        tri_type = tri.get("triangle_type", None)
        if tri_type == "ascending":
            # Örnek bullish
            trade_info = {
                "entry_price":  current_price,
                "stop_price":   current_price * 0.95,
                "target_price": current_price * 1.15,
                "direction":    "LONG",
                "pattern_raw":  tri
            }
            results["triangle"].append(trade_info)
        elif tri_type == "descending":
            # Bearish
            trade_info = {
                "entry_price":  current_price,
                "stop_price":   current_price * 1.05,
                "target_price": current_price * 0.85,
                "direction":    "SHORT",
                "pattern_raw":  tri
            }
            results["triangle"].append(trade_info)
        elif tri_type == "symmetrical":
            # Sizin stratejinize göre, symmetrical triangle bullish/bearish olabilir.
            # Örnek: no trade
            pass

    ############################################################################
    # 7) Wedge => rising => genelde bearish, falling => bullish
    ############################################################################
    wedge = patterns_dict.get("wedge", {})
    if wedge.get("found", False):
        w_type = wedge.get("wedge_type", None)
        if w_type == "rising":
            trade_info = {
                "entry_price":  current_price,
                "stop_price":   current_price * 1.05,
                "target_price": current_price * 0.85,
                "direction":    "SHORT",
                "pattern_raw":  wedge
            }
            results["wedge"].append(trade_info)
        elif w_type == "falling":
            trade_info = {
                "entry_price":  current_price,
                "stop_price":   current_price * 0.95,
                "target_price": current_price * 1.15,
                "direction":    "LONG",
                "pattern_raw":  wedge
            }
            results["wedge"].append(trade_info)

    ############################################################################
    # 8) Harmonic => tipik olarak bullish/bearish analizi pattern_name'e göre
    #    Örnek: "bat", "butterfly", "crab" vb. 
    #    Burada sadece bullish farz edelim:
    ############################################################################
    harm = patterns_dict.get("harmonic", {})
    if harm.get("found", False):
        pattern_name = harm.get("pattern_name", "")
        # D noktası:
        xabc_points = harm.get("xabc", [])
        if len(xabc_points) == 5:
            d_price = xabc_points[-1][1]
            direction = "LONG"  # varsayalım
            # Bu kısım pattern'e göre bull/bear seçilebilir
            # Örnek:
            stop_price   = d_price * 0.97
            target_price = d_price * 1.20
            trade_info = {
                "entry_price":  d_price,  # ya da current_price
                "stop_price":   stop_price,
                "target_price": target_price,
                "direction":    direction,
                "pattern_name": pattern_name,
                "pattern_raw":  harm
            }
            results["harmonic"].append(trade_info)

    return results

##############################################################################
# 11) Pattern'lere Göre STOP/TARGET Hesaplayan Fonksiyonlar
##############################################################################

def calculate_stop_and_target(patterns_dict, current_price):
    """
    patterns_dict => Formasyon tespiti çıktısı (ör. detect_all_patterns sonucu).
    current_price => Son bar fiyatı.
    
    Dönüş: (final_stop, final_target)
      - final_stop: "en sıkı" stop (fiyata en yakın => max() of stop_candidates)
      - final_target: "en muhafazakar" (min() of target_candidates)
    """
    stop_candidates = []
    target_candidates = []

    # 1) Inverse H&S
    inv_list = patterns_dict.get("inverse_headshoulders", [])
    if isinstance(inv_list, dict):
        # Bazı dedektörler list yerine dict döndürüyor olabilir
        # Bu durumda skip edebiliriz veya tek bir pattern gibi değerlendirebiliriz
        inv_list = [inv_list]
    for inv in inv_list:
        if inv.get("confirmed", False):
            # Head = inv["H"][1]
            headPrice = inv["H"][1]
            s = headPrice * 0.98  # başın %2 altı
            stop_candidates.append(s)

            # neckline => average
            if "neckline" in inv and inv["neckline"]:
                nx1, px1 = inv["neckline"][0]
                nx2, px2 = inv["neckline"][1]
                neckPrice = (px1 + px2) / 2
                mm = neckPrice - headPrice
                t = neckPrice + mm
                target_candidates.append(t)

    # 2) Double Bottom
    db_list = patterns_dict.get("double_bottom", [])
    for db in db_list:
        if db.get("confirmed", False):
            b1idx, b1price = db["bottoms"][0]
            b2idx, b2price = db["bottoms"][-1]
            dipPrice = min(b1price, b2price)
            s = dipPrice * 0.97
            stop_candidates.append(s)

            if db["neckline"]:
                neckidx, neckPrice = db["neckline"]
                mm = neckPrice - dipPrice
                t = neckPrice + mm
                target_candidates.append(t)

    # 3) Elliott (örnek)
    ell = patterns_dict.get("elliott", {})
    if ell.get("found", False) and ell.get("trend")=="UP":
        # p4 => wave[3], p5 => wave[4]
        # basitleştirilmiş
        wave_pivots= ell.get("pivots", [])
        if len(wave_pivots)==5:
            p4i,p4p= wave_pivots[3]
            p5i,p5p= wave_pivots[4]
            s = p4p * 0.98  # wave4 altı
            stop_candidates.append(s)
            target_candidates.append(p5p)  # wave5 hedef

    # 4) Wolfe
    wol = patterns_dict.get("wolfe", {})
    if wol.get("found", False) and wol.get("breakout", False):
        # w5 pivot altı stop
        # "intersection" line vs. Basit
        # Bazen w5 pivot wave[-1]
        # Aşağıda farazi "w5" ekleyelim
        # Bu dedektör kodu "w5": (idx, price, pivot_type) döndürmemiş olabilir => opsiyonel
        # Örnek:
        w5= wol.get("w5", None)
        if w5:
            idx5, px5 = w5[0], w5[1]
            s = px5 * 0.98
            stop_candidates.append(s)
        # Hedef => 1->4 line intersection vs. "EPA line" 
        # Yukarıda "intersection" diyorduk, ya da "wolfe_target"
        if "wolfe_target" in wol:
            t= wol["wolfe_target"]
            target_candidates.append(t)

    # Diğer patternler (triangle, wedge, harmonic, vs.) - opsiyonel
    
    # En sıkı stop = en yüksek stop
    if stop_candidates:
        final_stop = max(stop_candidates)
    else:
        final_stop = math.nan

    # En muhafazakar hedef = en düşük target
    if target_candidates:
        final_target = min(target_candidates)
    else:
        final_target = math.nan

    return final_stop, final_target


def check_additional_filters(df,time_frame):
    """
    df => pandas DataFrame (örnek: son bar datası).
    Ek göstergeler (RSI, Hacim, vb.) ile teyit.
    Return True/False => İşleme girilsin mi?
    """
    rsi_14  = get_col_name("RSI_14",  time_frame)

    # Örn: RSI>50 mi?
    if rsi_14 in df.columns:
        rsi_ok = df[rsi_14].iloc[-1] > 50
    else:
        rsi_ok = True

    # Örn: Son bar hacmi, son 20 bar ortalamasından büyük mü?
    volume  = get_col_name("Volume",  time_frame)

    if volume in df.columns:
        vol_now = df[volume].iloc[-1]
        vol_mean = df[volume].rolling(20).mean().iloc[-1]
        volume_ok = vol_now > vol_mean
    else:
        volume_ok = True

    return (rsi_ok and volume_ok)
