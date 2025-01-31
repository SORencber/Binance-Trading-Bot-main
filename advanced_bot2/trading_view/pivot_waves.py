# pivots_waves.py

import pandas as pd
import numpy as np

class AdvancedPivotScanner:
    def __init__(self, left_bars=2, right_bars=2, volume_filter=False, 
                 min_atr_factor=0.0, df=None):
        self.left_bars = left_bars
        self.right_bars= right_bars
        self.volume_filter= volume_filter
        self.min_atr_factor=min_atr_factor
        self.df = df.copy() if df is not None else None
        if self.df is not None:
            self._ensure_atr()

    def _ensure_atr(self):
        if "ATR" not in self.df.columns:
            self.df["H-L"]= self.df["High"]-self.df["Low"]
            self.df["H-PC"]=(self.df["High"]-self.df["Close"].shift(1)).abs()
            self.df["L-PC"]=(self.df["Low"] - self.df["Close"].shift(1)).abs()
            self.df["TR"] = self.df[["H-L","H-PC","L-PC"]].max(axis=1)
            self.df["ATR"]= self.df["TR"].rolling(14).mean()

    def find_pivots(self, price_series: pd.Series):
        pivots=[]
        n = len(price_series)
        for i in range(self.left_bars, n - self.right_bars):
            val = price_series.iloc[i]
            left_ = price_series.iloc[i-self.left_bars : i]
            right_= price_series.iloc[i+1 : i+1+self.right_bars]

            if all(val> x for x in left_) and all(val>= x for x in right_):
                if self._pivot_ok(i, val, +1):
                    pivots.append((i,val,+1))
            elif all(val< x for x in left_) and all(val<= x for x in right_):
                if self._pivot_ok(i, val, -1):
                    pivots.append((i,val,-1))
        return pivots

    def _pivot_ok(self, idx, val, ptype):
        if self.df is None:
            return True
        # volume
        if self.volume_filter and "Volume" in self.df.columns:
            vol_now = self.df["Volume"].iloc[idx]
            mean_vol= self.df["Volume"].iloc[max(0,idx-20): idx].mean()
            if vol_now < 1.2 * mean_vol:
                return False
        # ATR distance
        if self.min_atr_factor>0 and "ATR" in self.df.columns:
            atr_now = self.df["ATR"].iloc[idx]
            if pd.isna(atr_now):
                return True
            # ek kural => pivotun onceki pivotla mesafesi >= atr_now*x 
            # Bunu gerceklestirmek icin onceki pivot yok => iskelet
        return True


def build_zigzag_wave(pivots, df=None, min_wave_atr=0.0):
    sorted_p = sorted(pivots, key=lambda x: x[0])
    if not sorted_p:
        return []
    wave = [sorted_p[0]]
    for i in range(1,len(sorted_p)):
        curr = sorted_p[i]
        prev = wave[-1]
        if curr[2]== prev[2]:
            # ayni tip => daha ekstrem
            if curr[2]== +1:
                if curr[1]> prev[1]:
                    wave[-1]= curr
            else:
                if curr[1]< prev[1]:
                    wave[-1]= curr
        else:
            # min_wave_atr
            wave.append(curr)
    return wave
