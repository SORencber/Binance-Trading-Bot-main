# data_feed.py

import pandas as pd
import numpy as np
import time
from typing import Optional, Callable

class DataFeed:
    """
    Tam ticari uygulamada:
    - start() => WebSocket / REST => bar verisi al
    - on_bar_close => callback
    """
    def __init__(self, symbol:str, bar_interval:str="1h",
                 on_bar_close: Optional[Callable] = None):
        self.symbol = symbol
        self.bar_interval = bar_interval
        self.on_bar_close = on_bar_close
        self.df_bars = pd.DataFrame()

    def load_csv(self, path:str):
        df = pd.read_csv(path, parse_dates=True, index_col=0)
        df.sort_index(inplace=True)
        self.df_bars = df

    def get_bars(self):
        return self.df_bars

    def start_realtime_sim(self, n_updates=5, wait_sec=2):
        for _ in range(n_updates):
            time.sleep(wait_sec)
            new_bar = {
                "Open":10.0+np.random.random(),
                "High":10.3+np.random.random(),
                "Low":9.9,
                "Close":10.1+np.random.random(),
                "Volume": np.random.randint(1000,3000)
            }
            ts = pd.Timestamp.now()
            df_new = pd.DataFrame([new_bar], index=[ts])
            self.df_bars = pd.concat([self.df_bars, df_new])
            if self.on_bar_close:
                self.on_bar_close(self.df_bars)
