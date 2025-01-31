
class OneMinVolCollector:
    def __init__(self, lookback_bars=5, threshold=0.01):
        self.lookback= lookback_bars
        self.threshold= threshold

    def evaluate_1m_volatility(self, df_1m)-> float:
        if len(df_1m)< self.lookback:
            return 0
        sub= df_1m.iloc[-self.lookback:]
        first_close= sub.iloc[0]["Close_1m"]
        last_close= sub.iloc[-1]["Close_1m"]
        pct= (last_close- first_close)/(first_close+1e-9)
        if pct> self.threshold:
            return +1
        elif pct< -self.threshold:
            return -1
        return 0
