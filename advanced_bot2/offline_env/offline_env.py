# offline_env.py

import pandas as pd
import numpy as np

class OfflineEnv:
    """
    Gelişmiş bir offline environment, tablo tabanlı RL ajanları için.

    - Eylem seti:
        0 => SELL (Full kapat)
        1 => BUY  (Pozisyon yoksa al)
        2 => HOLD (Pozisyonu koru)
        3 => PARTIAL (Pozisyonun belirli bir yüzdesini sat)

    - State (durum) unsurları (discrete):
        (regime_bin, rsi_bin, stoch_bin, haspos, drawdown_bin)
      Örnek:
        regime_bin: 0=RANGE, 1=TREND
        haspos: 0/1
        rsi_bin, stoch_bin: 0..4 arası
        drawdown_bin: 0..4 arası (örnek)

    Reward shaping:
       - Pozisyon KAPATILDIĞINDA => realizedPnL
       - Pozisyon AÇIKKEN => unrealizedPnL'e küçük oranda +, eğer drawdown artmışsa - ceza
    Kısmi (PARTIAL) aksiyonda => realizedPnL (kısmî) eklenir, pozisyon kısmen azalır.
    Trailing stop => HOLD ya da PARTIAL eylemi sırasında devreye girebilir.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        initial_balance=200.0,
        fee_rate=0.001,
        max_bars=None,
        partial_ratio=0.3,   # PARTIAL satmak istediğimiz oransal miktar
        trailing_stop_pct=0.95,  # "highest_price * 0.95" gibi basit trailing
        max_drawdown_allowed=0.3  # kabaca
    ):
        """
        :param df: gerekli sütunları içeren DataFrame (Close, RSI_5m, StochK_5m, ADX_4h vb.)
        :param initial_balance: USDT
        :param fee_rate: komisyon
        :param max_bars: max bar sayısı (isterseniz)
        :param partial_ratio: partial SELL eyleminde, mevcut pozisyonun ne kadarı satılsın
        :param trailing_stop_pct: trailing stop seviyesi (0.95 => %5 geride)
        :param max_drawdown_allowed: Ceza vermek ya da durumu binleştirmek için
        """
        self.df = df.reset_index(drop=True)
        if max_bars:
            self.df = self.df.iloc[:max_bars].copy().reset_index(drop=True)

        self.n_bars = len(self.df)
        self.current_bar = 0

        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.fee_rate = fee_rate
        self.partial_ratio = partial_ratio
        self.trailing_stop_factor = trailing_stop_pct
        self.max_drawdown_allowed = max_drawdown_allowed

        # Pozisyon
        self.has_position = False
        self.quantity = 0.0
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.net_pnl = 0.0

        # Drawdown takibi:
        self.peak_balance = self.balance
        self.drawdown = 0.0

        # State discretization param
        self.rsi_bins = 5
        self.stoch_bins = 5
        self.drawdown_bins = 5

    def reset(self):
        self.current_bar = 0
        self.balance = self.initial_balance
        self.has_position = False
        self.quantity = 0.0
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.net_pnl = 0.0
        self.peak_balance = self.balance
        self.drawdown = 0.0

        return self._get_state()

    def step(self, action: int):
        """
        Aksiyon uzayı:
           0 => SELL (Full)
           1 => BUY
           2 => HOLD
           3 => PARTIAL SELL
        """

        row = self.df.iloc[self.current_bar]
        close_price = row["Close"]
        reward = 0.0
        done = False

        # 1) Aksiyon Uygula
        if action == 1:
            # BUY => Pozisyon yoksa al
            if not self.has_position:
                # Belirli bir meblağ ile (örn. 30 USDT) alım
                propose_usd = 30.0
                cost = propose_usd
                fee = cost * self.fee_rate
                total_spend = cost + fee
                if self.balance >= total_spend:
                    self.quantity = cost / close_price
                    self.balance -= total_spend
                    self.has_position = True
                    self.entry_price = close_price
                    self.highest_price = close_price
        elif action == 0:
            # SELL => full close
            if self.has_position:
                sell_amt = self.quantity * close_price
                fee = sell_amt * self.fee_rate
                net_amt = sell_amt - fee
                realized_pnl = (close_price - self.entry_price) * self.quantity

                self.balance += net_amt
                self.net_pnl += realized_pnl
                reward += realized_pnl  # realized kar/zar

                # pozisyonu sıfırla
                self.has_position = False
                self.quantity = 0.0
                self.entry_price = 0.0
                self.highest_price = 0.0
        elif action == 3:
            # PARTIAL => mevcudun partial_ratio kadarı sat
            if self.has_position:
                part_qty = self.quantity * self.partial_ratio
                if part_qty > 0:
                    sell_amt = part_qty * close_price
                    fee = sell_amt * self.fee_rate
                    net_amt = sell_amt - fee
                    realized_pnl = (close_price - self.entry_price) * part_qty

                    self.balance += net_amt
                    self.net_pnl += realized_pnl
                    reward += realized_pnl

                    self.quantity -= part_qty
                    if self.quantity <= 1e-8:
                        # Tamamen bitmişse
                        self.has_position = False
                        self.quantity = 0.0
                        self.entry_price = 0.0
                        self.highest_price = 0.0

        # HOLD (2) veya PARTIAL (3) esnasında pozisyon devam edebilir.
        # 2) Trailing Stop Check
        if self.has_position:
            # Highest price update
            if close_price > self.highest_price:
                self.highest_price = close_price

            # Trailing
            stop_price = self.highest_price * self.trailing_stop_factor
            if close_price < stop_price:
                # Stop out => full close
                sell_amt = self.quantity * close_price
                fee = sell_amt * self.fee_rate
                net_amt = sell_amt - fee
                realized_pnl = (close_price - self.entry_price) * self.quantity

                self.balance += net_amt
                self.net_pnl += realized_pnl
                reward += realized_pnl  # trailing reward

                # Pozisyon reset
                self.has_position = False
                self.quantity = 0.0
                self.entry_price = 0.0
                self.highest_price = 0.0
            else:
                # Pozisyon açıkken unrealized PnL üzerinden küçük bir reward
                unrealized_pnl = (close_price - self.entry_price) * self.quantity
                # Kârda isek hafif +, zararda isek hafif - => 0.01 katsayı
                reward += 0.01 * unrealized_pnl

        # 3) drawdown hesapla & ceza
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.drawdown = (self.peak_balance - self.balance) / self.peak_balance
        # Örnek: drawdown arttıkça negatif reward ekleyelim
        drawdown_penalty = -5.0 * self.drawdown  # scale
        reward += drawdown_penalty

        # 4) bar ilerle
        self.current_bar += 1
        if self.current_bar >= (self.n_bars - 1):
            done = True

        next_state = self._get_state()
        return next_state, reward, done

    def _get_state(self):
        """
        Durum (state) => (regime_bin, rsi_bin, stoch_bin, haspos, dd_bin)
        """
        if self.current_bar >= self.n_bars:
            self.current_bar = self.n_bars - 1

        row = self.df.iloc[self.current_bar]

        # regime
        adx_4h = row.get("ADX_4h", 20.0)
        regime = 1 if adx_4h > 25 else 0

        # RSI discretize
        rsi_val = row.get("RSI_5m", 50.0)
        rsi_bin = min(int(rsi_val // (100 / self.rsi_bins)), self.rsi_bins - 1)

        # StochK discretize
        stoch_k = row.get("StochK_5m", 50.0)
        stoch_bin = min(int(stoch_k // (100 / self.stoch_bins)), self.stoch_bins - 1)

        # has_position
        haspos = 1 if self.has_position else 0

        # drawdown bin
        # dd=0 => 0%, dd=0.3 => %30 => bin
        # 0..max_drawdown_allowed => 0..1 => 5 bin
        dd_ratio = min(self.drawdown / self.max_drawdown_allowed, 1.0)
        dd_bin = int(dd_ratio * (self.drawdown_bins - 1))

        return (regime, rsi_bin, stoch_bin, haspos, dd_bin)
