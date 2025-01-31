import pandas as pd
import numpy as np
import random
import json
from datetime import datetime
import math

class OfflineEnv:
    """
    Offline (CSV tabanlı) bir ortam (environment).
    Fiyat verisini bar bar ilerleterek agent'tan gelen aksiyonları uygular.
    """

    def __init__(self, df: pd.DataFrame, initial_balance=200.0, fee_rate=0.001):
        """
        :param df: Fiyat/indikatör verisi DataFrame
        :param initial_balance: Başlangıç bakiyesi (USDT)
        :param fee_rate: Trade komisyon oranı (örn. 0.001 = %0.1)
        """
        self.df = df.reset_index(drop=True)
        self.n_bars = len(df)

        self.initial_balance = initial_balance
        self.fee_rate = fee_rate
        
        # Env durumu
        self.current_bar = 0
        self.balance = initial_balance

        # Pozisyon bilgileri
        self.has_position = False
        self.quantity = 0.0
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.net_pnl = 0.0
        self.partial_done = []

    def reset(self):
        """
        Yeni episode başlat. Tüm değişkenleri sıfırla.
        """
        self.current_bar = 0
        self.balance = self.initial_balance
        self.has_position = False
        self.quantity = 0.0
        self.entry_price = 0.0
        self.highest_price = 0.0
        self.net_pnl = 0.0
        self.partial_done = []

        return self._get_state()

    def _get_state(self):
        """
        Environment'ın "state" tanımı:
          - ADX tabanlı 'regime' (TREND vs RANGE)
          - Kısa vadeli bir skor (RSI, Stoch vs.)
          - has_position
          - floating_pnl (oransal)
          - 4h MACD > 0 mu?
          (Bu gibi ek parametrelerle durumu zenginleştirebiliriz.)
        """
        row = self.df.iloc[self.current_bar]
        close_price = row["Close"]

        # 1) ADX'e göre Trend vs Range
        adx_4h = row.get("ADX_4h", 20.0)
        regime = "TREND" if adx_4h > 25 else "RANGE"

        # 2) Kısa vadeli skor (örnek: RSI_5m > 60 + StochK_5m >80 => +1, tersi => -1)
        rsi_5m = row.get("RSI_5m", 50.0)
        stoch_5m = row.get("StochK_5m", 50.0)
        short_sco = 0
        if rsi_5m > 60 and stoch_5m > 80:
            short_sco = 1
        elif rsi_5m < 40 and stoch_5m < 20:
            short_sco = -1

        # 3) has_position
        pos_flag = 1 if self.has_position else 0

        # 4) floating_pnl (yuvarlanmış)
        floating_pnl = 0.0
        if self.has_position and self.entry_price > 1e-9:
            floating_pnl = (close_price - self.entry_price)/self.entry_price
        floating_pnl = round(floating_pnl, 2)

        # 5) MACD_4h > 0 => 1, else 0
        macd_4h = row.get("MACD_4h", 0.0)
        macd_flag = 1 if macd_4h > 0 else 0

        # Tuple olarak döndürüyoruz. (String, int, int, float, int) 
        return (regime, short_sco, pos_flag, floating_pnl, macd_flag)

    def step(self, action_dict):
        """
        Ajan'ın seçtiği aksiyonu uygular:
         action_dict = {
           "trade_type": 0/1/2 (0=Hold,1=Buy,2=Sell),
           "stop_atr_mult": float,
           "partial_ratio": float,
           "partial_levels": [...],
           ...
         }
        Return: (next_state, reward, done)
        """
        row = self.df.iloc[self.current_bar]
        close_price = row["Close"]
        old_net_pnl = self.net_pnl  # reward hesabı için

        # Aksiyon tipine göre işlem
        ttype = action_dict["trade_type"]  # 0=Hold,1=Buy,2=Sell

        if ttype == 1:  # BUY
            if not self.has_position: 
                # Örnek sabit 30 USDT'lik alım
                propose_usd = 30.0
                if self.balance >= propose_usd:
                    fee = propose_usd*self.fee_rate
                    total_spend = propose_usd + fee
                    if self.balance >= total_spend:
                        self.has_position = True
                        self.entry_price = close_price
                        self.highest_price = close_price
                        self.quantity = propose_usd/close_price
                        self.balance -= total_spend
                        # partial satışı reset
                        self.partial_done = [False]*len(action_dict["partial_levels"])

        elif ttype == 2:  # SELL => full kapat
            if self.has_position:
                realized = (close_price - self.entry_price)*self.quantity
                sell_amt = close_price*self.quantity
                fee = sell_amt*self.fee_rate
                net_sell = sell_amt - fee

                self.balance += net_sell
                self.net_pnl += realized

                # Pozisyon kapat
                self.has_position = False
                self.quantity = 0.0
                self.entry_price = 0.0
                self.highest_price = 0.0
                self.partial_done = []

        # ttype=0 => Hold, ama risk yönetimi (partial, trailing) devrede
        partial_gain = 0.0
        if self.has_position:
            partial_gain = self.handle_risk_management(close_price, action_dict)

        # Reward = net_pnl değişimi
        new_net_pnl = self.net_pnl
        reward = new_net_pnl - old_net_pnl
        # partial_gain isterseniz reward'a eklenebilir, ama gain zaten net_pnl'e ekleniyor => double count yapmayın.

        # Zaman ilerlet, done?
        self.current_bar += 1
        done = (self.current_bar >= self.n_bars-1)

        next_state = self._get_state()
        return next_state, reward, done

    def handle_risk_management(self, close_price, action_dict):
        """
        Kademeli (partial) satış ve trailing stop.
        """
        gain = 0.0

        # 1) highest_price güncelle
        if close_price > self.highest_price:
            self.highest_price = close_price

        # 2) partial satış
        gain_ratio = 0
        if self.entry_price > 0:
            gain_ratio = (close_price - self.entry_price)/self.entry_price

        for i, lvl in enumerate(action_dict["partial_levels"]):
            if i < len(self.partial_done):
                if (not self.partial_done[i]) and (gain_ratio > lvl):
                    # partial_ratio kadar satış
                    part_qty = self.quantity*action_dict["partial_ratio"]
                    if part_qty>0:
                        realized = (close_price - self.entry_price)*part_qty
                        sell_amt = close_price*part_qty
                        fee = sell_amt*self.fee_rate
                        net_sell = sell_amt - fee

                        self.balance += net_sell
                        self.quantity -= part_qty
                        self.net_pnl += realized
                        gain += realized

                        self.partial_done[i] = True
                        # Pozisyon 0 olduysa
                        if self.quantity<=0:
                            self.has_position = False
                            self.quantity = 0.0
                            self.entry_price = 0.0
                            self.highest_price = 0.0
                            self.partial_done = []
                            break

        # 3) trailing stop
        stop_mult = action_dict["stop_atr_mult"]
        # Basit ATR => price*0.02 => mesafe
        atr_dist = 0.02*close_price
        trailing_stop = self.highest_price - atr_dist*stop_mult
        if close_price < trailing_stop and self.has_position:
            realized = (close_price - self.entry_price)*self.quantity
            sell_amt = close_price*self.quantity
            fee = sell_amt*self.fee_rate
            net_sell = sell_amt - fee

            self.balance += net_sell
            self.net_pnl += realized
            gain += realized

            # Pozisyon kapansın
            self.has_position = False
            self.quantity = 0.0
            self.entry_price = 0.0
            self.highest_price = 0.0
            self.partial_done = []

        return gain


class RLAgent:
    """
    Tabular Q-learning ajanı.
    State: (regime, short_sco, pos_flag, float_pnl, macd_flag) => string'e dönüştürerek saklarız.
    Action: 
      trade_type in [0,1,2], 
      stop_atr_mult in [1.0..5.0 adım 0.5], 
      partial_ratio in [0.2, 0.3, 0.4, 0.5],
      partial_levels in [[0.03,0.06,0.1],[0.02,0.04,0.08],[0.05,0.1,0.15]]
    """

    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q tablosu => {(state_str, action_tup): q_value}
        self.q_table = {}

        # Action space'i oluştur
        self.action_space = self._create_action_space()

    def _create_action_space(self):
        trade_types = [0,1,2]
        stops = [round(0.5*x,1) for x in range(2,11)]  # 1.0..5.0
        ratios = [0.2,0.3,0.4,0.5]
        pls = [
            [0.03,0.06,0.1],
            [0.02,0.04,0.08],
            [0.05,0.1,0.15]
        ]
        actions=[]
        for t in trade_types:
            for s in stops:
                for r in ratios:
                    for pset in pls:
                        actions.append({
                            "trade_type": t,
                            "stop_atr_mult": s,
                            "partial_ratio": r,
                            "partial_levels": pset
                        })
        return actions

    def _state_to_str(self, state):
        # state => (regime, short_sco, pos_flag, float_pnl, macd_flag)
        return f"{state[0]}|{state[1]}|{state[2]}|{state[3]}|{state[4]}"

    def _action_to_tuple(self, adict):
        # dict'i tuple'a dönüştür
        return (
            adict["trade_type"],
            adict["stop_atr_mult"],
            adict["partial_ratio"],
            tuple(adict["partial_levels"])
        )

    def _get_q(self, s_str, a_tup):
        return self.q_table.get((s_str,a_tup), 0.0)

    def _set_q(self, s_str, a_tup, val):
        self.q_table[(s_str,a_tup)] = val

    def select_action(self, state):
        """
        Epsilon-greedy eylem seçimi
        """
        s_str = self._state_to_str(state)
        if random.random() < self.epsilon:
            return random.choice(self.action_space)
        else:
            best_q = float("-inf")
            best_a = None
            for act in self.action_space:
                a_tup = self._action_to_tuple(act)
                qv = self._get_q(s_str, a_tup)
                if qv> best_q:
                    best_q = qv
                    best_a = act
            if best_a is None:
                best_a = random.choice(self.action_space)
            return best_a

    def update(self, state, action, reward, next_state):
        """
        Q-learning update
        """
        s_str = self._state_to_str(state)
        a_tup = self._action_to_tuple(action)
        old_q = self._get_q(s_str, a_tup)

        # next_state max Q
        ns_str = self._state_to_str(next_state)
        max_q_next = float("-inf")
        for act2 in self.action_space:
            at2 = self._action_to_tuple(act2)
            qv2 = self._get_q(ns_str,at2)
            if qv2> max_q_next:
                max_q_next = qv2
        if max_q_next == float("-inf"):
            max_q_next = 0.0

        new_q = old_q + self.alpha*(reward + self.gamma*max_q_next - old_q)
        self._set_q(s_str, a_tup, new_q)

    def decay_epsilon(self, decay=0.95, min_eps=0.01):
        self.epsilon = max(self.epsilon*decay, min_eps)

    def save_q_table(self, filepath="q_table.json"):
        """
        Basit bir JSON formatı
        """
        ser = {}
        for (st,act), qv in self.q_table.items():
            # act => (type,stop,ratio,(...levels...))
            act_str = f"{act[0]}|{act[1]}|{act[2]}|{'-'.join(map(str,act[3]))}"
            key_str = f"{st}:::{act_str}"
            ser[key_str] = qv
        with open(filepath,"w") as f:
            json.dump(ser,f,indent=2)
        print(f"Q-table saved to {filepath}")

    def load_q_table(self, filepath="q_table.json"):
        with open(filepath,"r") as f:
            ser = json.load(f)
        self.q_table={}
        for k,v in ser.items():
            st_part,act_part = k.split(":::")
            # act_part => "type|stop|ratio|0.03-0.06-0.1"
            ap = act_part.split("|")
            tt = int(ap[0])
            stp = float(ap[1])
            rr = float(ap[2])
            pl_strs = ap[3].split("-")
            pl_floats = tuple(map(float, pl_strs))
            self.q_table[(st_part, (tt,stp,rr,pl_floats))] = v
        print(f"Loaded Q-table from {filepath}, total: {len(self.q_table)}")


def train_offline_q():
    """
    Eğitim fonksiyonu:
    1) CSV'yi yükle
    2) Env oluştur
    3) RLAgent oluştur
    4) Episode döngüsü
    5) Sonuç kaydet
    """
    # 1) CSV oku
    df = pd.read_csv("data/price_data.csv")
    # df["Datetime"] = pd.to_datetime(df["Datetime"])  # Gerekirse
    # df.sort_values("Datetime", inplace=True)          # Gerekirse

    # 2) Env, Agent
    env = OfflineEnv(df, initial_balance=200.0, fee_rate=0.001)
    agent = RLAgent(alpha=0.1, gamma=0.99, epsilon=0.2)

    # 3) Eğitim
    num_episodes = 100
    all_rewards=[]
    for ep in range(num_episodes):
        state = env.reset()
        done=False
        ep_reward=0.0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            ep_reward += reward
            state = next_state
        
        agent.decay_epsilon(decay=0.95, min_eps=0.01)
        all_rewards.append(ep_reward)
        print(f"Ep={ep}, Reward={ep_reward:.2f}, PnL={env.net_pnl:.2f}, Bal={env.balance:.2f}, eps={agent.epsilon:.3f}")

    # 4) Kaydet
    agent.save_q_table("q_table.json")
    pd.DataFrame({"episode":range(num_episodes),"reward":all_rewards}).to_csv("train_report.csv",index=False)
    print("Training done. train_report.csv & q_table.json saved.")


if __name__=="__main__":
    train_offline_q()
