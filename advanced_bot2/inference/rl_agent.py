# inference/rl_agent.py

import random
import math
import json

class RLAgent:
    """
    Basit Q-öğrenme yaklaşımıyla, online RL pseudo-kodu.
    """

    def __init__(self,
                 alpha=0.1,
                 gamma=0.99,
                 epsilon=0.05,
                 max_stop_atr=5.0):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.max_stop_atr = max_stop_atr

        self.policy_dict = {
            "stop_atr_mult": 2.0,
            "partial_levels": [0.03, 0.06, 0.1],
            "partial_ratio": 0.3
        }

        self.q_table = {}
        self.step_count = 0
        self.action_space = self._generate_action_space()

    def _generate_action_space(self):
        # Basit discrete action uzayı
        stops = [round(x*0.5, 1) for x in range(2, 11)]  # 1.0..5.0 step=0.5
        ratios = [0.2, 0.3, 0.4, 0.5]
        partial_set = [
            [0.03, 0.06, 0.1],
            [0.02, 0.04, 0.08],
            [0.05, 0.1, 0.15]
        ]
        actions = []
        for s in stops:
            for r in ratios:
                for lvls in partial_set:
                    actions.append({
                        "stop_atr_mult": s,
                        "partial_levels": lvls,
                        "partial_ratio": r
                    })
        return actions

    def get_policy(self) -> dict:
        return self.policy_dict

    def observe_environment(self, features: dict) -> tuple:
        # Basit state => (symbol, regime)
        symbol = features.get("symbol","BTCUSDT")
        regime = features.get("regime","RANGE")
        return (symbol, regime)

    def _state_action_key(self, state, action) -> tuple:
        # action => freeze
        action_key = (
            action["stop_atr_mult"],
            action["partial_ratio"],
            tuple(action["partial_levels"])
        )
        return (state, action_key)

    def _get_q_value(self, state, action) -> float:
        key = self._state_action_key(state, action)
        return self.q_table.get(key, 0.0)

    def _set_q_value(self, state, action, value: float):
        key = self._state_action_key(state, action)
        self.q_table[key] = value

    def select_action(self, obs: tuple) -> dict:
        self.step_count += 1
        # Epsilon-greedy
        if random.random()< self.epsilon:
            chosen = random.choice(self.action_space)
            return chosen
        else:
            best_action = None
            best_q = float("-inf")
            for act in self.action_space:
                qv = self._get_q_value(obs, act)
                if qv> best_q:
                    best_q= qv
                    best_action= act
            if best_action is None:
                best_action= random.choice(self.action_space)
            return best_action

    def update(self, obs: tuple, action: dict, reward: float, next_obs: tuple):
        old_q = self._get_q_value(obs, action)
        # find max q next
        max_q_next= float("-inf")
        for a2 in self.action_space:
            qv= self._get_q_value(next_obs,a2)
            if qv> max_q_next:
                max_q_next= qv
        if max_q_next==float("-inf"):
            max_q_next= 0.0

        new_q = old_q + self.alpha*(reward + self.gamma*max_q_next - old_q)
        self._set_q_value(obs, action, new_q)

        # Basit negative reward => policy param ayarlama
        if reward<0:
            self.policy_dict["stop_atr_mult"] = min(action["stop_atr_mult"] + 0.05, self.max_stop_atr)

    def print_q_table(self, limit=10):
        count=0
        for k,v in self.q_table.items():
            print(k, "=>",v)
            count+=1
            if count>=limit:
                break


    def decay_epsilon(self, decay_rate=0.99):
        self.epsilon= self.epsilon* decay_rate
        if self.epsilon<0.01:
            self.epsilon=0.01

    def save_qtable(self, filepath="q_table.json"):

        # Geçici bir sözlük oluşturuyoruz.
        serial_dict = {}

        for (state, action_tuple), q_value in self.q_table.items():
            # state örn: ("SOLUSDT", "RANGE")
            # action_tuple örn: (1.5, 0.2, (0.03,0.06,0.1))

            # Bu tuple'ları string'e dönüştürelim:
            # Basit bir yol => repr(...) veya f-string.
            # Ama parsing için "ayraç bazlı" da yapabilirsiniz.
            
            # 1) state_str
            state_str = f"{state[0]}|{state[1]}"
            # 2) partial_levels -> '-'.join(...) 
            stop_atr_mult, partial_ratio, partial_levels = action_tuple
            pl_str = '-'.join(map(str, partial_levels))
            action_str = f"{stop_atr_mult}|{partial_ratio}|{pl_str}"

            # Tek key string yapıyoruz
            key_str = f"{state_str}:::{action_str}"

            serial_dict[key_str] = q_value

        # Artık serial_dict'in anahtarları string => json'a yazabiliriz.
        with open(filepath,"w") as f:
            json.dump(serial_dict, f, indent=2)

        print(f"Q-table saved to {filepath}.")
    # def load_qtable(self, filepath="q_table.json"):
    #     with open(filepath,"r") as f:
    #         serial_dict = json.load(f)  # string -> float

    #     self.q_table = {}

    #     for key_str, q_value in serial_dict.items():
    #         # key_str örn: "SOLUSDT|TREND:::1.5|0.2|0.03-0.06-0.1"
    #         state_part, action_part = key_str.split(":::")

    #         # state_part => "SOLUSDT|TREND"
    #         s_symbol, s_regime = state_part.split("|")
    #         state = (s_symbol, s_regime)

    #         # action_part => "1.5|0.2|0.03-0.06-0.1"
    #         a_split = action_part.split("|")  # ["1.5","0.2","0.03-0.06-0.1"]
    #         stop_atr_mult = float(a_split[0])
    #         partial_ratio = float(a_split[1])
    #         plvls_str = a_split[2].split("-")  # ["0.03","0.06","0.1"]
    #         partial_levels = tuple(map(float, plvls_str))

    #         action_tuple = (stop_atr_mult, partial_ratio, partial_levels)

    #         self.q_table[(state, action_tuple)] = q_value

    #     print(f"Q-table loaded from {filepath}. Total entries: {len(self.q_table)}")
