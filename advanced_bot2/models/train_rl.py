# models/train_rl.py

"""
RL => stable-baselines3 PPO
Tam örnek kod:
 - price_data.csv dosyasındaki veriyi yükler
 - NaN/Inf temizleme
 - İsteğe bağlı Normalizasyon (scikit-learn StandardScaler)
 - Basit custom gym env (PriceEnv)
 - PPO ile eğitir ve ppo_agent.zip olarak kaydeder
"""

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# VecNormalize kullanmak isterseniz (isteğe bağlı):
# from stable_baselines3.common.vec_env import VecNormalize
# from sklearn.preprocessing import StandardScaler

class PriceEnv(gym.Env):
    def __init__(self, df: pd.DataFrame):
        super(PriceEnv, self).__init__()

        # Tüm sayısal kolonları seç
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # DataFrame'i sakla, indeks resetle
        self.df = df.reset_index(drop=True)
        self.max_index = len(self.df) - 1

        # Gözlem uzayı: numeric_cols sayısı kadar boyut
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.numeric_cols),),
            dtype=np.float32
        )

        # Aksiyon uzayı (3 discrete aksiyon: 0=SELL,1=BUY,2=HOLD)
        self.action_space = gym.spaces.Discrete(3)

        # Durum değişkenleri
        self.index = 0
        self.has_position = False
        self.entry_price = 0.0

    def reset(self):
        self.index = 0
        self.has_position = False
        self.entry_price = 0.0
        return self._get_obs()

    def _get_obs(self):
        """
        Mevcut index'teki satırdan sadece sayısal kolonları al,
        float32 dizisine çevir ve döndür.
        """
        row = self.df.iloc[self.index][self.numeric_cols]
        obs = row.values.astype(np.float32)
        return obs

    def step(self, action):
        reward = 0.0
        done = False

        row = self.df.iloc[self.index]
        current_price = row["Close"]  # Kullandığınız close sütunu

        if action == 1:  # BUY
            if not self.has_position:
                self.has_position = True
                self.entry_price = current_price

        elif action == 0:  # SELL
            # Sadece pozisyon varsa ve entry_price > 0 ise reward hesapla
            if self.has_position and self.entry_price > 0:
                change = (current_price - self.entry_price) / self.entry_price
                # Ödülü çok büyük tutmamak için clip'leyebilirsiniz
                reward = float(np.clip(change * 100.0, -10, 10))
                self.has_position = False
                self.entry_price = 0.0

        # Zamanı ilerlet
        self.index += 1

        # max_index'e ulaşılırsa episode bitmiş say
        if self.index >= self.max_index:
            done = True
            # final kapanış
            if self.has_position and self.entry_price > 0:
                change = (current_price - self.entry_price) / self.entry_price
                reward += float(np.clip(change * 100.0, -10, 10))
                self.has_position = False

        # Gözlem
        if not done:
            obs = self._get_obs()
            # NaN/Inf kontrolü
            if np.isnan(obs).any() or np.isinf(obs).any():
                print("NaN/Inf tespit edildi! index:", self.index, obs)
                done = True
                # Bir ceza verelim
                reward = -10.0
        else:
            # Episode bitince gözlem 0 ver
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, done, {}

def main():
    # 1) Veri yükleme
    df = pd.read_csv("../data/price_data.csv")

    # 2) NaN/Inf temizleme
    # İstenen numeric sütunlar üzerinde NaN/Inf yoksa bu adım opsiyoneldir,
    # fakat güvenli olması açısından tavsiye edilir.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    # 3) (Opsiyonel) Normalizasyon
    # Eğer veri çok büyük (milyonlar gibi) veya çok farklı ölçeklerdeyse,
    # aşağıdaki gibi normalleştirebilirsiniz.
    #
    # from sklearn.preprocessing import StandardScaler
    # numeric_cols = df.select_dtypes(include=[np.number]).columns
    # scaler = StandardScaler()
    # df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 4) Ortam oluşturma
    env = DummyVecEnv([lambda: PriceEnv(df)])

    # (İsteğe Bağlı) VecNormalize kullanarak gözlem/ödülleri otomatik ölçekleyebilirsiniz:
    # env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 5) Modeli tanımla
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,  # Daha düşük LR, NaN riskini azaltır
        # n_steps=2048, batch_size=64, vs. başka parametreler ayarlanabilir
    )

    # 6) Eğitim
    model.learn(total_timesteps=5000)

    # 7) Kaydet
    model.save("ppo_agent.zip")
    print("Model ppo_agent.zip olarak kaydedildi.")

if __name__ == "__main__":
    main()
