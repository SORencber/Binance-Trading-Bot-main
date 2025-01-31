# offline_training_scripts/train_qlearning.py

import pandas as pd
from offline_env import OfflineEnv
from inference.rl_agent import RLAgent

def train_offline_q(
    csv_path="data/price_data.csv",
    n_episodes=200,
    alpha=0.1,
    gamma=0.99,
    epsilon = 0.9,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    initial_balance=1000.0,
    fee_rate=0.001,
    partial_ratio=0.3,
    trailing_stop_pct=0.95,
    max_drawdown_allowed=0.3
):
    # 1) CSV veriyi yükle
    df = pd.read_csv(csv_path)
    # df["Datetime"] = pd.to_datetime(df["Datetime"])
    # df.sort_values("Datetime", inplace=True)

    # 2) Environment
    env = OfflineEnv(
        df,
        initial_balance=initial_balance,
        fee_rate=fee_rate,
        max_bars=None,  # isterseniz kısıtlayabilirsiniz
        partial_ratio=partial_ratio,
        trailing_stop_pct=trailing_stop_pct,
        max_drawdown_allowed=max_drawdown_allowed
    )

    # 3) RLAgent
    agent = RLAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        state_shape=(2, 5, 5, 2, 5),
        n_actions=4
    )

    # 4) Eğitim döngüsü
    for ep in range(n_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, next_state, done)
            state = next_state

            total_reward += reward
            steps += 1

        agent.decay_epsilon()

        # Episode log
        print(f"[Episode {ep+1}/{n_episodes}] steps={steps}, "
              f"final_balance={env.balance:.2f}, netPnL={env.net_pnl:.2f}, "
              f"drawdown={env.drawdown:.2%}, total_reward={total_reward:.2f}, "
              f"epsilon={agent.epsilon:.3f}")

    # 5) Q tabloyu kaydet
    agent.save_qtable("q_table.npy")
    print("Training done => 'q_table.npy' saved.")

if __name__=="__main__":
    # Örnek kullanım
    train_offline_q(
        csv_path="data/price_data2.csv",
        n_episodes=100,
        alpha=0.05,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        initial_balance=1000.0
    )
