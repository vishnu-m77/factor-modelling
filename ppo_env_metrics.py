import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import mean_squared_error

# --- Market & Factor Setup ---
def download_data(tickers, start="2015-01-01", end="2023-12-31"):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

# --- Regime Classifier ---
def classify_regimes(market_returns, window_vol=21, window_mom=63):
    vol = market_returns.rolling(window_vol).std()
    mom = market_returns.rolling(window_mom).mean()
    regime = 2 * (vol > vol.median()).astype(int) + (mom > 0).astype(int)
    return regime.fillna(method='ffill').astype(int)

# --- Custom Gym Environment ---
class FactorPPOEnv(Env):
    def __init__(self, factor_returns, regimes, window=5, transaction_cost=0.001):
        super(FactorPPOEnv, self).__init__()
        self.factor_returns = factor_returns
        self.regimes = regimes
        self.window = window
        self.transaction_cost = transaction_cost

        self.n_factors = factor_returns.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window * self.n_factors + 1,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_factors,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window
        self.done = False
        self.last_weights = np.zeros(self.n_factors)
        return self._get_obs(), {}

    def _get_obs(self):
        windowed_returns = self.factor_returns.iloc[self.current_step - self.window:self.current_step].values.flatten()
        regime = [self.regimes.iloc[self.current_step]]
        return np.concatenate([windowed_returns, regime]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        turnover = np.sum(np.abs(action - self.last_weights))
        cost = self.transaction_cost * turnover

        returns = self.factor_returns.iloc[self.current_step].values
        reward = np.dot(action, returns) - cost

        self.last_weights = action
        self.current_step += 1
        self.done = self.current_step >= len(self.factor_returns)

        return self._get_obs(), reward, self.done, False, {}

# --- Evaluation Metrics ---
def sharpe_ratio(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

def max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown.min()

# --- Training & Evaluation ---
if __name__ == "__main__":
    tickers = ['MTUM', 'VLUE', 'QUAL']
    factor_returns = download_data(tickers)
    market_returns = download_data(['SPY'])['SPY']
    regimes = classify_regimes(market_returns)

    # Split into train/test
    split_idx = int(0.8 * len(factor_returns))
    train_env = DummyVecEnv([lambda: FactorPPOEnv(factor_returns.iloc[:split_idx], regimes.iloc[:split_idx])])
    test_env = FactorPPOEnv(factor_returns.iloc[split_idx:], regimes.iloc[split_idx:])

    model = PPO("MlpPolicy", train_env, verbose=1)
    model.learn(total_timesteps=10000)

    # Evaluate
    obs, _ = test_env.reset()
    rewards = []
    weights = []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = test_env.step(action)
        rewards.append(reward)
        weights.append(action)

    rewards = np.array(rewards)
    print("Sharpe Ratio:", sharpe_ratio(rewards))
    print("Max Drawdown:", max_drawdown(rewards))

    plt.plot(np.cumsum(rewards))
    plt.title("PPO Cumulative Return")
    plt.xlabel("Steps")
    plt.ylabel("Cumulative Reward")
    plt.show()