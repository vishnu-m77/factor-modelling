import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import Env, spaces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qlearner import QLearner

import warnings
warnings.filterwarnings("ignore")

### STEP 1: Download Factor ETF Data (e.g. MTUM, VLUE, QUAL)
def download_etf_data(tickers, start="2015-01-01", end="2023-12-31"):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

### STEP 2: Classify Regimes Using Simple Volatility + Momentum Rule
def classify_regimes(market_returns, window_vol=21, window_mom=63):
    vol = market_returns.rolling(window_vol).std()
    mom = market_returns.rolling(window_mom).mean()

    # Regime: 0 = Low vol, Low mom
    #         1 = High vol, Low mom
    #         2 = Low vol, High mom
    #         3 = High vol, High mom
    regime = 2 * (vol > vol.median()).astype(int) + (mom > 0).astype(int)
    return regime.fillna(method='ffill').astype(int)

### STEP 3: Define a Simple Gym-like Environment
class FactorEnv(Env):
    def __init__(self, factor_returns, regimes, window=5):
        super(FactorEnv, self).__init__()
        self.factor_returns = factor_returns
        self.regimes = regimes
        self.window = window

        self.action_space = spaces.Discrete(3**factor_returns.shape[1])  # 3 actions per factor: under, neutral, over
        self.observation_space = spaces.Discrete(10000)  # for tabular Q-learning

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window
        self.done = False
        self.weights = np.zeros(self.factor_returns.shape[1])  # Neutral to all factors
        return self._get_obs(), {}

    def _get_obs(self):
        recent_returns = self.factor_returns.iloc[self.current_step - self.window:self.current_step].values.flatten()
        regime = [self.regimes.iloc[self.current_step]]
        state = np.concatenate([recent_returns, regime])
        key = hash(tuple(np.round(state, 2))) % self.observation_space.n
        return key

    def _decode_action(self, action):
        base = 3
        decoded = []
        for _ in range(self.factor_returns.shape[1]):
            decoded.append((action % base) - 1)  # -1, 0, +1
            action //= base
        return np.array(decoded)

    def step(self, action):
        delta = self._decode_action(action)
        self.weights = np.clip(self.weights + delta * 0.1, -1, 1)  # rebalance up to +/-10%

        returns = self.factor_returns.iloc[self.current_step]
        reward = np.dot(self.weights, returns)

        self.current_step += 1
        if self.current_step >= len(self.factor_returns):
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

### MAIN SCRIPT
if __name__ == "__main__":
    tickers = ['MTUM', 'VLUE', 'QUAL']
    factor_returns = download_etf_data(tickers)
    market_returns = download_etf_data(['SPY'])['SPY']
    regimes = classify_regimes(market_returns)

    env = FactorEnv(factor_returns, regimes)
    learner = QLearner(num_states=10000, num_actions=env.action_space.n, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0)

    episodes = 10
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        action = learner.querysetstate(state)

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            action = learner.query(next_state, reward)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.4f}")

    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
import numpy as np
import pandas as pd
import yfinance as yf
from gymnasium import Env, spaces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from qlearner import QLearner

import warnings
warnings.filterwarnings("ignore")

### STEP 1: Download Factor ETF Data (e.g. MTUM, VLUE, QUAL)
def download_etf_data(tickers, start="2015-01-01", end="2023-12-31"):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    returns = data.pct_change().dropna()
    return returns

### STEP 2: Classify Regimes Using Simple Volatility + Momentum Rule
def classify_regimes(market_returns, window_vol=21, window_mom=63):
    vol = market_returns.rolling(window_vol).std()
    mom = market_returns.rolling(window_mom).mean()

    # Regime: 0 = Low vol, Low mom
    #         1 = High vol, Low mom
    #         2 = Low vol, High mom
    #         3 = High vol, High mom
    regime = 2 * (vol > vol.median()).astype(int) + (mom > 0).astype(int)
    return regime.fillna(method='ffill').astype(int)

### STEP 3: Define a Simple Gym-like Environment
class FactorEnv(Env):
    def __init__(self, factor_returns, regimes, window=5):
        super(FactorEnv, self).__init__()
        self.factor_returns = factor_returns
        self.regimes = regimes
        self.window = window

        self.action_space = spaces.Discrete(3**factor_returns.shape[1])  # 3 actions per factor: under, neutral, over
        self.observation_space = spaces.Discrete(10000)  # for tabular Q-learning

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window
        self.done = False
        self.weights = np.zeros(self.factor_returns.shape[1])  # Neutral to all factors
        return self._get_obs(), {}

    def _get_obs(self):
        recent_returns = self.factor_returns.iloc[self.current_step - self.window:self.current_step].values.flatten()
        regime = [self.regimes.iloc[self.current_step]]
        state = np.concatenate([recent_returns, regime])
        key = hash(tuple(np.round(state, 2))) % self.observation_space.n
        return key

    def _decode_action(self, action):
        base = 3
        decoded = []
        for _ in range(self.factor_returns.shape[1]):
            decoded.append((action % base) - 1)  # -1, 0, +1
            action //= base
        return np.array(decoded)

    def step(self, action):
        delta = self._decode_action(action)
        self.weights = np.clip(self.weights + delta * 0.1, -1, 1)  # rebalance up to +/-10%

        returns = self.factor_returns.iloc[self.current_step]
        reward = np.dot(self.weights, returns)

        self.current_step += 1
        if self.current_step >= len(self.factor_returns):
            self.done = True

        return self._get_obs(), reward, self.done, False, {}

### MAIN SCRIPT
if __name__ == "__main__":
    tickers = ['MTUM', 'VLUE', 'QUAL']
    factor_returns = download_etf_data(tickers)
    market_returns = download_etf_data(['SPY'])['SPY']
    regimes = classify_regimes(market_returns)

    env = FactorEnv(factor_returns, regimes)
    learner = QLearner(num_states=10000, num_actions=env.action_space.n, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0)

    episodes = 10
    rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        action = learner.querysetstate(state)

        while not done:
            next_state, reward, done, _, _ = env.step(action)
            action = learner.query(next_state, reward)
            total_reward += reward

        rewards.append(total_reward)
        print(f"Episode {ep+1}: Total Reward = {total_reward:.4f}")

    plt.plot(rewards)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
