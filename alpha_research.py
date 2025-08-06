# Directory Structure

# alpha_research/
# ├── data_loader.py           # Downloads and prepares features
# ├── regime.py                # Classifies market regimes
# ├── environment.py           # Custom Gym environment with realistic constraints
# ├── agent.py                 # PPO training and inference logic
# ├── evaluation.py            # Backtesting, Sharpe, drawdown, plots
# ├── run.py                   # Main script: training + evaluation loop
# ├── config.py                # Hyperparameters and feature toggles
# └── utils.py                 # Shared tools: turnover calc, reward shaping, etc.

# --- config.py ---

TICKERS = ["MTUM", "VLUE", "QUAL", "USMV"]
BENCHMARK = "SPY"
MACRO_INDICATORS = ["CPIAUCSL", "UNRATE"]  # Optional

FEATURE_WINDOWS = {
    "momentum": [63, 126],
    "volatility": [21],
    "rolling_sharpe": [63],
}

TRAIN_START = "2015-01-01"
TRAIN_END = "2020-12-31"
TEST_START = "2021-01-01"
TEST_END = "2023-12-31"

TRANSACTION_COST = 0.001
MAX_TURNOVER = 0.2

# --- data_loader.py ---

import yfinance as yf
import pandas as pd

def download_etf_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)["Adj Close"]
    return data.pct_change().dropna()

def compute_features(returns, config):
    feats = []
    for window in config["momentum"]:
        feats.append(returns.rolling(window).mean().rename(lambda x: f"mom_{x}"))
    for window in config["volatility"]:
        feats.append(returns.rolling(window).std().rename(lambda x: f"vol_{x}"))
    for window in config["rolling_sharpe"]:
        roll = returns.rolling(window)
        feats.append(((roll.mean() / roll.std()) * (252**0.5)).rename(lambda x: f"sharpe_{x}"))
    return pd.concat(feats, axis=1).dropna()

# --- regime.py ---

def classify_regimes(market_returns, window_vol=21, window_mom=63):
    vol = market_returns.rolling(window_vol).std()
    mom = market_returns.rolling(window_mom).mean()
    regime = 2 * (vol > vol.median()).astype(int) + (mom > 0).astype(int)
    return regime.fillna(method='ffill').astype(int)

# --- environment.py ---

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FactorEnv(gym.Env):
    def __init__(self, features, factor_returns, regimes, config):
        self.features = features
        self.factor_returns = factor_returns
        self.regimes = regimes
        self.window = 1
        self.transaction_cost = config["TRANSACTION_COST"]
        self.max_turnover = config["MAX_TURNOVER"]

        self.action_space = spaces.Box(low=-1, high=1, shape=(factor_returns.shape[1],), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(features.shape[1] + 1,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        self.step_idx = 0
        self.weights = np.zeros(self.factor_returns.shape[1])
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.append(self.features.iloc[self.step_idx].values, self.regimes.iloc[self.step_idx])
        return obs.astype(np.float32)

    def step(self, action):
        action = np.clip(action, -1, 1)
        turnover = np.sum(np.abs(action - self.weights))
        if turnover > self.max_turnover:
            action = self.weights + (action - self.weights) * (self.max_turnover / turnover)

        reward = np.dot(action, self.factor_returns.iloc[self.step_idx].values) - self.transaction_cost * turnover
        self.weights = action.copy()

        self.step_idx += 1
        self.done = self.step_idx >= len(self.factor_returns)
        return self._get_obs(), reward, self.done, False, {}

# --- agent.py ---

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

def train_agent(env_fn, total_timesteps=20000):
    vec_env = DummyVecEnv([env_fn])
    model = PPO("MlpPolicy", vec_env, verbose=0)
    model.learn(total_timesteps=total_timesteps)
    return model

def run_inference(model, env):
    obs, _ = env.reset()
    done = False
    rewards, weights = [], []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        rewards.append(reward)
        weights.append(action)
    return np.array(rewards), np.array(weights)

# --- evaluation.py ---

import numpy as np
import matplotlib.pyplot as plt

def sharpe(returns):
    return np.mean(returns) / np.std(returns) * np.sqrt(252)

def max_drawdown(returns):
    cum = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum)
    return np.min((cum - peak) / peak)

def plot_results(rewards):
    cum = np.cumsum(rewards)
    plt.plot(cum)
    plt.title("Cumulative Return")
    plt.xlabel("Step")
    plt.ylabel("Cumulative Reward")
    plt.show()

# --- run.py ---

from config import *
from data_loader import download_etf_data, compute_features
from regime import classify_regimes
from environment import FactorEnv
from agent import train_agent, run_inference
from evaluation import sharpe, max_drawdown, plot_results

returns = download_etf_data(TICKERS + [BENCHMARK], TRAIN_START, TEST_END)
factor_returns = returns[TICKERS]
market_returns = returns[BENCHMARK]

regimes = classify_regimes(market_returns)
features = compute_features(factor_returns, FEATURE_WINDOWS)

# Align
min_len = min(len(features), len(factor_returns), len(regimes))
factor_returns = factor_returns.iloc[-min_len:]
features = features.iloc[-min_len:]
regimes = regimes.iloc[-min_len:]

# Split
split = int(0.8 * len(features))
train_env_fn = lambda: FactorEnv(features.iloc[:split], factor_returns.iloc[:split], regimes.iloc[:split], globals())
test_env = FactorEnv(features.iloc[split:], factor_returns.iloc[split:], regimes.iloc[split:], globals())

# Train and Evaluate
model = train_agent(train_env_fn)
rewards, _ = run_inference(model, test_env)
print("Sharpe:", sharpe(rewards))
print("Max Drawdown:", max_drawdown(rewards))
plot_results(rewards)
