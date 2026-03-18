"""
train.py
--------
End-to-end training script for the RL pairs trader.

Steps
~~~~~
1. Build a historical training dataset (prices + cointegration + z-score + vol).
2. Validate the Gymnasium environment with SB3's built-in checker.
3. Train a PPO agent on that dataset.
4. Save the trained model to disk.

Usage
~~~~~
    cd /home/rshamostaani/Financial_experiments
    python -m trader.train

Adjust SYMBOL_A, SYMBOL_B, PERIOD, and TOTAL_TIMESTEPS below to your needs.
"""

import os
import sys
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

# Add repo root so sibling modules (arbit_gpu_vectorized, calc_z) are importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from trader.data_pipeline import build_training_data
from trader.agentClass import ArbitrageTradingEnv


# ============================================================================ #
#  Configuration — edit these to change the experiment
# ============================================================================ #

SYMBOL_A   = "AMD"
SYMBOL_B   = "NVDA"

# yfinance history window.  Intraday intervals are capped at 60 days by Yahoo.
PERIOD     = "60d"
INTERVAL   = "1h"

# Rolling window length for cointegration and z-score (in bars).
COINT_WINDOW = 20

# Starting capital per episode (USD).
INITIAL_CASH = 10_000.0

# Max fraction of balance the agent can deploy in a single direction.
MAX_POSITION_FRACTION = 0.5

# Total environment steps across all training episodes.
# Each episode = one full pass through the dataset.
# 200 000 steps over ~600 bars ≈ 333 episodes.
TOTAL_TIMESTEPS = 200_000

# Where to save the trained model.
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "arbitrage_rl_model")


# ============================================================================ #
#  Main
# ============================================================================ #

def main():

    # ---------------------------------------------------------------------- #
    # 1.  Build the training dataset
    # ---------------------------------------------------------------------- #
    print("=" * 60)
    print("Step 1 — Building training dataset")
    print("=" * 60)

    data = build_training_data(
        symbol_a     = SYMBOL_A,
        symbol_b     = SYMBOL_B,
        period       = PERIOD,
        interval     = INTERVAL,
        coint_window = COINT_WINDOW,
    )

    print(f"\nDataset shape : {data.shape}")
    print(f"Columns       : {list(data.columns)}")
    print(f"Date range    : {data.index[0]}  →  {data.index[-1]}\n")

    # ---------------------------------------------------------------------- #
    # 2.  Instantiate and validate the environment
    # ---------------------------------------------------------------------- #
    print("=" * 60)
    print("Step 2 — Instantiating and validating the environment")
    print("=" * 60)

    env = ArbitrageTradingEnv(
        data_feed             = data,
        initial_cash          = INITIAL_CASH,
        max_position_fraction = MAX_POSITION_FRACTION,
    )

    # SB3's check_env runs a handful of sanity checks (observation/action
    # shapes, dtype correctness, reset/step API compliance, etc.).
    print("Running SB3 environment check...")
    check_env(env, warn=True)
    print("Environment check passed.\n")

    # ---------------------------------------------------------------------- #
    # 3.  Define the PPO model
    # ---------------------------------------------------------------------- #
    print("=" * 60)
    print("Step 3 — Configuring PPO")
    print("=" * 60)

    # Small network to protect VRAM shared with the LLM.
    # pi = policy (actor) layers, vf = value function (critic) layers.
    tiny_net_kwargs = dict(
        activation_fn = torch.nn.ReLU,
        net_arch      = dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        policy        = "MlpPolicy",
        env           = env,
        policy_kwargs = tiny_net_kwargs,
        learning_rate = 3e-4,
        # n_steps: number of env steps collected per policy update.
        # Smaller → more frequent updates, lower VRAM per batch.
        n_steps       = 512,
        # batch_size: mini-batch size for gradient updates.
        batch_size    = 64,
        # n_epochs: number of gradient steps per collected batch.
        n_epochs      = 10,
        # gamma: discount factor.  Lower value (e.g. 0.95) makes the agent
        # focus on near-term PnL rather than long-horizon speculation.
        gamma         = 0.95,
        # gae_lambda: GAE smoothing for advantage estimates.
        gae_lambda    = 0.95,
        # clip_range: PPO clipping parameter.  0.2 is the standard default.
        clip_range    = 0.2,
        verbose       = 1,
        device        = "cuda",
    )

    print(f"Policy device : {model.device}")
    print(f"Policy network: {model.policy}\n")

    # ---------------------------------------------------------------------- #
    # 4.  Train
    # ---------------------------------------------------------------------- #
    print("=" * 60)
    print(f"Step 4 — Training for {TOTAL_TIMESTEPS:,} timesteps")
    print("=" * 60)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # ---------------------------------------------------------------------- #
    # 5.  Save
    # ---------------------------------------------------------------------- #
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to  {MODEL_SAVE_PATH}.zip")

    # ---------------------------------------------------------------------- #
    # 6.  Quick sanity evaluation — one full episode, no exploration
    # ---------------------------------------------------------------------- #
    print("\n" + "=" * 60)
    print("Step 5 — Sanity evaluation (one episode, deterministic)")
    print("=" * 60)

    obs, _ = env.reset()
    total_reward = 0.0
    step_count   = 0

    while True:
        # deterministic=True disables the stochastic exploration noise.
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count   += 1
        if terminated or truncated:
            break

    print(f"Steps run      : {step_count}")
    print(f"Final balance  : ${info['balance']:,.2f}  (started at ${INITIAL_CASH:,.2f})")
    print(f"Total reward   : {total_reward:.4f}")
    print(f"Final position : {info['position']:.3f}")


if __name__ == "__main__":
    main()
