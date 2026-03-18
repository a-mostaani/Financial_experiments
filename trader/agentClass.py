"""
agentClass.py
-------------
Gymnasium environment for RL-based pairs (spread) trading.

The env is designed to be fully compatible with Stable Baselines 3 (SB3).
SB3 calls env.step(action) with no extra arguments, so all market data is
read from an internal DataFrame (data_feed) that is advanced one row per step.

Typical usage
~~~~~~~~~~~~~
    from trader.data_pipeline import build_training_data
    from trader.agentClass import ArbitrageTradingEnv

    data = build_training_data("AMD", "NVDA", period="60d", interval="1h")
    env  = ArbitrageTradingEnv(data_feed=data, initial_cash=10_000.0)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from trader.feature_builder import build_observation, OBS_DIM

# --- MEMORY FENCING ---
# Restrict PyTorch to a small fraction of GPU VRAM so the LLM (loaded
# separately) is not starved.  Adjust the fraction for your hardware.
# Example: 1 GB reserved out of 12 GB total → 0.08.
torch.cuda.set_per_process_memory_fraction(0.05, device=0)
torch.cuda.empty_cache()


class ArbitrageTradingEnv(gym.Env):
    """
    Pairs-trading environment for Engle-Granger cointegrated assets.

    Action space
    ------------
    Continuous scalar in [-1, 1]:
        > 0   →  LONG  the spread  (buy symbol_a, short symbol_b)
        ≈ 0   →  HOLD  (no position change)
        < 0   →  SHORT the spread  (short symbol_a, buy symbol_b)

    The magnitude encodes conviction / size as a fraction of current balance.
    The env scales the raw action by `max_position_fraction` before execution,
    so the agent never risks more than that fraction of its balance at once.

    Observation space
    -----------------
    9-element float32 vector produced by feature_builder.build_observation():
        [0] sentiment_score     [-1, 1]  encoded sentiment label
        [1] spread_pct          [-1, 1]  tanh(bid-ask spread * 50)
        [2] z_score             [-1, 1]  tanh(z / 3)
        [3] beta                [-1, 1]  tanh(hedge ratio)
        [4] adf_t               [-1, 1]  tanh(ADF t-stat / 3)
        [5] coint_pvalue_inv    [ 0, 1]  1 - MacKinnon p-value
        [6] cointegrated        {0, 1}   cointegration flag
        [7] cash_ratio          [ 0, 1]  available_cash / initial_cash
        [8] market_volatility   [-1, 1]  tanh(annualised vol * 10)

    Episode termination
    -------------------
    terminated = True  when balance reaches 0  (ruin)
    truncated  = True  when all rows in data_feed are consumed
    """

    def __init__(
        self,
        data_feed: pd.DataFrame,
        initial_cash: float = 10_000.0,
        max_position_fraction: float = 0.5,
        transaction_cost_floor: float = 0.0001,
    ):
        """
        Parameters
        ----------
        data_feed               : DataFrame from data_pipeline.build_training_data().
                                  Required columns: z_score, spread_pct, beta, adf_t,
                                  pvalue, cointegrated, market_volatility, sentiment_label.
        initial_cash            : Starting capital in USD.
        max_position_fraction   : Maximum fraction of balance deployed in any
                                  one direction.  0.5 → at most 50 % at risk.
        transaction_cost_floor  : Minimum spread_pct applied when computing
                                  transaction costs; prevents zero-cost trades
                                  on missing or stale spread data.
        """
        super().__init__()

        # Store the data feed with a clean integer index for iloc access.
        self.data_feed = data_feed.reset_index(drop=True)
        self.n_steps   = len(self.data_feed)

        # ------------------------------------------------------------------ #
        # Action space: one continuous scalar in [-1, 1].
        #   Positive  →  long the spread  (long A, short B)
        #   Negative  →  short the spread (short A, long B)
        #   Magnitude →  fraction of max_position_fraction to deploy
        # ------------------------------------------------------------------ #
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # ------------------------------------------------------------------ #
        # Observation space: OBS_DIM normalised floats, all in [-1, 1].
        # ------------------------------------------------------------------ #
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # ---- Hyperparameters ----
        self.initial_cash            = initial_cash
        self.max_position_fraction   = max_position_fraction
        self.transaction_cost_floor  = transaction_cost_floor

        # ---- Episode state (reset in reset()) ----
        self.balance       = initial_cash
        self.position      = 0.0   # current spread position fraction [-max_pos, +max_pos]
        self.prev_z_score  = 0.0   # z_score from previous step, needed for Δz PnL
        self.current_step  = 0     # pointer into data_feed

    # ---------------------------------------------------------------------- #
    #  Core Gym interface
    # ---------------------------------------------------------------------- #

    def step(self, action: np.ndarray):
        """
        Execute one trading step.

        1. Read current market row from data_feed.
        2. Interpret action → target position.
        3. Compute transaction cost on position change.
        4. Compute mark-to-market PnL from Δz_score.
        5. Update balance, position, and step pointer.
        6. Build normalised observation from updated state.
        7. Compute reward.

        Parameters
        ----------
        action : (1,) float32 array from the policy network.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        row = self.data_feed.iloc[self.current_step]

        # ---- Read market data for this bar ----
        z_score    = float(row["z_score"])
        spread_pct = float(row["spread_pct"])

        # ---- 1. Interpret the action ----
        # Clip for safety (policy outputs can overshoot [-1, 1] during training).
        # Scale by max_position_fraction to cap capital at risk.
        raw_action      = float(np.clip(action[0], -1.0, 1.0))
        target_position = raw_action * self.max_position_fraction

        # ---- 2. Transaction cost ----
        # Cost = |Δposition| × effective_spread × balance.
        # The effective spread uses a floor so free-trade assumptions are avoided.
        position_delta    = target_position - self.position
        effective_spread  = max(spread_pct, self.transaction_cost_floor)
        transaction_cost  = abs(position_delta) * effective_spread * self.balance

        # ---- 3. Execute position change ----
        self.position = target_position

        # ---- 4. Mark-to-market PnL ----
        #
        # The spread value (in z-score units) changes by Δz each bar.
        # Dollar PnL for a spread position:
        #
        #   PnL = (position_fraction × balance) × Δz / z_scale
        #
        # Dividing Δz by 3 keeps PnL proportional to a ±1σ move.
        # Sign convention:
        #   position > 0 (long spread)  profits when z_score RISES
        #   position < 0 (short spread) profits when z_score FALLS
        #                               (classic mean-reversion thesis)
        delta_z        = z_score - self.prev_z_score
        dollar_position = self.position * self.balance
        step_pnl        = dollar_position * (delta_z / 3.0)

        # ---- 5. Update balance ----
        self.balance += step_pnl - transaction_cost
        self.balance  = max(self.balance, 0.0)  # ruin floor: balance can't go negative

        # ---- 6. Advance step pointer and store z_score for next step ----
        self.prev_z_score  = z_score
        self.current_step += 1

        # ---- 7. Build observation (uses updated balance → cash_ratio) ----
        obs = self._get_current_observation(row)

        # ---- 8. Compute reward ----
        reward = self._calculate_reward(step_pnl, transaction_cost)

        # Episode ends when ruined or when all historical data is consumed.
        terminated = self.balance <= 0.0
        truncated  = self.current_step >= self.n_steps

        info = {
            "step_pnl":         step_pnl,
            "transaction_cost": transaction_cost,
            "position":         self.position,
            "balance":          self.balance,
            "delta_z":          delta_z,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):  # options is part of the Gym API; unused here
        """
        Reset the environment to the beginning of the data feed.
        Called by SB3 at the start of each rollout.
        """
        super().reset(seed=seed)
        _ = options  # explicitly acknowledged as unused (required by Gym API signature)

        self.balance       = self.initial_cash
        self.position      = 0.0
        self.prev_z_score  = 0.0
        self.current_step  = 0

        # Return observation from the very first bar.
        return self._get_current_observation(self.data_feed.iloc[0]), {}

    # ---------------------------------------------------------------------- #
    #  Internal helpers
    # ---------------------------------------------------------------------- #

    def _get_current_observation(self, row: pd.Series) -> np.ndarray:
        """
        Build the normalised (OBS_DIM,) observation vector from a data row.

        The row must contain the columns produced by data_pipeline.build_training_data():
            sentiment_label, spread_pct, z_score, beta, adf_t,
            pvalue, cointegrated, market_volatility.

        available_cash and initial_cash come from instance state so that the
        cash_ratio feature always reflects the current episode balance.
        """
        return build_observation(
            sentiment_label   = str(row.get("sentiment_label", "neutral")),
            spread_pct        = float(row["spread_pct"]),
            z_score           = float(row["z_score"]),
            beta              = float(row["beta"]),
            adf_t             = float(row["adf_t"]),
            coint_pvalue      = float(row["pvalue"]),
            cointegrated      = bool(row["cointegrated"]),
            available_cash    = self.balance,
            initial_cash      = self.initial_cash,
            market_volatility = float(row["market_volatility"]),
        )

    def _calculate_reward(self, step_pnl: float, transaction_cost: float) -> float:
        """
        Reward = PnL reward  −  transaction cost penalty  −  position risk penalty.

        Component details
        -----------------
        pnl_reward    : step_pnl normalised by initial_cash.
                        Keeps reward scale stable regardless of balance drift.

        cost_penalty  : transaction_cost normalised by initial_cash.
                        Already subtracted from balance (real cost), but
                        penalised again here to discourage overtrading /
                        rapid position flipping.

        risk_penalty  : small quadratic penalty on |position|.
                        Discourages holding large positions when the signal
                        is weak.  Coefficient 0.001 keeps it from dominating
                        the PnL signal (max value ≈ 0.001 × 0.25 = 0.00025).
        """
        pnl_reward   = step_pnl / self.initial_cash
        cost_penalty = transaction_cost / self.initial_cash
        risk_penalty = 0.001 * (self.position ** 2)

        return float(pnl_reward - cost_penalty - risk_penalty)
