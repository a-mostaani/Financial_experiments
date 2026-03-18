import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from trader.feature_builder import build_observation, OBS_DIM

# --- MEMORY FENCING ---
# Restrict PyTorch to only use a fraction of the GPU to protect the LLM.
# Adjust the fraction based on your total VRAM (e.g., 1 GB / 12 GB ≈ 0.08).
torch.cuda.set_per_process_memory_fraction(0.05, device=0)
torch.cuda.empty_cache()


class ArbitrageTradingEnv(gym.Env):
    """
    VRAM-optimized pairs-trading environment.

    The agent trades a cointegrated spread (asset A vs asset B).
    At each step it receives a 9-dimensional observation built from live
    market data and outputs a continuous action in [-1, 1]:

        action > 0  →  LONG  the spread  (buy A, short B)
        action ≈ 0  →  HOLD  (no change to position)
        action < 0  →  SHORT the spread  (sell A, buy B)

    The magnitude encodes conviction / position size as a fraction of balance.
    """

    def __init__(
        self,
        initial_cash: float = 10000.0,
        max_position_fraction: float = 0.5,
        transaction_cost_floor: float = 0.0001,
    ):
        """
        Parameters
        ----------
        initial_cash            : Starting capital in USD.
        max_position_fraction   : Max fraction of balance that can be deployed
                                  in a single direction.  0.5 means at most 50 %
                                  of the current balance is at risk at any time.
        transaction_cost_floor  : Minimum spread_pct used when computing
                                  transaction costs, to avoid a zero-cost assumption
                                  on illiquid data.
        """
        super(ArbitrageTradingEnv, self).__init__()

        # ------------------------------------------------------------------ #
        # Action space: one continuous scalar in [-1, 1]
        #   Positive  → long spread  (long A, short B)
        #   Negative  → short spread (short A, long B)
        #   Magnitude → how much of the allowed capital to deploy
        # ------------------------------------------------------------------ #
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # ------------------------------------------------------------------ #
        # Observation space: 9 normalised floats from feature_builder
        #   [0] sentiment_score   [1] spread_pct      [2] z_score
        #   [3] beta              [4] adf_t            [5] coint_pvalue_inv
        #   [6] cointegrated      [7] cash_ratio       [8] market_volatility
        # All features are squashed to [-1, 1] via tanh (or clipped), so
        # bounds are set accordingly.
        # ------------------------------------------------------------------ #
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        # ---- Hyperparameters ----
        self.initial_cash = initial_cash
        self.max_position_fraction = max_position_fraction
        self.transaction_cost_floor = transaction_cost_floor

        # ---- Episode state (initialised properly in reset()) ----
        self.balance = initial_cash

        # Current spread position as a fraction of balance in [-max_pos, +max_pos].
        # Positive  → we are long the spread.
        # Negative  → we are short the spread.
        self.position = 0.0

        # z_score at the previous step, needed to compute the change in spread
        # value (Δz) which drives the mark-to-market PnL.
        self.prev_z_score = 0.0

    # ---------------------------------------------------------------------- #
    #  Core Gym interface
    # ---------------------------------------------------------------------- #

    def step(self, action, **obs_kwargs):
        """
        Execute one environment step.

        Parameters
        ----------
        action      : (1,) float32 array from the policy network.
        **obs_kwargs: Live market data forwarded to _get_current_observation().
                      Must include at minimum:
                        z_score    (float) — from calc_z
                        spread_pct (float) — from calc_spread

        Returns
        -------
        obs, reward, terminated, truncated, info
        """

        # ---- 1. Pull the two market values needed for trade execution ----
        # These are also used by _get_current_observation, but we need them
        # here before building obs so we can compute PnL first.
        z_score    = obs_kwargs.get("z_score", 0.0)
        spread_pct = obs_kwargs.get("spread_pct", 0.0)

        # ---- 2. Interpret the action ----
        # Clip to [-1, 1] for safety (policy outputs can occasionally overshoot).
        # Scale by max_position_fraction so the agent never risks more than that
        # fraction of current balance.
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        target_position = raw_action * self.max_position_fraction
        # target_position is now in [-max_position_fraction, +max_position_fraction]

        # ---- 3. Compute transaction cost ----
        # Every time the agent changes its position it pays the bid-ask spread
        # on the notional amount traded.
        #   notional_traded = |Δposition| * balance
        #   cost = notional_traded * spread_pct
        # We use a floor so data gaps never make trading appear free.
        position_delta = target_position - self.position
        effective_spread = max(spread_pct, self.transaction_cost_floor)
        transaction_cost = abs(position_delta) * effective_spread * self.balance

        # ---- 4. Execute position change ----
        self.position = target_position

        # ---- 5. Mark-to-market PnL ----
        # The spread can be approximated in z-score units. The change in spread
        # value over one step is proportional to Δz_score.
        #
        #   dollar_position = position_fraction * balance
        #   PnL = dollar_position * Δz / z_scale
        #
        # Dividing Δz by 3 keeps PnL in a useful scale: a 1σ move on a full
        # position changes the balance by ~balance * max_pos_fraction / 3.
        #
        # Sign convention:
        #   Long spread (position > 0) profits when z_score rises
        #     (spread widens further from mean).
        #   Short spread (position < 0) profits when z_score falls
        #     (spread reverts to mean — the typical pairs-trade thesis).
        delta_z = z_score - self.prev_z_score
        dollar_position = self.position * self.balance
        step_pnl = dollar_position * (delta_z / 3.0)

        # ---- 6. Update balance ----
        self.balance += step_pnl - transaction_cost
        self.balance = max(self.balance, 0.0)  # ruin floor — can't go negative

        # ---- 7. Store z_score for the next step ----
        self.prev_z_score = z_score

        # ---- 8. Build observation with updated balance (cash_ratio) ----
        obs = self._get_current_observation(**obs_kwargs)

        # ---- 9. Compute reward ----
        reward = self._calculate_reward(step_pnl, transaction_cost)

        # Episode ends only when the agent is fully ruined
        terminated = self.balance <= 0.0
        truncated  = False

        info = {
            "step_pnl":         step_pnl,
            "transaction_cost": transaction_cost,
            "position":         self.position,
            "balance":          self.balance,
            "delta_z":          delta_z,
        }

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, _options=None):
        super().reset(seed=seed)
        self.balance       = self.initial_cash
        self.position      = 0.0   # flat at the start of every episode
        self.prev_z_score  = 0.0   # no history at episode start
        return self._get_current_observation(), {}

    # ---------------------------------------------------------------------- #
    #  Internal helpers
    # ---------------------------------------------------------------------- #

    def _get_current_observation(
        self,
        sentiment_label: str  = "neutral",
        spread_pct:      float = 0.0,
        z_score:         float = 0.0,
        beta:            float = 1.0,
        adf_t:           float = 0.0,
        coint_pvalue:    float = 1.0,
        cointegrated:    bool  = False,
        market_volatility: float = 0.0,
    ) -> np.ndarray:
        """
        Build the normalised (9,) observation vector by delegating to
        feature_builder.build_observation().

        Parameter sources
        -----------------
        sentiment_label   : sentiment_worker  — analyze_sentiment() return value
        spread_pct        : calc_spread       — get_bid_ask_spread()["spread_pct"]
        z_score           : calc_z            — latest value of rolling_zscore()
        beta              : arbit_gpu_vectorized — latest window's OLS hedge ratio
        adf_t             : arbit_gpu_vectorized — latest window's ADF t-stat
        coint_pvalue      : arbit_gpu_vectorized — latest window's MacKinnon p-value
        cointegrated      : arbit_gpu_vectorized — latest window's cointegration flag
        market_volatility : caller-supplied    — rolling std of returns for the pair
        """
        return build_observation(
            sentiment_label   = sentiment_label,
            spread_pct        = spread_pct,
            z_score           = z_score,
            beta              = beta,
            adf_t             = adf_t,
            coint_pvalue      = coint_pvalue,
            cointegrated      = cointegrated,
            available_cash    = self.balance,
            initial_cash      = self.initial_cash,
            market_volatility = market_volatility,
        )

    def _calculate_reward(self, step_pnl: float, transaction_cost: float) -> float:
        """
        Reward function — three components:

        1. PnL reward
           Normalised by initial_cash so the reward scale stays consistent
           across episodes regardless of how the balance drifts.

        2. Transaction cost penalty
           Already subtracted from balance (real cost), but also penalised
           here to discourage rapid position flipping (overtrading).

        3. Position risk penalty
           A small quadratic penalty on |position| discourages the agent from
           holding unnecessarily large positions when the signal is weak.
           Coefficient is intentionally small so it shapes behaviour without
           dominating the PnL signal.
        """
        pnl_reward    = step_pnl / self.initial_cash
        cost_penalty  = transaction_cost / self.initial_cash
        risk_penalty  = 0.001 * (self.position ** 2)   # quadratic, max ≈ 0.001 * 0.25

        return float(pnl_reward - cost_penalty - risk_penalty)


# --- INITIALIZING THE AGENT ON GPU ---
# env = ArbitrageTradingEnv()

# # Define a strictly small Neural Network architecture
# # pi = Policy Network (Actor), vf = Value Network (Critic)
# # 2 layers of 64 neurons takes virtually zero VRAM.
# tiny_net_kwargs = dict(activation_fn=torch.nn.ReLU,
#                        net_arch=dict(pi=[64, 64], vf=[64, 64]))

# model = PPO(
#     "MlpPolicy",
#     env,
#     policy_kwargs=tiny_net_kwargs,
#     verbose=1,
#     device="cuda", # Pushes the RL calculations to the 1GB allocated GPU space
#     learning_rate=0.0003,
#     batch_size=64 # Keep batch size small to prevent VRAM spikes during updates
# )

# print("Starting VRAM-safe training...")
# model.learn(total_timesteps=100000)
# model.save("arbitrage_rl_gpu_model")
