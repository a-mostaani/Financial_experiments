import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from trader.feature_builder import build_observation, OBS_DIM

# --- MEMORY FENCING ---
# Restrict PyTorch to only use a fraction of the GPU to protect the LLM
# Adjust the fraction based on your total VRAM (e.g., 1GB / 12GB = ~0.08)
torch.cuda.set_per_process_memory_fraction(0.05, device=0)
torch.cuda.empty_cache()

class ArbitrageTradingEnv(gym.Env):
    """VRAM-optimized custom trading environment."""
    
    def __init__(self, initial_cash: float = 10000.0):
        super(ArbitrageTradingEnv, self).__init__()

        # Action Space: Continuous [-1, 1] for Short/Hold/Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation Space: 9 normalised features built by feature_builder.build_observation
        #   [0] sentiment_score, [1] spread_pct, [2] z_score, [3] beta,
        #   [4] adf_t, [5] coint_pvalue_inv, [6] cointegrated,
        #   [7] cash_ratio, [8] market_volatility
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

        self.initial_cash = initial_cash
        self.balance = initial_cash

    def step(self, action, **obs_kwargs):
        obs = self._get_current_observation(**obs_kwargs)

        # Execute trade and calculate reward (Placeholder)
        step_reward = self._calculate_reward(action, obs)

        terminated = self.balance <= 0
        truncated = False

        return obs, step_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_cash
        return self._get_current_observation(), {}
        
    def _get_current_observation(
        self,
        sentiment_label: str = "neutral",
        spread_pct: float = 0.0,
        z_score: float = 0.0,
        beta: float = 1.0,
        adf_t: float = 0.0,
        coint_pvalue: float = 1.0,
        cointegrated: bool = False,
        market_volatility: float = 0.0,
    ) -> np.ndarray:
        """
        Build the normalised observation vector.

        Pass live values from:
          - sentiment_label   : sentiment_worker  (analyze_sentiment output)
          - spread_pct        : calc_spread        (get_bid_ask_spread()["spread_pct"])
          - z_score           : calc_z             (latest rolling_zscore value)
          - beta, adf_t,
            coint_pvalue,
            cointegrated      : arbit_gpu_vectorized (latest window row from scan_cointegration_windows_gpu)
          - market_volatility : rolling std of returns for the traded asset
        """
        return build_observation(
            sentiment_label=sentiment_label,
            spread_pct=spread_pct,
            z_score=z_score,
            beta=beta,
            adf_t=adf_t,
            coint_pvalue=coint_pvalue,
            cointegrated=cointegrated,
            available_cash=self.balance,
            initial_cash=self.initial_cash,
            market_volatility=market_volatility,
        )

    def _calculate_reward(self, action, obs):
        return 0.0

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