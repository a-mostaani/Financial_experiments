import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO

# --- MEMORY FENCING ---
# Restrict PyTorch to only use a fraction of the GPU to protect the LLM
# Adjust the fraction based on your total VRAM (e.g., 1GB / 12GB = ~0.08)
torch.cuda.set_per_process_memory_fraction(0.05, device=0)
torch.cuda.empty_cache()

class ArbitrageTradingEnv(gym.Env):
    """VRAM-optimized custom trading environment."""
    
    def __init__(self):
        super(ArbitrageTradingEnv, self).__init__()
        
        # Action Space: Continuous [-1, 1] for Short/Hold/Long
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation Space: [price_change, sentiment_score, sentiment_volatility, bid_ask_spread, z_score]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )
        
        self.balance = 10000.0

    def step(self, action):
        obs = self._get_current_observation()
        
        # Execute trade and calculate reward (Placeholder)
        step_reward = self._calculate_reward(action, obs)
        
        terminated = self.balance <= 0
        truncated = False 
        
        return obs, step_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 10000.0
        return self._get_current_observation(), {}
        
    def _get_current_observation(self):
        # The data format adapter processes the raw API and LLM text outputs,
        # perfectly standardizing them into this clean float array for the GPU.
        price_change = 0.005 # Normalized price
        sentiment_score = 0.8
        sentiment_volatility = 0.05
        bid_ask_spread = 0.02
        z_score = 2.1
        
        return np.array([
            price_change, 
            sentiment_score, 
            sentiment_volatility, 
            bid_ask_spread, 
            z_score
        ], dtype=np.float32)

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