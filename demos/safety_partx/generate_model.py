from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./logs_CartpoleEnv/")
event_callback = EveryNTimesteps(n_steps=10000, callback=checkpoint_on_event)

itera = 100000

env = gym.make("CartPole-v1")

modelppo = PPO("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log=f"Cartpole_env_tensorboardfiles{itera}/")

modelppo.learn(itera, callback=event_callback)
# modelppo.save(f"new_rl_model_{itera}_steps")
