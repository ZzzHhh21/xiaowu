import os
import time
import torch
import argparse
import numpy as np
from mujoco import mjx

import gymnasium as gym
from gymnasium.envs import register
from gymnasium.wrappers import TimeLimit

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

register(
    id='xiaowu-xiaowureach-v0',
    entry_point="humanoid_bench.env:HumanoidEnv",
    max_episode_steps=2000,
    kwargs={
        'robot': 'xiaowu',
        'control': 'pos',
        'task': 'xiaowureach',
    }
)

parser = argparse.ArgumentParser()
parser.add_argument("--env_name",        default="xiaowu-xiaowureach-v0", type=str)
parser.add_argument("--seed",            default=0, type=int)
parser.add_argument("--num_envs",        default=4, type=int)
parser.add_argument("--learning_rate",   default=3e-5, type=float)
parser.add_argument("--steps_per_epoch", default=8000, type=int)       # 一次训练最大4000step -- 一共4个环境 -- 一个环境1000step
parser.add_argument("--epochs",          default=20000, type=int)        # 一共训练1000次
parser.add_argument("--max_ep_len",      default=2000, type=int)        # 一幕最大1000step
ARGS = parser.parse_args()


def make_env(rank):
    def _init():
        env = gym.make(ARGS.env_name)
        env.reset(seed=ARGS.seed + rank)
        env.action_space.seed(ARGS.seed + rank)  # 确保种子影响 action 采样
        return env
    return _init


class CombinedCallback(BaseCallback):
    def __init__(self, 
                 env_name: str,
                 seed: int,
                 total_epochs: int,
                 save_freq: int = 50,
                 save_path: str = "logs",
                 verbose: int = 0):
        super().__init__(verbose)
        
        clean_env_name = env_name.replace(":", "-").replace("/", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_path, f"ppo_{timestamp}_{clean_env_name}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.env_name = env_name
        self.seed = seed
        self.save_freq = save_freq
        
        self.total_epochs = total_epochs
        self.current_epoch = 0
        # dist_reward + vel_reward_wrist + vel_reward_elbow + position_penalty 
        self.ep_returns = []
        self.ep_lengths = []
        self.infos = {}  

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.ep_returns.append(info["episode"]["r"])
                self.ep_lengths.append(info["episode"]["l"])
                self.infos = info 
        return True

    def _on_rollout_end(self) -> None:
        self.current_epoch += 1   
        if self.current_epoch % self.save_freq == 0:
            checkpoint = {
                "model_state_dict": self.model.policy.state_dict(),       
                "optimizer_state_dict": self.model.policy.optimizer.state_dict(), 
                "iter": self.current_epoch,                              
                "infos": self.infos,                                    
            }
            base_name = f"{self.seed}_{self.current_epoch}"
            pt_path = os.path.join(self.save_dir, f"{base_name}.pt")
            torch.save(checkpoint, pt_path)
            
            if self.verbose:
                print(f"Saved models: {pt_path}")
        
        # 日志记录逻辑
        if self.ep_returns:
            self.logger.record("train/mean_return", np.mean(self.ep_returns))
            self.logger.record("train/mean_length", np.mean(self.ep_lengths))
            self.ep_returns = []
            self.ep_lengths = []
        self.logger.record("train/current_epoch", self.current_epoch)
        self.logger.record("train/total_epochs", self.total_epochs)
        if self.verbose:
            print(f"Epoch {self.current_epoch}/{self.total_epochs}")

def train():
    total_steps = ARGS.steps_per_epoch * ARGS.epochs
    env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=ARGS.learning_rate,
        n_steps=ARGS.steps_per_epoch // ARGS.num_envs,
        tensorboard_log="logs"
    )
    
    callbacks = [
        CombinedCallback(
            env_name=ARGS.env_name,
            seed=ARGS.seed,
            total_epochs=ARGS.epochs,
            verbose=1
        )
    ]
        
    model.learn(
        total_timesteps=total_steps,
        callback=callbacks,
        tb_log_name=f"ppo_{ARGS.env_name}"
    )

if __name__ == "__main__":
        train()