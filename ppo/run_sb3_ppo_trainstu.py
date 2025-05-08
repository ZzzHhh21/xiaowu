import os
import time
import torch
import argparse
import numpy as np
from mujoco import mjx
from collections import deque
import gymnasium as gym
from gymnasium.envs import register
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from torch.nn import functional as F
from gymnasium.spaces import Box

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device("cpu")

# 注册环境
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

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument("--env_name",        default="xiaowu-xiaowureach-v0", type=str)
parser.add_argument("--seed",            default=0, type=int)
parser.add_argument("--num_envs",        default=2, type=int)
parser.add_argument("--learning_rate",   default=3e-5, type=float)
parser.add_argument("--steps_per_epoch", default=4000, type=int)
parser.add_argument("--epochs",          default=6000, type=int)
parser.add_argument("--max_ep_len",      default=2000, type=int)
parser.add_argument("--expert_path",     default="/home/zzzhhh/tencent/0_12000.pt", type=str)
ARGS = parser.parse_args()

# 学生策略的简化观测
class PartialObsWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(137,), dtype=np.float64)
    
    def _extract_partial_obs(self, full_obs):
        # 计算各部分在观测中的位置
        qpos = full_obs[:38]
        qvel = full_obs[38:75]
        return np.concatenate([qpos, qvel])
    
    def reset(self):
        obs = self.venv.reset()
        return self._extract_partial_obs(obs)
    
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        partial_obs = self._extract_partial_obs(obs)
        return partial_obs, rewards, dones, infos


def make_env(rank, student=False):
    def _init():
        env = gym.make(ARGS.env_name)
        env.reset(seed=ARGS.seed + rank)
        env.action_space.seed(ARGS.seed + rank)
        return env
    return _init

# DAgger数据集
class DAggerDataset:
    def __init__(self, max_size=1e6):
        self.max_size = int(max_size)
        self.observations = []
        self.expert_actions = []
    
    def add(self, obs, expert_act):
        self.observations.extend(obs)
        self.expert_actions.extend(expert_act)
        if len(self.observations) > self.max_size:
            remove_count = len(self.observations) - self.max_size
            self.observations = self.observations[remove_count:]
            self.expert_actions = self.expert_actions[remove_count:]
    
    def get_batches(self, batch_size=64):
        indices = np.random.permutation(len(self.observations))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            yield (
                np.array([self.observations[i] for i in batch_idx]),
                np.array([self.expert_actions[i] for i in batch_idx])
            )

# 加载专家策略
def load_expert_model(expert_model_path, env):
    checkpoint = torch.load(expert_model_path)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-5, n_steps=2000)
    model.policy.load_state_dict(checkpoint["model_state_dict"])
    model.policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model

# 评估策略
def evaluate_policy(model, env, num_episodes=10):
    current_returns = np.zeros(env.num_envs)
    completed_episodes = 0
    all_returns = []
    
    obs = env.reset()
    while completed_episodes < num_episodes:
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        current_returns += rewards
        
        for i, done in enumerate(dones):
            if done:
                all_returns.append(current_returns[i])
                current_returns[i] = 0
                completed_episodes += 1
                if completed_episodes >= num_episodes:
                    break
    return np.mean(all_returns)

# DAgger训练
def dagger_train(expert_model, student_model, expert_env, student_env, 
                epochs=10000, steps_per_epoch=6500, num_envs=5,
                save_dir=None, save_freq=10, seed=0):
    dataset = DAggerDataset(max_size=1e6)
    
    for epoch in range(epochs):
        obs = expert_env.reset()  # 完整观测环境
        dones = np.zeros(num_envs, dtype=bool)
        for _ in range(steps_per_epoch // num_envs):
            partial_obs = student_env._extract_partial_obs(obs)

            # 学生策略执行（训练时保留随机性）
            student_actions, _ = student_model.predict(partial_obs, deterministic=False)
            obs, _, dones, _ = expert_env.step(student_actions)

            expert_actions, _ = expert_model.predict(obs, deterministic=True)
            dataset.add(partial_obs, expert_actions)  # 记录专家动作

        # 监督学习阶段
        student_model.policy.train()
        total_loss = 0.0
        for _ in range(5):  # 多个训练循环
            for obs_batch, act_batch in dataset.get_batches(batch_size=256):
                obs_tensor = torch.as_tensor(obs_batch, dtype=torch.float32, device=student_model.device)
                act_tensor = torch.as_tensor(act_batch, dtype=torch.float32, device=student_model.device)   
                
                # 计算MSE损失
                dist = student_model.policy.get_distribution(obs_tensor)
                student_act = dist.distribution.mean  
                loss = F.mse_loss(student_act, act_tensor)

                # student_act = student_model.predict(obs_tensor, deterministic=True)
                # student_act = torch.as_tensor(student_act, dtype=torch.float32, device=student_model.device).requires_grad_(True)
                # loss = F.mse_loss(student_act, act_tensor)
                
                student_model.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student_model.policy.parameters(), 0.5)
                student_model.policy.optimizer.step()
                total_loss += loss.item()
        
        # 评估当前策略
        stu_mean_return = evaluate_policy(student_model, student_env, num_episodes=10)
        # tea_mean_return = evaluate_policy(expert_model, expert_env, num_episodes=10)
        # print(f"Epoch {epoch+1}/{epochs} | Tea_Return: {tea_mean_return:.2f}")
        print(f"Epoch {epoch+1}/{epochs} | Stu_Return: {stu_mean_return:.2f}")

        # 保存模型
        if (epoch + 1) % save_freq == 0:
            checkpoint = {
                "model_state_dict": student_model.policy.state_dict(),
                "optimizer_state_dict": student_model.policy.optimizer.state_dict(),
                "iter": epoch + 1,
            }
            filename = f"stu_{seed}_{epoch+1}.pt"
            torch.save(checkpoint, os.path.join(save_dir, filename))
            print(f" Saved student model at epoch {epoch+1}: {filename}")


if __name__ == "__main__":
    expert_env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    student_env = SubprocVecEnv([make_env(i) for i in range(ARGS.num_envs)])
    student_env = PartialObsWrapper(student_env)
    
    expert_model = load_expert_model(ARGS.expert_path, expert_env)

    clean_env_name = ARGS.env_name.replace(":", "-").replace("/", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("logs_stu", f"stu_ppo_{timestamp}_{clean_env_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化学生模型 (使用部分观测空间)
    student_model = PPO(
        "MlpPolicy",
        student_env,
        learning_rate=3e-5,
        n_steps=ARGS.steps_per_epoch // ARGS.num_envs,
        verbose=1
    )

    # 使用DAgger训练
    dagger_train(
        expert_model=expert_model,
        student_model=student_model,
        expert_env=expert_env,
        student_env=student_env,
        epochs=ARGS.epochs,
        steps_per_epoch=ARGS.steps_per_epoch,
        num_envs=ARGS.num_envs,
        save_dir=save_dir,
        save_freq=10,
        seed=ARGS.seed
    )