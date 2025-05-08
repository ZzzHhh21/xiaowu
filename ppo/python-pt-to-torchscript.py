import torch
import torch.nn as nn
from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.envs import register

# 注册环境（与你原代码一致）
episode_step = 2000
register(
    id='xiaowu-xiaowureach-v0',
    entry_point="humanoid_bench.env:HumanoidEnv",
    max_episode_steps=episode_step,
    kwargs={
        'robot': 'xiaowu',
        'control': 'pos',
        'task': 'xiaowureach',
    }
)

# 封装一个纯推理策略模块
class TorchScriptPolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net

    def forward(self, obs):
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        action = self.action_net(latent_pi)
        return action

def export_model(checkpoint_path, export_path="policy_scripted.pt"):
    # 创建环境（无渲染模式避免报错）
    env = gym.make(
        "xiaowu-xiaowureach-v0",
        render_mode="rgb_array",  # 设置为 "rgb_array" 来兼容 mujoco
        max_episode_steps=episode_step,
        kwargs={
            'robot': 'xiaowu',
            'control': 'pos',
            'task': 'xiaowureach',
        }
    )

    # 初始化模型结构
    model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-5, 
            n_steps=episode_step,     
        )
    
    # 加载 checkpoint state_dict
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.policy.load_state_dict(checkpoint["model_state_dict"])
    model.policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 封装为兼容 TorchScript 的模型
    wrapper = TorchScriptPolicyWrapper(model.policy).eval().to("cpu")

    # 使用 dummy input trace（注意：放到 CPU）
    obs_dim = env.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_dim).to("cpu")

    # 使用 TorchScript trace 导出
    traced = torch.jit.trace(wrapper, dummy_input)

    # 保存为 TorchScript 模型
    traced.save(export_path)
    print(f"TorchScript 模型已保存: {export_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="路径到原始 .pt 模型 state_dict")
    parser.add_argument("--out", default="./torchscript/policy_scripted.pt", help="保存为 TorchScript 的路径")
    args = parser.parse_args()

    export_model(args.ckpt, args.out)
