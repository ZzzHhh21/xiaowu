import os
import time
import torch
import argparse
import gymnasium as gym
from gymnasium.envs import register
from stable_baselines3 import PPO

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

def load_model_and_play(model_path: str, render: bool = True):
    try:
        # 创建环境（与训练一致）
        env = gym.make(
            "xiaowu-xiaowureach-v0",
            render_mode="human" if render else None,
            max_episode_steps=episode_step,
            kwargs={
                'robot': 'xiaowu',
                'control': 'pos',
                'task': 'xiaowureach',
            }
        )

        checkpoint = torch.load(model_path)

        # 初始化空白模型
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-5, 
            n_steps=episode_step,     
        )

        # 加载完整模型状态
        model.policy.load_state_dict(checkpoint["model_state_dict"])
        model.policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # model.policy.eval()

        # 打印加载信息
        print(f"\n成功加载模型: {model_path}")
        print(f"训练迭代次数: {checkpoint['iter']}")

        # 运行策略
        episode_count = 0
        max_episodes = 20  # 限制测试的episode数量
        
        while episode_count < max_episodes:
            obs, _ = env.reset()
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                if render:
                    env.render()
                    time.sleep(0.01)  # 控制渲染速度

            episode_count += 1
            print(f"Episode {episode_count}: Done")

    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str, help="Path to .pt model file")
    parser.add_argument("--no_render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    load_model_and_play(
        model_path=args.model,
        render=not args.no_render
    )
