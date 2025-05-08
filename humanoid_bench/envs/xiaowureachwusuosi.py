import os
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.spaces import Box
from mujoco import mjx

class XiaoWuReach:
    dof = 0

    frame_skip = 1 ##### 原10

    camera_name = "cam_default"
    max_episode_steps = 2000

    def __init__(self, robot=None, env=None, enable_dr=True, dr_scales=None, dr_noise_std=None, **kwargs):
        self.robot = robot
        self._env = env
        self.unwrapped = self
        self._actuator_names = None

        self.enable_dr = enable_dr # 是否启用域随机化
        self.dr_scales = dr_scales or { # 各部分观测的缩放范围(使用对数范围)
            'qpos': 0.1, 
            'qvel': 0.1,   
            'force': 0,
            'prev_action': 0,
            'kp': 0.1, 
            'kv': 0.1
        }
        self.dr_noise_std = dr_noise_std or { # 各部分观测的噪声标准差
            'qpos': 0,
            'qvel': 0,
            'force': 0.01,
            'prev_action': 0,
            'kp': 0, 
            'kv': 0
        }
        self.dr_params = {}  # 存储当前episode的随机参数

        self._initial_zero_steps = 100  # 空指令持续步数
        self._current_step = 0  # 当前步数计数器
        self._current_epoch = 0

        # 安全初始化与env相关的属性
        self._original_kp = None
        self._original_kv = None

        # 只有在env存在时才初始化MuJoCo模型参数
        if self._env is not None:
            self._original_kp = self._env.model.actuator_gainprm[:, 0].copy()
            self._original_kv = self._env.model.actuator_biasprm[:, 1].copy()
        
        if env:
            self._env.viewer = self._env.mujoco_renderer._get_viewer(self._env.render_mode)

        self._episode_total_reward = 0.0  # 跟踪回合累计奖励
        self._episode_steps = 0           # 跟踪回合步数

        # 需要固定为0的关节列表
        self._fixed_joints = [
            "J_LAT_HIP", "J_LAT_LEG_L", "J_LAT_WHEEL_L", "J_LAT_FOOT_L",
            "J_LAT_LEG_R", "J_LAT_WHEEL_R", "J_LAT_FOOT_R",
            "J_MED_HIP", "J_MED_LEG_L", "J_MED_WHEEL_L", "J_MED_FOOT_L",
            "J_MED_LEG_R", "J_MED_WHEEL_R", "J_MED_FOOT_R",
            "J_WAIST_PITCH", "J_WAIST_YAW", "J_HEAD",
            "J_SHOULDER_L3", "J_ELBOW_L1", "J_ELBOW_L2", "J_WRIST_L1", "J_WRIST_L2",
            "J_SHOULDER_R3", "J_ELBOW_R1", "J_ELBOW_R2", "J_WRIST_R1", "J_WRIST_R2"
        ]

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf,shape=(137,), dtype=np.float64)
    
    def _apply_domain_randomization(self, obs_components):
        if not self.enable_dr or not self.dr_params:
            return obs_components
        # 对各分量分别应用缩放和噪声
        randomized = []
        for name, data in obs_components.items():
            scaled = data * self.dr_params[f"{name}_scale"]
            noised = scaled + np.random.normal(0, self.dr_noise_std[name], scaled.shape)
            randomized.append(noised)
        
        return randomized

    def get_obs(self):
        current_actuator_force = self._env.data.actuator_force.flat.copy()
        # 计算力变化率
        if self._prev_actuator_force is not None:
            force_change_rate = (current_actuator_force - self._prev_actuator_force) / (np.abs(self._prev_actuator_force) + 1e-6)
        else:
            force_change_rate = np.zeros_like(current_actuator_force)
        
        # 更新上一帧的力信息
        self._prev_actuator_force = current_actuator_force.flat.copy()
        prev_action = self._prev_action.flat.copy() if self._prev_action is not None else np.zeros(self._env.action_space.shape)

        obs_components = { 
            'qpos': self._env.data.qpos.flat.copy(), # 38
            'qvel': self._env.data.qvel.flat.copy(), # 75
            'force': force_change_rate,
            'prev_action': prev_action
        }
        randomized = self._apply_domain_randomization(obs_components)
        return np.concatenate(randomized)

    def get_terminated(self):
        return False, {}

    def get_reward(self):

        POSITION_THRESHOLD_SMALL = 0.05
        POSITION_THRESHOLD_MEDIUM = 0.1
        POSITION_THRESHOLD_LARGE = 0.15

        VELOCITY_THRESHOLD_SMALL = 0.05
        VELOCITY_THRESHOLD_MEDIUM = 0.15
        VELOCITY_THRESHOLD_LARGE = 0.3

        elbow_l_pos_err = np.linalg.norm(self.robot.elbow_left_position() - self.robot.elbow_left_target_position())
        elbow_r_pos_err = np.linalg.norm(self.robot.elbow_right_position() - self.robot.elbow_right_target_position())
        elbow_l_vel = np.linalg.norm(self.robot.elbow_left_lin_vel())
        elbow_r_vel = np.linalg.norm(self.robot.elbow_right_lin_vel())

        ##### 
        head_height = self.robot.head_height()
        elbow_right_height = self.robot.elbow_right_height()
        elbow_left_height = self.robot.elbow_left_height()

        if elbow_right_height >= head_height:
            height_right_bonus = -5
        else:
            height_right_bonus = 0
        
        if elbow_left_height >= head_height:
            height_left_bonus = -5
        else:
            height_left_bonus = 0
        
        height_bonus = height_right_bonus + height_left_bonus
        
        # 分别计算肘部奖励
        def calculate_part_reward(pos_err, vel, part_name):
            pos_bonus = -5 * pos_err
            vel_bonus = -1 * vel

            if pos_err <= POSITION_THRESHOLD_SMALL:  
                pos_ratio = 1 - pos_err / POSITION_THRESHOLD_SMALL
                precision_pos_bonus= 3 + 3 * pos_ratio  
                
            elif pos_err <= POSITION_THRESHOLD_MEDIUM: 
                pos_ratio = 1 - (pos_err - POSITION_THRESHOLD_SMALL) / (POSITION_THRESHOLD_MEDIUM - POSITION_THRESHOLD_SMALL)
                precision_pos_bonus = 1 + 2 * pos_ratio  
                
            elif pos_err <= POSITION_THRESHOLD_LARGE: 
                pos_ratio = 1 - (pos_err - POSITION_THRESHOLD_MEDIUM) / (POSITION_THRESHOLD_LARGE - POSITION_THRESHOLD_MEDIUM)
                precision_pos_bonus = 0 + 1 * pos_ratio                
            else: 
                precision_pos_bonus = -10 * pos_err

            return {
                f"{part_name}_total": pos_bonus + vel_bonus  + precision_pos_bonus, 
                f"{part_name}_reached3": pos_err <= 0.03,
                f"{part_name}_reached5": pos_err <= 0.03
            }
            
            
        # 计算各部位奖励
        elbow_l_reward = calculate_part_reward(elbow_l_pos_err, elbow_l_vel, "elbow_l")
        elbow_r_reward = calculate_part_reward(elbow_r_pos_err, elbow_r_vel, "elbow_r")

        self._left_reached5 = elbow_l_reward["elbow_l_reached5"]
        self._right_reached5 = elbow_r_reward["elbow_r_reached5"]
        self._is_success = self._left_reached5 and self._right_reached5

        left_success_bonus = 2 if self._left_reached5 else 0.0
        right_success_bonus = 2 if self._right_reached5 else 0.0
        success_bonus = 5 if self._is_success else 0.0
        
        # 总奖励
        total_reward = (
            height_bonus +
            elbow_l_reward["elbow_l_total"] +
            elbow_r_reward["elbow_r_total"] +
            left_success_bonus +
            right_success_bonus +
            success_bonus
        )
        
        # 调试信息
        reward_info = {
            "success_bonus": success_bonus,
            "is_success": self._is_success,
            **elbow_l_reward,
            **elbow_r_reward,
        }
        
        return total_reward, reward_info
    
    def _sample_log_uniform(self, low, high):
        log_low = np.log10(low)
        log_high = np.log10(high)
        return 10 ** np.random.uniform(log_low, log_high)

    def reset_model(self):
        self._current_step = 0
        self._current_epoch += 1

        self._prev_actuator_force = None
        self._prev_action = None
        self._is_success = False

        self._left_reached5 = False
        self._right_reached5 = False

        self._left_permanently_locked = False  # 重置左手永久锁定
        self._right_permanently_locked = False # 重置右手永久锁定

        self._env.model.actuator_gainprm[:, 0] = self._original_kp.copy()
        self._env.model.actuator_biasprm[:, 1] = self._original_kv.copy()
        if self.enable_dr:
            # 使用对数均匀分布采样缩放参数
            self.dr_params = {
                'qpos_scale': self._sample_log_uniform(
                    1 - self.dr_scales['qpos'], 
                    1 + self.dr_scales['qpos']
                ),
                'qvel_scale': self._sample_log_uniform(
                    1 - self.dr_scales['qvel'], 
                    1 + self.dr_scales['qvel']
                ),
                'force_scale': self._sample_log_uniform(
                    1 - self.dr_scales['force'], 
                    1 + self.dr_scales['force']
                ),
                'prev_action_scale': self._sample_log_uniform(
                    1 - self.dr_scales['prev_action'], 
                    1 + self.dr_scales['prev_action']
                ),
                'kp_scale': self._sample_log_uniform(
                    1 - self.dr_scales['kp'], 
                    1 + self.dr_scales['kp']
                ),
                'kv_scale': self._sample_log_uniform(
                    1 - self.dr_scales['kv'], 
                    1 + self.dr_scales['kv']
                ),
            }
            for i in range(self._env.model.nu):
                if self._env.model.actuator_gainprm[i, 0] != 0: 
                    self._env.model.actuator_gainprm[i, 0] *= self.dr_params['kp_scale']
                if self._env.model.actuator_biasprm[i, 1] != 0: 
                    self._env.model.actuator_biasprm[i, 1] *= self.dr_params['kv_scale']
        else:
            self.dr_params = {}

        return self.get_obs()

    def step(self, action):
        self._current_step += 1
        # 定义左右臂的关节索引
        self._left_arm_joint_names = ["J_SHOULDER_L1", "J_SHOULDER_L2", "J_SHOULDER_L3", "J_ELBOW_L1", "J_ELBOW_L2", "J_WRIST_L1", "J_WRIST_L2"] 
        self._right_arm_joint_names = ["J_SHOULDER_R1","J_SHOULDER_R2","J_SHOULDER_R3", "J_ELBOW_R1", "J_ELBOW_R2", "J_WRIST_R1", "J_WRIST_R2"] 

        # 定义左右臂的关节索引
        if self._actuator_names is None and self._env is not None:
            self._actuator_names = [mujoco.mj_id2name(self._env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(self._env.model.nu)]
        
        self._left_arm_indices = [i for i, name in enumerate(self._actuator_names) if name in self._left_arm_joint_names]
        self._right_arm_indices = [i for i, name in enumerate(self._actuator_names) if name in self._right_arm_joint_names]

        if self._current_step <= self._initial_zero_steps:
            unnormalized_action = np.zeros_like(action)
        else:
            current_qpos = self._env.data.qpos.copy()[-len(action):]
            unnormalized_action = self.unnormalize_action(action)
            smoothing_factor = 0.2

            unnormalized_action = (1 - smoothing_factor) * current_qpos + smoothing_factor * unnormalized_action

        if hasattr(self, '_fixed_joints'):
            for i, name in enumerate(self._actuator_names):
                if name in self._fixed_joints:
                    unnormalized_action[i] = 0.0

        self._prev_action = unnormalized_action.flat.copy()

        self._env.do_simulation(unnormalized_action, self._env.frame_skip)
        obs = self.get_obs()
        reward, reward_info = self.get_reward()

        self._episode_total_reward += reward  # 累计奖励
        self._episode_steps += 1              # 累计步数

        terminated, terminated_info = self.get_terminated()
        truncated = self._episode_steps >= self.max_episode_steps 

        info = {
            "per_timestep_reward": reward,
            **reward_info,
            **terminated_info,
        }

        # 如果是回合结束-添加 episode 信息
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_total_reward,
                "l": self._episode_steps,
            }
            # 重置累计变量
            self._episode_total_reward = 0.0
            self._episode_steps = 0

        return obs, reward, terminated, truncated, info

    def normalize_action(self, action):
        return (
            2
            * (action - self._env.action_low)
            / (self._env.action_high - self._env.action_low)
            - 1
        )

    def unnormalize_action(self, action):
        return (action + 1) / 2 * (
            self._env.action_high - self._env.action_low
        ) + self._env.action_low

    def render(self):
        return self._env.mujoco_renderer.render(
            self._env.render_mode, self._env.camera_id, self.camera_name
        )