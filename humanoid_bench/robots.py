
import numpy as np

class XiaoWu:
     # 总自由度 = root自由关节(7) + 其他关节(31) = 38
    # dof = 11
    # num_actuators = 4  
    dof = 38
    num_actuators = 31  
    num_targets = 2 

    def __init__(self, env=None):
        self._env = env
        self.last_action = np.zeros(self.num_actuators)
    
    # 目标位置
    def elbow_left_target_position(self):
        return self._env.named.data.site_xpos["E_L_SITE_Target"].copy()
    
    def elbow_right_target_position(self):
        return self._env.named.data.site_xpos["E_R_SITE_Target"].copy()
    
    def head_target_position(self):
        return self._env.named.data.site_xpos["H_SITE_Target"].copy()
    

    # 位置
    def elbow_left_position(self):
        return self._env.named.data.site_xpos["E_L_SITE"].copy()
    
    def elbow_left_height(self):
        return self._env.named.data.site_xpos["E_L_SITE"][2].copy()
    
    def elbow_right_position(self):
        return self._env.named.data.site_xpos["E_R_SITE"].copy()
    
    def elbow_right_height(self):
        return self._env.named.data.site_xpos["E_R_SITE"][2].copy()
    
    def head_position(self):
        return self._env.named.data.site_xpos["H_SITE"].copy()
    
    def head_height(self):
        return self._env.named.data.site_xpos["H_SITE_Target"][2].copy()
    
    def root_height(self):
        return self._env.named.data.site_xpos["ROOT"][2].copy()

    # 肘-腕线速度 -- 腰角速度
    def elbow_left_lin_vel(self):
        return self._env.named.data.sensordata["elbowl_site_linvel"].copy()
    
    def elbow_right_lin_vel(self):
        return self._env.named.data.sensordata["elbowr_site_linvel"].copy()
    
    def wrist_left_lin_vel(self):
        return self._env.named.data.sensordata["wristl_site_linvel"].copy()
    
    def wrist_right_lin_vel(self):
        return self._env.named.data.sensordata["wristr_site_linvel"].copy()
    
    def head_lin_vel(self):
        return self._env.named.data.sensordata["head_site_linvel"].copy()


