import gym
import numpy as np


class uavbase(gym.Env):
    """
    uavbase 是一个基础的无人机环境，用于实现无人机的基本动作(单个)
    """
    def __init__(
        self, initx=None, inity=None, initz=None, movemode=None, # 移动相关参数
        NetStruct=None, xlim=None, ylim=None, zlim=None, # 移动限制相关参数
    ):
        pass

    def seed(self, seed=None):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass