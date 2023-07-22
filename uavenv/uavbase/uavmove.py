import gym
import numpy as np

class uavposition:
    """
    uavposition 是一个无人机的位置类，用于存储无人机的位置信息
    """
    def __init__(self, position, azimuth, elevation, speed):
        self.position = position
        self.azimuth = azimuth # 方位角
        self.elevation = elevation # 俯仰角
        self.speed = speed # 速度
        self.speeds = []
        self.azimuths = []
        self.elevations = []


class uavbase(gym.Env):
    def __init__():
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