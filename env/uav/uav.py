from typing import List, Optional, Union
import gym
from gym import spaces
import numpy as np
import random

class systemEnv(gym.Env):
    """
    """
    def __init__(
        self, n_channels=6,
    ):
        self.n_channels = n_channels
        # 动作空间&观测空间
        part1 = spaces.Box(low=0, high=self.n_channels-1, shape=(self.n_slaves+self.n_clusters,2)) # 选择的 channel, power
        part2 = spaces.Box(low=0, high=,shape=(self.n_channels,)) #! 每个信道的增益 high是多少我还不知道
        part3 = spaces.Box(low=0, high=max(xlim, ylim, zlim_max, zlim_min), shape=(self.n_slaves+self.n_clusters+1,3)) # 位置信息
        self.observation_space = [spaces.Tuple((part1, part2, part3)) for _ in range(self.n_clusters))]
        self.action_space = [part1 for _ in range(self.n_clusters)]

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def seed(self):
        pass