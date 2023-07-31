from typing import List, Optional, Union
import gym
from gym import spaces
import numpy as np
import random

class systemEnv(gym.Env):
    """
    """
    def __init__(
        self, n_clusters=3, n_channels=6, n_slaves=3, n_jammers=3, area_type="small_and_medium_size_cities", jamming_mode='Markov', fc=800*1e6, hb=50, hm=20,
        xlim=1000, ylim=1000, zlim_max=200, zlim_min=50, max_radius=50, master_velocity=10, slave_velocity=10, moving_factor=0.1, dt=0.1, **kwargs
    ):
        # 定义channel参数
        self.n_clusters = n_clusters
        self.n_channels = n_channels
        self.n_slaves = n_slaves
        self.n_jammers = n_jammers
        self.area_type = area_type
        self.jamming_mode = jamming_mode
        self.fc = fc
        self.hb = hb
        self.hm = hm
        # 定义移动参数
        self.xlim = xlim
        self.ylim = ylim
        self.zlim_max = zlim_max
        self.zlim_min = zlim_min
        self.max_radius = max_radius
        self.master_velocity = master_velocity
        self.slave_velocity = slave_velocity
        self.moving_factor = moving_factor
        self.dt = dt
        # 定义动作空间&观测空间
        part1 = spaces.Box(low=0, high=self.n_channels-1, shape=(self.n_slaves+self.n_clusters,2)) # 选择的 channel, power
        part2 = spaces.Box(low=0, high=1,shape=(self.n_channels,)) #! 每个信道的增益 high是多少我还不知道 注意这个地方没有写对
        part3 = spaces.Box(low=0, high=max(xlim, ylim, zlim_max, zlim_min), shape=(self.n_slaves+self.n_clusters+1,3)) # 位置信息
        self.observation_space = [spaces.Tuple((part1, part2, part3)) for _ in range(self.n_clusters)]
        self.action_space = [part1 for _ in range(self.n_clusters)]
        # 定义
        


    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def seed(self):
        pass