import gym
import numpy as np

class UAVposition:
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

class channels:
    """
    channels 是一个信道类，用于存储多个无人机的信道信息
    """
    def __init__(self, n_channel, n_uav):
        self.n_channel = n_channel
        self.n_uav = n_uav
    
    def update_positions(self, positions):
        self.positions = positions
    
    def update_path_loss(self):
        self.path_loss = np.zeros(shape=(len(self.positions),len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.path_loss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])
    
    def get_path_loss(self, x, y):
        distance = np.sqrt(np.sum(np.square(x - y))+1e-6)
        # 这个是路径损耗模型，可以根据需要修改
        loss_db = 103.8 + 20.9 * np.log10(distance*1e-3)
        return loss_db

    def update_fast_fading(self): # 瑞利衰落
        h = (np.random.normal(size=(self.n_uav, self.n_uav, self.channels)) + 1j * np.random.normal(size=(self.n_uav, self.n_uav, self.channels))) / np.sqrt(2)
        self.fast_fading_db = 20 * np.log10(np.abs(h))



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