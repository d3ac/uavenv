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
    channels 是一个信道类，用于存储多个无人机的信道信息, 考虑信道的衰落
    如果 n_jammer 不为空，则表示为干扰信道
    """
    def __init__(self, n_channel, n_uav, n_jammer=None):
        self.n_channel = n_channel
        self.n_uav = n_uav
        self.n_jammer = n_jammer
    
    def update_positions(self, positions, jammer_positions=None):
        self.uav_positions = positions
        self.jammer_positions = jammer_positions
    
    def update_path_loss(self):
        if self.n_jammer is None:
            self.path_loss = np.zeros(shape=(len(self.uav_positions),len(self.uav_positions)))
            for i in range(len(self.uav_positions)):
                for j in range(len(self.uav_positions)):
                    self.path_loss[i][j] = self.get_path_loss(self.uav_positions[i], self.uav_positions[j])
        else:
            self.path_loss = np.zeros(shape=(len(self.jammer_positions),len(self.jammer_positions)))
            for i in range(len(self.jammer_positions)):
                for j in range(len(self.jammer_positions)):
                    self.path_loss[i][j] = self.get_path_loss(self.jammer_positions[i], self.jammer_positions[j])

    def get_path_loss(self, x, y):
        distance = np.sqrt(np.sum(np.square(x - y))+1e-6)
        # 这个是路径损耗模型，可以根据需要修改
        loss_dB = 103.8 + 20.9 * np.log10(distance*1e-3)
        return loss_dB

    def update_fast_fading(self): # 瑞利衰落
        if self.n_jammer is None:
            h = (np.random.normal(size=(self.n_uav, self.n_uav, self.channels)) + 1j * np.random.normal(size=(self.n_uav, self.n_uav, self.channels))) / np.sqrt(2)
        else:
            h = (np.random.normal(size=(self.n_jammer, self.n_uav, self.channels)) + 1j * np.random.normal(size=(self.n_jammer, self.n_uav, self.channels))) / np.sqrt(2)
        self.fast_fading_dB = 20 * np.log10(np.abs(h))


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

class UAV(gym.Env):
    """
    
    """

    def __init__(
        self, seed=None, length=500, width=250, low_height=60, high_height=120, moving_smooth_sigma=0.2,
        uav_power_list=[36, 33, 30, 27], jammer_power=30, sig2_dB=-114, uav_dBi=3, jammer_dBi=3, uavNoiseFactor=9, bandwidth=1.8 * 1e6,
        n_clusters=3, n_cluster_members=2
    ):
        self.seed(seed)
        # 设置状态空间范围
        self.length = length
        self.width = width
        self.low_height = low_height
        self.high_height = high_height
        # uav移动参数
        self.moving_smooth_factor = 0.8
        self.moving_smooth_sigma = moving_smooth_sigma
        # uav, jammer的参数
        self.uav_power_list = uav_power_list
        self.jammer_power = jammer_power #TODO 这个可以改成一个列表，表示多个干扰器的功率
        self.sig2 = 10 ** (sig2_dB / 10) #! 这个是什么, 我还不知道
        self.uav_dBi = uav_dBi
        self.jammer_dBi = jammer_dBi
        self.uav_Noise_Factor = uavNoiseFactor
        self.bandwidth = bandwidth
        # 移动模型
        self.n_clusters = n_clusters # UAV的集群数
        self.n_cluster_members = n_cluster_members # 每个集群的成员数(不包括簇头)
        self.n_members = self.n_clusters * self.n_cluster_members # 总的成员数
        self.n_uavs = self.n_members + self.n_clusters
        self.n_channel_users = self.n_cluster_members #! 每个信道的用户数 (这个是不是有问题?可能还要加一)
        self.n_