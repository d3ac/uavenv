import gym
from gym import spaces
import numpy as np
import random


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
        # uav, jammer的参数
        self.uav_power_list = uav_power_list
        self.jammer_power = jammer_power #TODO 这个可以改成一个列表，表示多个干扰器的功率
        self.sig2 = 10 ** (sig2_dB / 10) #! 这个是什么, 我还不知道
        self.uav_dBi = uav_dBi
        self.jammer_dBi = jammer_dBi
        self.uav_Noise_Factor = uavNoiseFactor
        self.bandwidth = bandwidth
        # uav & jammer & channel数量
        self.n_clusters = n_clusters # UAV簇的个数
        self.n_cluster_members = n_cluster_members # 每个簇的成员数(不包括簇头)
        self.n_members = self.n_clusters * self.n_cluster_members # 总的成员数
        self.n_uavs = self.n_members + self.n_clusters
        self.n_channel_users = self.n_cluster_members #! 每个信道的用户数 (这个是不是有问题?可能还要加一)
        self.n_jammers = n_jammers # 干扰器的数量
        self.n_channels = n_channels #! 信道的数量 瞅瞅这个在那里用到了, 有什么用, 原来代码里面的注释是直接算出来的
        # 观测模型
        self.prob_missed_detection = 0   # 漏检概率
        self.prob_false_alarm = 0        # 误检概率
        # 移动参数
        self.moving_smooth_factor = 0.8
        self.moving_smooth_sigma = moving_smooth_sigma
        self.max_rp_distance = max_rp_distance # 簇内的参考节点围绕中心运动时的最大允许距离
        self.max_uav_distance = max_uav_distance # 簇内的无人机围绕中心运动时的最大允许距离
        self.is_jammer_moving = is_jammer_moving # 干扰器是否移动
        self.interference_type = interference_type # 干扰类型
        self.policy = policy #! 策略, 这个我觉得不应该放在这个地方, 应该放在外面, 后面改改
        # 保存cluster数据
        self.uav_list = list(range(self.n_uavs))
        self.master_list = random.sample(self.uav_list, k=self.n_clusters) # 随机选择簇头
        self.member_list = list(set(self.uav_list) - set(self.master_uav_list)) #成员列表
        self.uav_pairs = np.zeros([self.n_clusters], dtype=np.int32)
        self.uav_clusters = np.zeros([self.n_clusters, self.n_cluster_members, 2], dtype=np.int32)
        # action
        self.action_range = self.n_channels * len(self.uav_power_list) #! 选择信道和功率,组合成为动作空间, 这个可以改改
        self.action_dim = self.action_range ** self.n_channel_users #!很怪
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.n_clusters)] # 也可以改改
        # reward
        self.uav_hopping_cnt = np.zeros([self.n_clusters], dtype=np.int32) #! 跳频次数 不过我觉得可以改成如果选择跳频再定义他
        self.energy_reward = 0
        self.hopping_reward = 0
        #!他这个后面还有依托东西
    
    def a(self):
        self.observed_state_list = []