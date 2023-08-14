import gym
from gym import spaces
import numpy as np
import random
from UAVenv.uav.channel import Channel, JammerChannel, ClusterChannel
from UAVenv.utils import obs_Normalizer


class systemEnv(gym.Env):
    """
    """
    def __init__(
        self, episode_max=300, n_clusters=3, n_channels=6, n_slaves=3, n_jammers=3, area_type="small_and_medium_size_cities",
        jamming_mode='Markov', fc=800*1e6, hb=50, hm=20, power_list=[36, 33, 30, 27], jammer_power = 30,
        xlim=1000, ylim=1000, zlim_max=200, zlim_min=50, max_radius=50, master_velocity=10, slave_velocity=10, moving_factor=0.1, dt=0.1,
        training=True, test_obs=None, **kwargs
    ):
        # 定义模型参数
        self.episode_cnt = 1
        self.episode_max = episode_max
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
        self.power_list = power_list
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
        # 定义观测空间
        # part1 = spaces.Box(low=0, high=self.n_channels-1, shape=(self.n_slaves+self.n_clusters,)) # channel
        # part2 = spaces.Box(low=min(power_list), high=max(power_list), shape=(self.n_slaves,)) # power
        # part3 = spaces.Box(low=0, high=max(xlim, ylim, zlim_max, zlim_min), shape=(self.n_slaves+self.n_clusters, 3)) # position
        # part4 = spaces.Box(low=0, high=1,shape=(self.n_channels,))
        # self.observation_space = [spaces.Tuple((part1, part2, part3, part4)) for _ in range(self.n_clusters)]
        channel_num = self.n_slaves + self.n_clusters
        pow_num = self.n_slaves + self.n_clusters
        position_num = (self.n_slaves + self.n_clusters) * 3
        SNR_num = self.n_slaves + self.n_clusters - 1
        single = spaces.Box(low=-1, high=1, shape=(channel_num + pow_num + position_num + SNR_num,))
        self.observation_space = spaces.Tuple((single for _ in range(self.n_clusters)))
        # 定义动作空间
        # part1 = spaces.Box(low=0, high=self.n_channels-1, shape=(self.n_clusters,))
        # part2 = spaces.Box(low=min(power_list), high=max(power_list), shape=(self.n_clusters,))
        # part3 = spaces.Box(low=0, high=self.n_channels-1, shape=(self.n_slaves,))
        # part4 = spaces.Box(low=min(power_list), high=max(power_list), shape=(self.n_slaves,))
        # self.action_space = [spaces.Tuple((part1, part2, part3, part4)) for _ in range(self.n_clusters)]
        master = [self.n_channels for _ in range(self.n_clusters)] + [len(power_list) for _ in range(self.n_clusters)]
        slaves = [self.n_channels for _ in range(self.n_slaves)] + [len(power_list) for _ in range(self.n_slaves)]
        self.action_space = spaces.Tuple((spaces.MultiDiscrete(master + slaves) for _ in range(self.n_clusters)))
        # 定义
        self.channel = Channel(n_clusters, n_channels, n_slaves, area_type, fc, hb, hm, power_list, **kwargs)
        self.jammer = JammerChannel(n_jammers, n_channels, jamming_mode, area_type, fc, hb, hm, jammer_power, **kwargs)
        # normalize
        self.obs_normalizer = obs_Normalizer(training, test_obs)
    
    def calc_SNR(self):
        SNR = [[] for _ in range(self.n_clusters)]
        for i in range(self.n_clusters):
            for j in range(self.n_slaves):
                jam, gain = 1e-3, 0
                jam += self.channel.Clusters[i].cluster_pathloss_interference(j)  # cluster内部通信的干扰
                jam += self.channel.cluster_pathloss_interference_slaves(j, i)    # cluster之间master to master通信的干扰
                #!还可以加上 cluster之间master to slave通信的干扰
                jam += self.jammer.jamming_pathloss_slaves(self.channel.Clusters[i].position[j+1], self.channel.Clusters[i].position[0], self.channel.Clusters[i].channel_select[j]) # jammer对cluster内部通信的干扰
                gain = self.channel.Clusters[i].channel_power[j] * (10 ** (self.channel.Clusters[i].pathloss[j] / 10))
                SNR[i].append(10 * np.log10(gain / jam))
            jam = self.channel.calc_jam(i, self.jammer)
            gain = self.channel.calc_gain(i)
            SNR[i].extend(list(10 * np.log10(np.array(gain) / np.array(jam))))
        return SNR

    def observe(self):
        channel, power, position = self.channel.observe()
        SNR = self.calc_SNR() #! 这个实际上应该是 power和速率的某个函数
        channel, power, position, SNR = np.array(channel), np.array(power), np.array(position), np.array(SNR)
        return (channel, power, position, SNR)

    def reset(self, seed=None):
        # 设置种子
        self.seed(seed)
        # jammer
        self.jammer.reset_JammerChannel()
        self.channel.reset_Channel()
        # master和slaves通信信道
        self.channel.act(init=True)
        channel, power, position, SNR = self.observe()
        observation = self.obs_normalizer.merge_obs(channel, power, position, SNR)
        self.episode_cnt = 1
        return observation, {}

    def render(self):
        pass

    def seed(self, seed=None): # 传入None就是每局都是随机值
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def reward(self, SNR, frequency_hopping_cnt): #!还有改进的空间, 还有一件事就是这个reward是不是负数?为了梯度下降
        reward = np.sum(SNR, axis=1) - frequency_hopping_cnt
        return (reward - 113.65684593479375) / 70.85946069952426

    @property
    def done(self):
        self.episode_cnt += 0.5 # 因为每个episode都会调用两次, 一次是done一次是truncated
        return np.array([self.episode_cnt >= self.episode_max for _ in range(self.n_clusters)])

    @property
    def trucated(self):
        return self.done

    def step(self, actions):
        self.channel.position_step()
        self.jammer.position_step()
        frequency_hopping_cnt = self.channel.act(actions)
        channel, power, position, SNR = self.observe()
        reward = self.reward(SNR, frequency_hopping_cnt)
        observation = self.obs_normalizer.merge_obs(channel, power, position, SNR)
        return observation, reward, self.trucated, self.done, {}
    
    def generate_random_actions(self):
        actions = []
        for i in range(self.n_clusters):
            channel = np.random.randint(0, self.n_channels, (self.n_clusters,))
            power = np.random.randint(0, len(self.power_list), (self.n_clusters,))
            master = np.concatenate((channel, power), axis=0)
            
            channel = np.random.randint(0, self.n_channels, (self.n_slaves,))
            power = np.random.randint(0, len(self.power_list), (self.n_slaves,))
            slaves = np.concatenate((channel, power), axis=0)
            actions.append(np.concatenate((master, slaves), axis=0))
        return np.array(actions)