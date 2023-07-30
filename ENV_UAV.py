from __future__ import division
from collections import deque
import math
import numpy as np
import random
import gym
from gym import spaces
from copy import deepcopy
from scipy.special import comb, perm
from itertools import combinations, permutations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import seaborn as sns

#无人机，干扰机都和无人机通信, 无人机之间交互Q表,接收机与发射机之间有距离限制
class UAVchannels:
    def __init__(self, n_uav, n_channel):
        self.h_bs = 25  # BS antenna height
        self.h_uav = 1.5  # uav antenna height
        self.n_uav = n_uav
        self.n_channel = n_channel

    def update_positions(self, positions):
        self.positions = positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.positions[j])

    #无人机之间的位置路径损耗
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d3 = abs(position_A[2] - position_B[2])
        distance = np.sqrt(d1**2 + d2**2 + d3**2) + 0.001
        PL_los = 103.8 + 20.9*np.log10(distance*1e-3)
        return PL_los

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_uav, self.n_uav, self.n_channel)) + 1j *
                              np.random.normal(size=(self.n_uav, self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))

class Jammerchannels:
    def __init__(self, n_jammer, n_uav, n_channel):
        self.h_jammer = 1.5  # jammer antenna height
        self.h_uav = 1.5  # uav antenna height
        self.n_jammer = n_jammer
        self.n_uav = n_uav
        self.n_channel = n_channel

    def update_positions(self, positions, uav_positions): #!注意这个地方合并了之后,传入的参数的顺序变了
        self.positions = positions
        self.uav_positions = uav_positions

    def update_pathloss(self):
        self.PathLoss = np.zeros(shape=(len(self.positions), len(self.uav_positions)))
        for i in range(len(self.positions)):
            for j in range(len(self.uav_positions)):
                self.PathLoss[i][j] = self.get_path_loss(self.positions[i], self.uav_positions[j])

    # position A表示干扰机的位置 position B表示无人机的位置
    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d3 = abs(position_A[2] - position_B[2])
        distance = np.sqrt(d1 ** 2 + d2 ** 2 + d3 ** 2) + 0.001
        PL_los = 103.8 + 20.9 * np.log10(distance * 1e-3)
        return PL_los

    def update_fast_fading(self):
        h = 1 / np.sqrt(2) * (np.random.normal(size=(self.n_jammer, self.n_uav, self.n_channel)) +
                              1j * np.random.normal(size=(self.n_jammer, self.n_uav, self.n_channel)))
        self.FastFading = 20 * np.log10(np.abs(h))

class UAV:
    def __init__(self, start_position, start_direction, start_velocity, start_p):
            self.position = start_position
            self.direction = start_direction
            self.velocity = start_velocity
            self.p = start_p  # 无人机与地平面的夹角度
            
            self.uav_velocity = []
            self.uav_direction = []
            self.uav_p = []
            self.destinations = []
            self.connections = []

class Jammer:
    def __init__(self, start_position, start_direction, velocity, start_p):
            self.position = start_position
            self.direction = start_direction
            self.velocity = velocity
            self.p = start_p  # 无人机与地平面的夹角度
            self.jammer_velocity = []
            self.jammer_direction = []
            self.jammer_p = []

class RP:  # 参考节点
    def __init__(self, start_position):
            self.position = start_position
            self.connections = []

class DRQN(nn.Module):
    def __init__(self, action):
        super(DRQN, self).__init__()
        #self.env = Environ()
        self.lstm_i_dim = 16    # input dimension of LSTM
        self.lstm_h_dim = 16    # output dimension of LSTM
        self.lstm_N_layer = 1   # number of layers of LSTM
        self.input = 4   #网络总输入:每个无人机簇的状态，8个快衰落（2个成员x4个信道）、6个信道（6个成员）、6个功率（6个成员）、1个干扰机信道状态
        self.output = action     #网络总输出
        self.lstm = nn.LSTM(input_size=self.lstm_i_dim, hidden_size=self.lstm_h_dim, num_layers=self.lstm_N_layer)
        self.fc1 = nn.Linear(self.input, self.lstm_i_dim)
        self.fc2 = nn.Linear(self.lstm_i_dim, 16)
        self.fc3 = nn.Linear(self.lstm_h_dim, 16)
        self.fc4 = nn.Linear(16, self.output)

    def forward(self, x, hidden):
        h1 = self.fc1(x)
        h2, new_hidden = self.lstm(h1, hidden)
        h3 = F.relu(self.fc3(h2))
        h4 = self.fc4(h3)
        return h4, new_hidden

class ReplayMemory(object):
    def __init__(self, max_epi_num=50, max_epi_len=300):
        # capacity is the maximum number of episodes
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.memory = deque(maxlen=self.max_epi_num)   #双端队列deque容器为一个给定类型的元素进行线性处理，像向量一样，它能够快速地随机访问任一个元素，并且能够高效地插入和删除容器的尾部元素。但它又与vector不同，deque支持高效插入和删除容器的头部元素
        self.is_av = False
        self.current_epi = 0
        self.memory.append([])

    def reset(self):
        self.current_epi = 0
        self.memory.clear()
        self.memory.append([])

    def create_new_epi(self):
        self.memory.append([])
        self.current_epi = self.current_epi + 1
        if self.current_epi > self.max_epi_num - 1:
            self.current_epi = self.max_epi_num - 1

    def remember(self, state, action, reward):
        if len(self.memory[self.current_epi]) < self.max_epi_len:
            self.memory[self.current_epi].append([state, action, reward])

    def sample(self):
        epi_index = random.randint(0, len(self.memory)-2)
        if self.is_available():
            return self.memory[epi_index]
        else:
            return []

    def size(self):
        return len(self.memory)

    def is_available(self):
        self.is_av = True
        if len(self.memory) <= 1:
            self.is_av = False
        return self.is_av

    def print_info(self):
        for i in range(len(self.memory)):
            print('epi', i, 'length', len(self.memory[i]))

class Agent:
    def __init__(self, i, action_dim, max_epi_num=50, max_epi_len=300):
        self.name = 'agent%d' % i
        self.max_epi_num = max_epi_num
        self.max_epi_len = max_epi_len
        self.drqn = DRQN(action_dim)
        self.N_action = action_dim
        self.buffer = ReplayMemory(max_epi_num=self.max_epi_num, max_epi_len=self.max_epi_len)
        self.gamma = 0.9
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.drqn.parameters(), lr=1e-3)  # 创建一个标准来测量输入x和目标y中每个元素之间的均方误差

    def remember(self, state, action, reward):
        state = np.array(state)
        self.buffer.remember(state, action, reward)

    def train(self):
        if self.buffer.is_available():
            memo = self.buffer.sample()
            obs_list = []
            action_list = []
            reward_list = []
            for i in range(len(memo)):
                obs_list.append(memo[i][0])
                action_list.append(memo[i][1])
                reward_list.append(memo[i][2])
            # obs_list = self.img_list_to_batch
            obs_list = (np.array(obs_list)).reshape(-1, 1, self.drqn.input)
            obs_list = torch.FloatTensor(obs_list)
            hidden = (Variable(torch.zeros(1, 1, 16).float()), Variable(torch.zeros(1, 1, 16).float()))
            Q, hidden = self.drqn.forward(obs_list, hidden)
            Q_est = Q.clone()
            for t in range(len(memo) - 1):
                max_next_q = torch.max(Q_est[t+1, 0, :]).clone().detach()
                Q_est[t, 0, action_list[t]] = reward_list[t] + self.gamma * max_next_q
            T = len(memo) - 1
            Q_est[T, 0, action_list[T]] = reward_list[T]

            loss = self.loss_fn(Q, Q_est)
            self.optimizer.zero_grad()#梯度初始化为零
            loss.backward()# 反向传播求梯度
            self.optimizer.step()# 更新所有参数

    def get_action(self, obs, hidden, epsilon):
        obs = np.array(obs)
        obs = obs.reshape(-1, 1, self.drqn.input)
        obs = torch.FloatTensor(obs)#转换为一个向量
        if random.random() > epsilon:
            q, new_hidden = self.drqn.forward(obs, hidden)
            action = q[0].max(1)[1].data[0].item()
        else:
            q, new_hidden = self.drqn.forward(obs, hidden)
            action = random.randint(0, self.N_action-1)
        return action, new_hidden
#!为什么不把multiagent当作multitask呢
class Environ(gym.Env):
    def __init__(self):
        self.seed_set()
        #设置运动范围
        self.length = 500  # 1000
        self.width = 250  # 500
        self.low_height = 60
        self.high_height = 120

        self.moving_smooth_factor = 0.8
        self.sigma = 0.2

        #无人机、干扰机各种参数
        self.uav_power_list = [36, 33, 30, 27]  # dBm = 1W-2W          #10-23
        self.uav_power_min = min(self.uav_power_list)
        self.uav_power_max = max(self.uav_power_list)
        self.jammer_power = 30  # dBm
        self.sig2_dB = -114  # dBm       Noise power
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.uavAntGain = 3  # dBi       uav antenna gain
        self.uavNoiseFigure = 9  # dB    uav receiver noise figure
        self.jammerAntGain = 3  # dBi       jammer antenna gain
        self.bandwidth = 1.8 * 1e+6  # 1.5 * 1e+6   # Hz

        #数据传输过程的参数
        self.data_size = 8 * 1024 ** 2
        self.t_Rx = 0.98  # 传输时间,单位都是s
        self.t_collect = 0.5  # 收集数据
        self.timestep = 0.2  # 频谱感知，选动作 + ACK + 学习
        self.timeslot = self.t_Rx + self.timestep  # 时隙
        self.t_uav = 0
        self.jammer_start = 0.2  # 干扰机开始干扰时间
        self.t_dwell = 2.28  # 干扰机扫频停留时间
        self.t_jammer = 0

        #参考点群移动模型
        self.n_ch = 3  # UAV簇头个数
        self.n_cm_for_a_ch = 2  # 每个ch的簇成员个数
        self.n_cm = self.n_ch * self.n_cm_for_a_ch  # UAV簇成员个数
        self.n_uav = self.n_ch + self.n_cm  # number of UAVs
        self.n_rp = self.n_uav  # 簇成员的参考节点个数
        self.n_des = self.n_cm_for_a_ch  # 每个ch的通信目标数
        self.n_jammer = 2  # number of jammers
        self.n_channel = 6  # int(self.n_ch+self.n_jammer-1)  # number of channels
        self.channel_indexes = np.arange(self.n_channel) #! 我觉得到时候再写 

        self.p_md = 0  # 漏警概率
        self.p_fa = 0  # 虚警概率

        self.max_distance1 = 99 #群组内节点的RP围绕逻辑中心运动时最大允许的半径
        self.max_distance2 = 1  #每个节点围绕其RP运动时最大允许的半径
        self.is_jammer_moving = True
        self.type_of_interference = "saopin"
        # "markov"首先干扰机通过检测智能体的主要变化,
        # 识别agent的工作模式并且建立工作模式状态转移的马尔可夫链,
        # 然后利用合适的算法对建立的agent工作模式转移马尔可夫链计算转移概率,
        # 最后将agent工作模式转移概率转化为矩阵形式就对agent下一个工作模式进行预测,
        # 从而使得干扰机能够最大限度的对agent进行干扰
        self.policy = None  # 对应算法
        self.training = True

        self.uav_list = list(np.arange(self.n_uav))
        self.ch_list = random.sample(self.uav_list, k=self.n_ch) #由于随机数种子的原因，每次都选择2、7、3作为簇头
        self.cm_list = list(set(self.uav_list) - set(self.ch_list))
        self.rp_cm_list = self.cm_list #! 讲道理这个没用
        self.uav_pairs = np.zeros([self.n_ch, self.n_des, 2], dtype=np.int32)
        self.uav_clusters = np.zeros([self.n_ch, self.n_cm_for_a_ch, 2], dtype=np.int32)

        self.channel_range = self.n_channel
        self.power_range = len(self.uav_power_list)
        self.action_range = self.channel_range * self.power_range
        self.action_dim = self.action_range ** self.n_des
        self.action_space = [spaces.Discrete(self.action_dim) for _ in range(self.n_ch)]  # 网络#gym.spaces.Discrete（）功能：创建⼀个离散的n维空间，n为整数

        #与奖励相关参数
        self.uav_jump_count = np.zeros([self.n_ch], dtype=np.int32)
        self.rew_energy = 0
        self.rew_jump = 0

        #添加智能体
        n_episode = 1500
        n_steps = 1000
        self.agents = [Agent(i, self.action_dim, max_epi_num=200, max_epi_len=n_steps) for i in range(self.n_ch)]

        self.n_step = 0

        # 网络
        self.all_observed_states()
        self.reset()
        self.state_dim = len(self.get_state()[0])
        self.observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(self.state_dim,)) for _ in range(self.n_ch)]

    def seed_set(self, seed=2020):
        random.seed(seed)
        np.random.seed(seed)

    def all_observed_states(self):
        self.observed_state_list = []
        observed_state = 0
        self.all_observed_states_list = []
        if self.type_of_interference == "saopin":
            self.step_forward = 1
            if self.p_md == 0 and self.p_fa == 0:
                self.observed_state_dim = int(comb(self.n_channel, self.n_jammer))      #从n_channel中无重复无序的选取n_jammer
                self.all_observed_states_list.extend(list(combinations(self.channel_indexes, self.n_jammer)))

            elif self.p_md > 0 and self.p_fa == 0:  # 漏警
                #!没看懂是怎么漏的
                for i in range(self.n_jammer + 1):
                    self.observed_state_dim += int(comb(self.n_channel, i))
                    self.all_observed_states_list.extend(list(combinations(self.channel_indexes, i)))

            elif self.p_md == 0 and self.p_fa > 0:  # 虚警
                for i in range(self.n_jammer, self.n_channel + 1):
                    self.observed_state_dim += int(comb(self.n_channel, i))
                    self.all_observed_states_list.extend(list(combinations(self.channel_indexes, i)))
            #!为什么没有两个都可以漏的概率?

        elif self.type_of_interference == "markov":
            self.all_jammer_states_list = []
            self.jammer_state_dim = int(perm(self.n_channel, self.n_jammer))#perm()全排列
            self.all_jammer_states_list.extend(list(permutations(self.channel_indexes, self.n_jammer)))#permutations给定一个数组集合，返回所有可能的排列。
            self.p_trans = np.random.uniform(0, 1, [self.jammer_state_dim, self.jammer_state_dim])#从[0,1)的均匀分布里随机取self.jammer_state_dim*self.jammer_state_dim个数
            p_trans_sum = np.sum(self.p_trans, axis=1)#将矩阵p_trans每一行的数相加得到列向量
            for i in range(self.jammer_state_dim):
                for j in range(self.jammer_state_dim):
                    self.p_trans[i][j] = self.p_trans[i][j]/p_trans_sum[i]#将矩阵p_trans的每个元素用每一行元素之和进行归一化
            if self.p_md == 0 and self.p_fa == 0:
                self.observed_state_dim = int(comb(self.n_channel, self.n_jammer))#comb返回从n_channel种可能性中选择n_jammer个无序结果的方式数量，无重复，也称为组合。
                self.all_observed_states_list.extend(list(combinations(self.channel_indexes, self.n_jammer)))

    def renew_uavs(self): # 初始化簇头
        for i in range(self.n_ch):
            # 更新簇头无人机的速度、方向、夹角
            ch_id = self.ch_list[i]
            start_velocity = random.uniform(10, 20)
            start_direction = random.uniform(0, 2 * math. pi)
            start_p = random.uniform(0, 2 * math. pi)
            # 更新簇头无人机的三维坐标
            ch_xpos = random.uniform(0.0, self.length)
            ch_ypos = random.uniform(0.0, self.width)
            ch_zpos = random.uniform(self.low_height, self.high_height)
            start_position = [ch_xpos, ch_ypos, ch_zpos]

            self.uavs[ch_id] = UAV(start_position, start_direction, start_velocity, start_p)
            self.rps[ch_id] = RP(start_position)

            self.uavs[ch_id].uav_velocity.append(start_velocity)
            self.uavs[ch_id].uav_direction.append(start_direction)
            self.uavs[ch_id].uav_p.append(start_p)

    def renew_uav_clusters(self):
        cm_list = deepcopy(self.cm_list) #将复制对象完全复制一遍，并作为一个独立的新个体单元存在。即使改变被复制对象，deepcopy新个体也不会发生变化
        rp_cm_list = deepcopy(self.rp_cm_list)
        for i in range(self.n_ch):
            ch_id = self.ch_list[i]
            cms = random.sample(cm_list, k=self.n_cm_for_a_ch)
            rps = cms
            for j in range(self.n_cm_for_a_ch):
                self.uav_clusters[i][j][0] = ch_id
                self.uav_clusters[i][j][1] = cms[j]
                self.uav_pairs[i][j][0] = ch_id
                self.uav_pairs[i][j][1] = cms[j]
                self.uavs[ch_id].connections.append(cms[j])
                self.uavs[ch_id].destinations.append(cms[j])

                ch_pos = [self.uavs[ch_id].position[0], self.uavs[ch_id].position[1], self.uavs[ch_id].position[2]]

                # 参考节点的位置设定
                R1 = random.uniform(0.0, self.max_distance1)
                d1 = random.uniform(0.0, 2 * math.pi)
                p1 = random.uniform(0.0, 2 * math.pi)

                rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                while ((rp_xpos < 0) or (rp_xpos > self.length) or (rp_ypos < 0) or (rp_ypos > self.width) or (rp_zpos < self.low_height) or (rp_zpos > self.high_height)):
                    R1 = random.uniform(0.0, R1)
                    d1 = random.uniform(0.0, 2 * math.pi)
                    p1 = random.uniform(0.0, 2 * math.pi)

                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                # 簇内节点的位置设定
                R2 = random.uniform(0.0, self.max_distance2)
                d2 = random.uniform(0.0, 2 * math.pi)
                p2 = random.uniform(0.0, 2 * math.pi)

                cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
                cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
                cm_zpos = rp_zpos + R2 * math.sin(p2)

                while ((cm_xpos < 0) or (cm_xpos > self.length) or (cm_ypos < 0) or (cm_ypos > self.width) or (cm_zpos < self.low_height) or (cm_zpos > self.high_height)):
                    # 簇内节点的位置设定
                    R2 = random.uniform(0.0, self.max_distance2)
                    d2 = random.uniform(0.0, 2 * math.pi)
                    p2 = random.uniform(0.0, 2 * math.pi)

                    cm_xpos = rp_xpos + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_ypos + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_zpos + R2 * math.sin(p2)


                start_position = [cm_xpos, cm_ypos, cm_zpos]
                start_position_rp = [rp_xpos, rp_ypos, rp_zpos]
                start_direction = None
                start_velocity = None
                start_p = None

                self.uavs[cms[j]] = UAV(start_position, start_direction, start_velocity, start_p)
                self.uavs[cms[j]].connections.append(ch_id)
                self.uavs[cms[j]].destinations.append(ch_id)
                self.rps[rps[j]] = RP(start_position_rp)

            cm_list = list(set(cm_list) - set(cms))
            rp_cm_list = list(set(rp_cm_list) - set(rps))

        # print(self.uav_clusters)

    def renew_jammers(self):
        if self.is_jammer_moving:
            for i in range(self.n_jammer):
                start_velocity = random.uniform(10.0, 20.0)
                start_direction = random.uniform(0, 2 * math.pi)
                start_p = random.uniform(0, 2 * math.pi)

                xpos = random.uniform(0.0, self.length)
                ypos = random.uniform(0.0, self.width)
                zpos = random.uniform(self.low_height, self.high_height)
                start_position = [xpos, ypos, zpos]

                self.jammers.append(Jammer(start_position, start_direction, start_velocity, start_p))
                self.jammers[i].jammer_velocity.append(start_velocity)
                self.jammers[i].jammer_direction.append(start_direction)
                self.jammers[i].jammer_p.append(start_p)

    def renew_neighbors_of_uavs(self):#不确定有用到
        for i in range(len(self.uavs)):
            self.uavs[i].neighbors = []
        z = np.array([[complex(c.position[0], c.position[1], c.position[2]) for c in self.uavs]])
        Distance = abs(z.T - z)
        for i in range(len(self.uavs)):
            sort_idx = np.argsort(Distance[:, i])  # 返回数组值从小到大的索引值
            for j in range(len(sort_idx)-1):
                self.uavs[i].neighbors.append(sort_idx[j + 1])

    def new_random_game(self):
        # 一个发送机若有多个通信目标，每个元素是智能体为每个通信目标分配的信道，假设各不相同
        self.uav_channels = np.zeros([self.n_ch, self.n_des], dtype=np.int32)   # 每个智能体观察到的全局动作（假设智能体可以观察到其他智能体已经完成的动作）
        self.uav_powers = np.zeros([self.n_ch, self.n_des], dtype=np.int32)
        self.uav_jump_count = np.zeros([self.n_ch], dtype=np.int32)
        for i in range(self.n_ch):#为每个簇的每个簇成员随机选择信道和功率
            for j in range(self.n_des):
                self.uav_channels[i][j] = random.randint(0, self.n_channel - 1)  #包括上下限
                self.uav_powers[i][j] = self.uav_power_list[random.randint(0, len(self.uav_power_list) - 1)]  # 包括上下限
        if self.type_of_interference == "saopin":
            self.jammer_channels = random.sample(range(0, self.n_channel), k=self.n_jammer)  #不包括 stop

        elif self.type_of_interference == "markov":
            self.jammer_channels = random.choices(self.all_jammer_states_list, k=1)[0]
        self.jammer_channels_list = []
        self.jammer_index_list = []
        #如果传输阶段先后干扰两个信道,0是后半段 改变后的信道，1是前半段 改变前的信道
        self. jammer_time = np.zeros([2])  #! 每个干扰机在传输阶段最多先后干扰两个信道，目前假设各个干扰机时间线相同

        #print("jammer_channels", self.jammer_channels)

        self.uavs = [None] * self.n_uav
        self.rps = [None] * self.n_rp
        self.jammers = []
        self.renew_uavs()       #更新簇头无人机的位置、方向和速度、夹角
        self.renew_uav_clusters()       #更新簇头和簇内无人机的位置 并且保证簇内成员与簇头在通信范围内
        self.renew_jammers()        #更新干扰机的位置
        self.UAVchannels = UAVchannels(self.n_uav, self.n_channel)
        self.Jammerchannels = Jammerchannels(self.n_jammer, self.n_uav, self.n_channel)
        self.renew_channels()

    def get_state(self):
        if self.policy == "Q_learning":
            uav_state = 0
            channels_observed = tuple(sorted(self.jammer_channels))
            for i in range(self.n_ch):
                uav_state += self.uav_channels[i] * (self.action_range ** i)
            observed_state_idx = self.all_observed_states_list.index(channels_observed)
            joint_state = uav_state * self.observed_state_dim + observed_state_idx
            return joint_state

        elif self.policy == "Sensing_Based_Method":
            if not isinstance(self.jammer_channels, list):
                jammer_channels = list(self.jammer_channels)
            else:
                jammer_channels = self.jammer_channels
            if self.p_md == 0 and self.p_fa == 0:
                channels_observed = np.zeros([self.n_channel], dtype=np.int32)
                channels_observed[jammer_channels] = 1
            else:
                channels_observed = np.zeros([self.n_channel], dtype=np.int32)
                for i in range(self.n_channel):
                    if i in jammer_channels:
                        if random.random() < self.p_md:
                            channels_observed[i] = 0  # 漏警
                        else:
                            channels_observed[i] = 1  # 发现干扰
                    else:
                        if random.random() < self.p_fa:
                            channels_observed[i] = 1  # 虚警
                        else:
                            channels_observed[i] = 0  # 发现未干扰
            return channels_observed

        else:
            joint_state = []
            csi = np.zeros([self.n_ch, self.n_des, self.n_channel])
            self.uav_jump_count = np.zeros([self.n_ch], dtype=np.int32)

            for i in range(self.n_ch):
                for j in range(self.n_des):
                    tra_id = self.uav_pairs[i][j][0]        # 接收机和发射机
                    rec_id = self.uav_pairs[i][j][1]
                    csi[i][j] = (self.UAVchannels_with_fastfading[tra_id][rec_id] - 80) / 60
            uav_channels = self.uav_channels / self.n_channel #! 这里也是
            uav_powers = self.uav_powers / self.uav_power_max #! 这样归一化有问题呀
            jammer_channels = np.asarray([x / self.n_channel for x in self.jammer_channels])      #for item in list,获取列表中的每一项
            for i in range(self.n_ch):
                joint_state.append(np.concatenate((csi[i].reshape([-1]), uav_channels.reshape([-1]),
                                                   uav_powers.reshape([-1]), jammer_channels)).astype(np.float32))

            return joint_state

    def compute_reward(self, i, j, other_channel_list, other_index_list):
        uav_interference = 0   # 其他的transmitter对transmitter i的干扰
        uav_interference_from_jammer0 = 0    #后半段干扰机干扰
        uav_interference_from_jammer1 = 0   #前半段干扰机干扰

        transmitter_idx = self.uav_pairs[i][j][0]
        receiver_idx = self.uav_pairs[i][j][1]
        uav_signal = 10 ** ((self.uav_powers[i][j] - self.UAVchannels_with_fastfading[transmitter_idx, receiver_idx, self.uav_channels[i][j]] +
                                2 * self.uavAntGain - self.uavNoiseFigure) / 10)
        if self.uav_channels[i][j] in other_channel_list:
            index = np.where(other_channel_list == self.uav_channels[i][j])
            for k in range(len(index)):
                transmitter_idx = other_index_list[index[k][0]]
                uav_interference += 10 ** ((self.uav_powers[i][j] - self.UAVchannels_with_fastfading[transmitter_idx, receiver_idx, self.uav_channels[i][j]] +
                                               2 * self.uavAntGain - self.uavNoiseFigure) / 10)     #无人机内部干扰

        if self.uav_channels[i][j] in self.jammer_channels_list:
            idx = np.where(self.jammer_channels_list == self.uav_channels[i][j])
            if self.jammer_time[0] == self.t_Rx or self.jammer_time[0] == self.t_Rx-self.jammer_start:     # 传输时间干扰机没换信道
                for m in range(len(idx)):
                    jammer_idx = self.jammer_index_list[idx[m][0]]
                    uav_interference += 10 ** ((self.jammer_power - self.Jammerchannels_with_fastfading[jammer_idx, receiver_idx, self.uav_channels[i][j]] +
                                                   self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)
                uav_rate = np.log2(1 + np.divide(uav_signal, (uav_interference + self.sig2)))
                uav_rate *= self.bandwidth
                transmit_time = self.data_size / uav_rate


            else:    # 传输时间干扰机换了信道，判断干扰了前半段还是后半段
                for m in range(len(idx)):
                    jammer_idx = self.jammer_index_list[idx[m][0]]
                    if idx[m][0] % 2 == 0:   # 后半段(self.jammer_channels_list先存入的后半段干扰信道序号）
                        uav_interference_from_jammer0 += 10 ** ((self.jammer_power - self.Jammerchannels_with_fastfading[jammer_idx, receiver_idx, self.uav_channels[i][j]] +
                                                                  self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)

                uav_rate = np.log2(1 + np.divide(uav_signal, (uav_interference + uav_interference_from_jammer0 + self.sig2)))
                uav_rate *= self.bandwidth
                transmit_time1 = self.data_size / uav_rate

                for l in range(len(idx)):
                    jammer_idx = self.jammer_index_list[idx[l][0]]
                    if idx[l][0] % 2 == 1:   # 前半段
                        uav_interference_from_jammer1 += 10 ** ((self.jammer_power - self.Jammerchannels_with_fastfading[jammer_idx, receiver_idx, self.uav_channels[i][j]] +
                                                                  self.jammerAntGain + self.uavAntGain - self.uavNoiseFigure) / 10)
                uav_rate = np.log2(1 + np.divide(uav_signal, (uav_interference + uav_interference_from_jammer1 + self.sig2)))
                uav_rate *= self.bandwidth
                transmit_time2 = self.data_size / uav_rate

                if transmit_time2 > self.jammer_time[1]:
                    transmit_time1 = (self.data_size - uav_rate * self.jammer_time[1]) / (self.data_size / transmit_time1)
                    transmit_time = transmit_time1 + transmit_time2
                else:
                    transmit_time = transmit_time2

        else:
            uav_rate = np.log2(1 + np.divide(uav_signal, (uav_interference + self.sig2)))
            uav_rate *= self.bandwidth
            transmit_time = self.data_size / uav_rate
        if transmit_time < self.t_Rx:
            return transmit_time
        else:
            return self.t_Rx

    def get_reward(self):
        uav_rewards = np.zeros([self.n_ch], dtype=float)

        if self.jammer_channels_list == []:
            for i in range(self.n_jammer):
                self.jammer_channels_list.append(self.jammer_channels[i])
                self.jammer_index_list.append(i)
            self.jammer_time[0] = self.t_Rx
        # print(self.jammer_channels_list)

        tra = 0
        rec = 0
        while tra < self.n_ch:
            other_channel_list = []
            other_index_list = []
            for i in range(self.n_ch):
                for j in range(self.n_des):
                    if i==tra and j==rec:
                        continue
                    other_channel_list.append(self.uav_channels[i][j])      #排除自己通信信道的其他信道
                    other_index_list.append(self.uav_pairs[i][j][0])        #排除自己簇头的其他簇头（但是还是有自己簇头的呀）

            tra_time = self.compute_reward(tra, rec, other_channel_list, other_index_list) # 传输时间

            energy = 10 ** (self.uav_powers[tra][rec] / 10 - 3) * tra_time      # 能量奖励
            self.rew_energy += energy
            jump = self.uav_jump_count[tra]  # 跳频开销
            self.rew_jump += jump
            # uav_rewards[tra] += (0.5 * energy - 0.5 * jump)
            uav_rewards[tra] += - (0.8 * energy + 0.2 * jump)

            rec += 1
            if rec == 2:
                tra += 1
                rec = 0

        self.jammer_channels_list = []
        self.jammer_index_list = []
        self.jammer_time = np.zeros([2])
        return uav_rewards

    def reward_details(self):

        return self.rew_energy / self.n_ch, self.rew_jump / self.n_ch

    def clear_reward(self):
        self.rew_energy = 0
        self.rew_jump = 0

    def renew_jammer_channels_after_Rx(self):
        self.t_uav += self.t_Rx
        self.t_jammer += self.t_Rx
        # self.jammer_channels_list = []
        if np.floor_divide((self.t_jammer - self.t_Rx), self.t_dwell) == np.floor_divide(self.t_jammer, self.t_dwell) - 1: #这一步是要判断什么
        # （干扰机时间-传输时间0.98）/干扰机扫频停留时间2.28 == 干扰机时间/干扰机扫频停留时间 - 1
            if self.type_of_interference == "saopin":
                for i in range(self.n_jammer):
                    self.jammer_channels[i] += self.step_forward
                    self.jammer_channels[i] = int(self.jammer_channels[i] % self.n_channel)

                if self.t_jammer % self.t_dwell == 0:
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(
                            (self.jammer_channels[i] + self.n_channel - 1) % self.n_channel)
                        self.jammer_index_list.append(i)
                    self.jammer_time[0] = self.t_Rx

                else:  # 正好在Rx中间切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(self.jammer_channels[i])  # 后半段
                        self.jammer_index_list.append(i)
                        self.jammer_channels_list.append(
                            (self.jammer_channels[i] + self.n_channel - 1) % self.n_channel)  # jammer_channels[i]-1
                        self.jammer_index_list.append(i)
                    change_times = np.floor_divide(self.t_jammer, self.t_dwell)
                    change_point = change_times * self.t_dwell

                    self.jammer_time[0] = self.t_jammer - change_point  # 0对应传输后半段的干扰时间
                    self.jammer_time[1] = self.t_Rx - self.jammer_time[0]

            elif self.type_of_interference == "markov":
                old_jammer_channels = self.jammer_channels
                idx = self.all_jammer_states_list.index(self.jammer_channels)
                p = self.p_trans[idx]
                self.jammer_channels = random.choices(self.all_jammer_states_list, weights=p, k=1)[0]

                if self.t_jammer % self.t_dwell == 0:  # 传输完成后切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(old_jammer_channels[i])
                        self.jammer_index_list.append(i)
                    self.jammer_time[0] = self.t_Rx

                else:  # 传输中切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(self.jammer_channels[i])  # 后半段
                        self.jammer_index_list.append(i)
                        self.jammer_channels_list.append(old_jammer_channels[i])  # jammer_channels[i]-1
                        self.jammer_index_list.append(i)
                    change_times = np.floor_divide(self.t_jammer, self.t_dwell)
                    change_point = change_times * self.t_dwell

                    self.jammer_time[0] = self.t_jammer - change_point  # 0对应传输后半段的干扰时间
                    self.jammer_time[1] = self.t_Rx - self.jammer_time[0]

            # print("change_channels", self.jammer_channels)

    def renew_jammer_channels_after_collect(self):
        self.t_uav += self.t_collect
        self.t_jammer += self.t_collect
        if np.floor_divide((self.t_jammer - self.t_collect), self.t_dwell) == np.floor_divide(self.t_jammer,
                                                                                             self.t_dwell) - 1:
            if self.type_of_interference == "saopin":
                for i in range(self.n_jammer):
                    self.jammer_channels[i] += self.step_forward
                    self.jammer_channels[i] = int(self.jammer_channels[i] % self.n_channel)

                    self.jammer_channels_list.append(self.jammer_channels[i])
                    self.jammer_index_list.append(i)
                self.jammer_time[0] = self.t_Rx

            elif self.type_of_interference == "markov":
                idx = self.all_jammer_states_list.index(self.jammer_channels)
                p = self.p_trans[idx]
                self.jammer_channels = random.choices(self.all_jammer_states_list, weights=p, k=1)[0]

                # if self.t_jammer % self.t_dwell == 0:  传输开始前切换干扰信道
                for i in range(self.n_jammer):
                    self.jammer_channels_list.append(self.jammer_channels[i])
                    self.jammer_index_list.append(i)
                self.jammer_time[0] = self.t_Rx

            print("change_channels", self.jammer_channels)

        # 每次学习之后干扰机换一个信道
    def renew_jammer_channels_after_learn(self):
            self.t_uav += self.timestep
            self.t_jammer += self.timestep
            if np.floor_divide((self.t_jammer - self.timestep), self.t_dwell) == np.floor_divide(self.t_jammer,
                                                                                                 self.t_dwell) - 1:  # 这里是什么意思
                if self.type_of_interference == "saopin":
                    for i in range(self.n_jammer):
                        self.jammer_channels[i] += self.step_forward
                        self.jammer_channels[i] = int(self.jammer_channels[i] % self.n_channel)

                        self.jammer_channels_list.append(self.jammer_channels[i])
                        self.jammer_index_list.append(i)
                    self.jammer_time[0] = self.t_Rx

                elif self.type_of_interference == "markov":
                    idx = self.all_jammer_states_list.index(self.jammer_channels)
                    p = self.p_trans[idx]
                    self.jammer_channels = random.choices(self.all_jammer_states_list, weights=p, k=1)[0]

                    # if self.t_jammer % self.t_dwell == 0:  传输开始前切换干扰信道
                    for i in range(self.n_jammer):
                        self.jammer_channels_list.append(self.jammer_channels[i])
                        self.jammer_index_list.append(i)
                    self.jammer_time[0] = self.t_Rx

                print("change_channels", self.jammer_channels)

        # 更新簇头的位置，在无人机获知网络状态信息阶段，簇头无人机根据方向，delta距离来更新其位置【对这里的方位有点不太明白】
    def renew_positions_of_chs(self):
            # ========================================================
            # This function update the position of each ch
            # ===========================================================
            self.xyz_delta_dis = [[0, 0, 0] for _ in range(self.n_ch)]  # 拷贝成[[0,0],[0,0],[0,0],[0,0]]
            for ch in range(self.n_ch):
                i = self.ch_list[ch]
                delta_distance = self.uavs[i].velocity * self.timestep
                d = self.uavs[i].direction
                p = self.uavs[i].p

                x_delta_distance = delta_distance * math.cos(d) * math.cos(p)
                y_delta_distance = delta_distance * math.sin(d) * math.cos(p)
                z_delta_distance = delta_distance * math.sin(p)

                xpos = self.uavs[i].position[0] + x_delta_distance
                ypos = self.uavs[i].position[1] + y_delta_distance
                zpos = self.uavs[i].position[2] + z_delta_distance

                if (xpos < 0):
                    self.uavs[i].direction = math.pi - self.uavs[i].direction
                    xpos = abs(x_delta_distance) - self.uavs[i].position[0]

                if (xpos > self.length):
                    self.uavs[i].direction = math.pi - self.uavs[i].direction
                    xpos = 2 * self.length - abs(x_delta_distance) - self.uavs[i].position[0]

                if (ypos < 0):
                    self.uavs[i].direction = 2 * math.pi - self.uavs[i].direction
                    ypos = abs(y_delta_distance) - self.uavs[i].position[1]

                if (ypos > self.width):
                    self.uavs[i].direction = 2 * math.pi - self.uavs[i].direction
                    ypos = 2 * self.width - abs(y_delta_distance) - self.uavs[i].position[1]

                if (zpos < self.low_height):
                    self.uavs[i].p = 2 * math.pi - self.uavs[i].p
                    zpos = 2 * self.low_height - self.uavs[i].position[2] + abs(z_delta_distance)

                if (zpos > self.high_height):
                    self.uavs[i].p = 2 * math.pi - self.uavs[i].p
                    zpos = 2 * self.high_height - self.uavs[i].position[2] - abs(z_delta_distance)

                self.xyz_delta_dis[ch] = [x_delta_distance, y_delta_distance, z_delta_distance]
                self.uavs[i].position = [xpos, ypos, zpos]
                self.rps[i].position = self.uavs[i].position

                self.uavs[i].velocity = self.moving_smooth_factor * self.uavs[i].velocity + (1 - self.moving_smooth_factor) * np.average(
                    self.uavs[i].uav_velocity) + (
                                                1 - self.moving_smooth_factor ** 2) ** 0.5 * np.random.normal(0, self.sigma)
                self.uavs[i].direction = self.moving_smooth_factor * self.uavs[i].direction + (1 - self.moving_smooth_factor) * np.average(
                    self.uavs[i].uav_direction) + (
                                                 1 - self.moving_smooth_factor ** 2) ** 0.5 * np.random.normal(0, self.sigma)
                self.uavs[i].p = self.moving_smooth_factor * self.uavs[i].p + (1 - self.moving_smooth_factor) * np.average(self.uavs[i].uav_p) + (
                            1 - self.moving_smooth_factor ** 2) ** 0.5 * np.random.normal(0, self.sigma)

                self.uavs[i].uav_velocity.append(self.uavs[i].velocity)
                self.uavs[i].uav_direction.append(self.uavs[i].direction)
                self.uavs[i].uav_p.append(self.uavs[i].p)

    def renew_positions_of_cms(self):
        for i in range(self.n_ch):
            ch_id = self.ch_list[i]
            cm_id = self.uavs[ch_id].connections
            ch_pos = [self.uavs[ch_id].position[0], self.uavs[ch_id].position[1], self.uavs[ch_id].position[2]]
            # 簇头位置没变化时，即最开始的时候
            if self.xyz_delta_dis[i] == [0, 0, 0]:
                for j in cm_id:
                    # 更新参考点的位置
                    R1 = random.uniform(0, self.max_distance1)
                    d1 = random.uniform(0, 2 * math.pi)
                    p1 = random.uniform(0, 2 * math.pi)

                    rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                    rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                    rp_zpos = ch_pos[2] + R1 * math.sin(p1)

                    while ((rp_xpos < 0) or (rp_xpos > self.length) or (rp_ypos < 0) \
                            or (rp_ypos > self.width) or (rp_zpos < self.low_height) or (rp_zpos > self.high_height)):
                        R1 = random.uniform(0, R1)
                        d1 = random.uniform(0, 2 * math.pi)
                        p1 = random.uniform(0, 2 * math.pi)

                        rp_xpos = ch_pos[0] + R1 * math.cos(d1) * math.cos(p1)
                        rp_ypos = ch_pos[1] + R1 * math.sin(d1) * math.cos(p1)
                        rp_zpos = ch_pos[2] + R1 * math.sin(p1)
                    rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                    self.rps[j].position = rp_pos


                    # 更新簇内节点的位置
                    R2 = random.uniform(0, self.max_distance2)
                    d2 = random.uniform(0, 2 * math.pi)
                    p2 = random.uniform(0, 2 * math.pi)

                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                    self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]

                    while ((self.uavs[j].position[0] < 0) or (self.uavs[j].position[0] > self.length) or (self.uavs[j].position[1] < 0) \
                            or (self.uavs[j].position[1] > self.width) or (self.uavs[j].position[2] < self.low_height) or (self.uavs[j].position[2] > self.high_height)):
                        R2 = random.uniform(0, self.max_distance2)
                        d2 = random.uniform(0, 2 * math.pi)
                        p2 = random.uniform(0, 2 * math.pi)

                        cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                        cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                        cm_zpos = rp_pos[2] + R2 * math.sin(p2)
                        self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]
            else:
                for j in cm_id:
                    # 更新参考点的位置
                    rp_xpos = self.rps[j].position[0] + self.xyz_delta_dis[i][0]
                    rp_ypos = self.rps[j].position[1] + self.xyz_delta_dis[i][1]
                    rp_zpos = self.rps[j].position[2] + self.xyz_delta_dis[i][2]

                    if (rp_xpos < 0):
                        rp_xpos = abs(self.xyz_delta_dis[i][0]) - self.rps[j].position[0]

                    if (rp_xpos > self.length):
                        rp_xpos = 2 * self.length - abs(self.xyz_delta_dis[i][0]) - self.rps[j].position[0]

                    if (rp_ypos < 0):
                        rp_ypos = abs(self.xyz_delta_dis[i][1]) - self.rps[j].position[1]

                    if (rp_ypos > self.width):
                        rp_ypos = 2 * self.width - self.rps[j].position[1] - abs(self.xyz_delta_dis[i][1])

                    if (rp_zpos < self.low_height):
                        rp_zpos = 2 * self.low_height - self.rps[j].position[2] + abs(self.xyz_delta_dis[i][2])

                    if (rp_zpos > self.high_height):
                       rp_zpos = 2 * self.high_height - self.rps[j].position[2] - abs(self.xyz_delta_dis[i][2])

                    rp_pos = [rp_xpos, rp_ypos, rp_zpos]
                    self.rps[j].position = rp_pos

                    # 更新簇内节点的位置
                    R2 = random.uniform(0, self.max_distance2)
                    d2 = random.uniform(0, 2 * math.pi)
                    p2 = random.uniform(0, 2 * math.pi)

                    cm_xpos = rp_pos[0] + R2 * math.cos(d2) * math.cos(p2)
                    cm_ypos = rp_pos[1] + R2 * math.sin(d2) * math.cos(p2)
                    cm_zpos = rp_pos[2] + R2 * math.sin(p2)

                    while ((cm_xpos < 0) or (cm_xpos > self.length) or (cm_ypos < 0) or (cm_ypos > self.width) or (cm_zpos < self.low_height) or (cm_zpos > self.high_height)):
                        R2 = random.uniform(0, self.max_distance2)
                        d2 = random.uniform(0, 2 * math.pi)
                        p2 = random.uniform(0, 2 * math.pi)

                        cm_xpos = self.rps[j].position[0] + R2 * math.cos(d2) * math.cos(p2)
                        cm_ypos = self.rps[j].position[1] + R2 * math.sin(d2) * math.cos(p2)
                        cm_zpos = self.rps[j].position[2] + R2 * math.sin(p2)
                    self.uavs[j].position = [cm_xpos, cm_ypos, cm_zpos]

    # 干扰机在收集数据时间里更新其位置
    def renew_positions_of_jammers(self):
            # ========================================================
            # This function update the position of each vehicle
            # ===========================================================

            i = 0
            # for i in range(len(self.position)):
            while (i < len(self.jammers)):
                # print ('start iteration ', i)
                # print(self.position, len(self.position), self.direction)
                delta_distance = self.jammers[i].velocity * self.t_collect
                d = self.jammers[i].direction
                p = self.jammers[i].p

                x_delta_distance = delta_distance * math.cos(d) * math.cos(p)
                y_delta_distance = delta_distance * math.sin(d) * math.cos(p)
                z_delta_distance = delta_distance * math.sin(p)

                xpos = self.jammers[i].position[0] + x_delta_distance
                ypos = self.jammers[i].position[1] + y_delta_distance
                zpos = self.jammers[i].position[2] + z_delta_distance

                if (self.jammers[i].position[0] + x_delta_distance < 0):
                    self.jammers[i].direction = math.pi - self.jammers[i].direction
                    xpos = abs(x_delta_distance) - self.jammers[i].position[0]

                if (self.jammers[i].position[0] + x_delta_distance > self.length):
                    self.jammers[i].direction = math.pi - self.jammers[i].direction
                    xpos = 2 * self.length - abs(x_delta_distance) - self.jammers[i].position[0]

                if (self.jammers[i].position[1] + y_delta_distance < 0):
                    self.jammers[i].direction = 2 * math.pi - self.jammers[i].direction
                    ypos = abs(y_delta_distance) - self.jammers[i].position[1]

                if (self.jammers[i].position[1] + y_delta_distance > self.width):
                    self.jammers[i].direction = 2 * math.pi - self.jammers[i].direction
                    ypos = 2 * self.width - abs(y_delta_distance) - self.jammers[i].position[1]

                if (self.jammers[i].position[2] + z_delta_distance < self.low_height):
                    self.jammers[i].p = 2 * math.pi - self.jammers[i].p
                    zpos = 2 * self.low_height - self.jammers[i].position[2] + abs(z_delta_distance)

                if (self.jammers[i].position[2] + z_delta_distance > self.high_height):
                    self.jammers[i].p = 2 * math.pi - self.jammers[i].p
                    zpos = 2 * self.high_height - (self.jammers[i].position[2] + abs(z_delta_distance))

                self.jammers[i].position = [xpos, ypos, zpos]

                self.jammers[i].velocity = self.moving_smooth_factor * self.jammers[i].velocity + (1 - self.moving_smooth_factor) * np.average(
                    self.jammers[i].jammer_velocity) + (1 - self.moving_smooth_factor) ** 0.5 * np.random.normal(0, self.sigma)
                self.jammers[i].direction = self.moving_smooth_factor * self.jammers[i].direction + (1 - self.moving_smooth_factor) * np.average(
                    self.jammers[i].jammer_direction) + (1 - self.moving_smooth_factor) ** 0.5 * np.random.normal(0, self.sigma)
                self.jammers[i].p = self.moving_smooth_factor * self.jammers[i].p + (1 - self.moving_smooth_factor) * np.average(self.jammers[i].jammer_p) + (
                            1 - self.moving_smooth_factor) ** 0.5 * np.random.normal(0, self.sigma)

                self.jammers[i].jammer_velocity.append(self.jammers[i].velocity)
                self.jammers[i].jammer_direction.append(self.jammers[i].direction)
                self.jammers[i].jammer_p.append(self.jammers[i].p)
                i += 1

    def renew_channels(self):
        # =======================================================================
        # This function updates all the channels including V2V and V2I channels
        # =========================================================================
        uavs_ = [u.position for u in self.uavs]
        uav_positions = uavs_
        jammer_positions = [j.position for j in self.jammers]
        self.Jammerchannels.update_positions(jammer_positions, uav_positions)
        self.UAVchannels.update_positions(uav_positions)
        self.Jammerchannels.update_pathloss()
        self.UAVchannels.update_pathloss()
        self.Jammerchannels.update_fast_fading()
        self.UAVchannels.update_fast_fading()

        UAVchannels_with_fastfading = np.repeat(self.UAVchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        self.UAVchannels_with_fastfading = UAVchannels_with_fastfading - self.UAVchannels.FastFading
        Jammerchannels_with_fastfading = np.repeat(self.Jammerchannels.PathLoss[:, :, np.newaxis], self.n_channel, axis=2)
        self.Jammerchannels_with_fastfading = Jammerchannels_with_fastfading - self.Jammerchannels.FastFading

    def act(self):
        self.renew_jammer_channels_after_Rx()
        reward = self.get_reward()
        self.renew_positions_of_chs()
        self.renew_positions_of_cms()
        if self.is_jammer_moving:
            self.renew_positions_of_jammers()
        self.renew_channels()
        return reward

        # 分解动作值的操作，由簇头到每个选择的通信信道数
    def decomposition_action(self, action):
            for i in range(self.n_ch):
                for j in range(self.n_des):
                    a = action[i] % self.action_range
                    channel_last = self.uav_channels[i][j]
                    self.uav_channels[i][j] = int(a / len(self.uav_power_list))
                    self.uav_powers[i][j] = self.uav_power_list[a % len(self.uav_power_list)]
                    if (self.uav_channels[i][j] != channel_last):
                        self.uav_jump_count[i] += 1
                    action[i] = int(action[i] / self.action_range)

    def get_action(self, action):
            if self.policy == "Q_learning":
                action_all = np.zeros([self.n_ch, self.n_ch], dtype=np.int32)
                # action_cur = np.zeros([self.n_uav], dtype=int)
                for i in range(self.n_ch):
                    for j in range(self.n_ch - 1, -1, -1):
                        action_all[i, j] = int(np.floor(action[i] / (self.action_range ** j)))
                        action[i] -= action_all[i, j] * (self.action_range ** j)
                action_real = action_all.diagonal()
                print(action_real)
                return action_real
            elif self.policy == "Sensing_Based_Method":
                return action
            else:
                action_all = np.zeros([self.n_ch, self.n_ch], dtype=np.int32)
                for i in range(self.n_ch):
                    for j in range(self.n_ch - 1, -1, -1):
                        action_all[i, j] = int(np.floor(action[i] / (self.action_range ** j)))
                        action[i] -= action_all[i, j] * (self.action_range ** j)

                action_real = action_all.diagonal()
                print(action_real)
                return action_real

    def reset(self):
        self.new_random_game()
        state = self.get_state()
        return state

    def step(self, a):
        action = np.array(deepcopy(a),dtype=np.int32)
        # action = deepcopy(a).astype(np.int32)
        # print("a", a)
        self.decomposition_action(action)
        reward = self.act()
        state_next = self.get_state()  # 得到新的状态

        done = False  # 不会中途停止，一直走完整个回合

        return state_next, reward, done, {}

    def smooth(self, data, sm=1):
        smooth_data = []
        # if sm > 1:
        #     for d in data:
        z = np.ones(len(data))
        y = np.ones(sm) * 1.0
        d = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        smooth_data.append(d)
        return smooth_data

    def plot(self, cost_list):
        y_data = self.smooth(cost_list, 19)
        x_data = np.arange(len(cost_list))
        # sns.set(style="darkgrid", font_scale=1.5)
        # sns.tsplot(time=x_data, data=y_data, color='b', linestyle='-')
        np.savetxt('DRQN_po.txt', y_data[0], fmt='%f')


        plt.plot(x_data, y_data[0])
        plt.ylabel('DRQN_po_reward')
        plt.xlabel('training Episode')
        # plt.ylim(0.5, 1.0)
        plt.show()

        plt.plot(x_data, cost_list)
        plt.ylabel('reward')
        plt.xlabel('training Episode')
        # plt.ylim(0.5, 1.0)
        plt.show()