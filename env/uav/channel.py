import numpy as np
from env.uav.moving import UAVMoving, JammerMoving

def generate_complex_gaussian(size): # 生成复数高斯噪声
    real_part = np.random.normal(0, 1, size)
    imag_part = np.random.normal(0, 1, size)
    return real_part + 1j*imag_part

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1-p2)))

def calc_pathloss_params(fc, hb, hm, area_type="small_and_medium_size_cities"):
    """
    使用Okumura-Hata模型 https://www.wiley.com/legacy/wileychi/molisch/supp2/appendices/c07_Appendices.pdf
    默认参数计算出的数据 : A:110.2, B:33.8, C:0
    """
    if area_type == "small_and_medium_size_cities":
        C = 0
        ahm = (1.1*np.log10(fc)-0.7)*hm - (1.56*np.log10(fc)-0.8)
    else:
        if fc <= 200:
            ahm = 8.29*(np.log10(1.54*hm))**2 - 1.1
        else:
            ahm = 3.2*(np.log10(11.75*hm))**2 - 4.97
        if area_type == "metropolitan_areas":
            C = 0
        if area_type == "suburban_environments":
            C = -2*(np.log10(fc/28))**2 - 5.4
        if area_type == "rural_area":
            C = -4.78*(np.log10(fc))**2 + 18.33*np.log10(fc) - 40.98
    A = 69.55 + 26.16 * np.log10(fc) - 13.82 * np.log10(hb) - ahm
    B = 44.9 - 6.55 * np.log10(hb)
    return A, B, C

class ClusterChannel(UAVMoving):
    """
    一个簇群的信道类
    """
    def __init__(
        self, n_channels=6, n_slaves=3, area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, power_list=[36, 33, 30, 27],**kwargs
    ):
        super().__init__(**kwargs)
        # 通信参数
        self.n_channels = n_channels
        self.n_slaves = n_slaves
        self.area_type = area_type
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.pathloss = np.zeros(shape=(n_slaves,))
        self.FastFading = np.zeros(shape=(n_slaves, n_channels))
        # action
        self.power_list = power_list
        self.channel_power = np.zeros(shape=(n_slaves,), dtype=np.int32)
        self.channel_select = np.zeros(shape=(n_slaves,), dtype=np.int32)
        if n_channels < n_slaves:
            raise ValueError("The number of channels should be greater than the number of slaves.")
        self.calc_pathloss()
        self.calc_fast_fading()
        self.A, self.B, self.C = calc_pathloss_params(self.fc, self.hb, self.hm, self.area_type)
    
    def cluster_pathloss_interference(self, k):
        """
        计算簇群内部的干扰, 第k个slave受到的干扰
        """
        ans = 0
        for i in range(self.n_slaves):
            if k == i or self.channel_select[k] != self.channel_select[i]:
                continue
            d = distance(self.position[k+1], self.position[i+1]) + distance(self.position[i+1], self.position[0]) + 1e-3
            ans += max(self.A + self.B * np.log10(d) + self.C, 0)
        return ans
    
    def calc_pathloss(self):
        """
        计算master和slaves的pathloss
        ! 注意实际上使用Okumura-Hata模型有一点不合适
        """
        for i in range(len(self.n_slaves)):
            d = distance(self.position[i+1], self.position[0]) + 1e-3
            self.pathloss[i] = max(self.A + self.B * np.log10(d) + self.C, 0) # 防止出现负数
    
    def calc_fast_fading(self):
        h = generate_complex_gaussian(size=(self.n_slaves, self.n_channels)) / np.sqrt(2) # 每一个slave对于所有的信道都先算出来
        self.FastFading = 20 * np.log10(np.abs(h))
    
    def act(self, channel=None, channel_power=None, init=False):
        if not init:
            self.channel_power = channel_power
            cnt = 0
            for i in range(self.n_slaves):
                if self.channel_select[i] != channel[i]:
                    self.channel_select[i] = channel[i]
                    cnt += 1
            return cnt
        else:
            self.channel_power = [np.random.choice(self.power_list) for _ in range(self.n_slaves)]
            self.channel_select = np.random.choice(self.n_channels, size=(self.n_slaves,), replace=False)
    
    def observe(self):
        return (self.channel_select, self.channel_power, self.position)

class Channel(object):
    """
    所有簇群的信道类
    """
    def __init__(
        self, n_clusters=3, n_channels=6, n_slaves=3, area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, power_list=[36, 33, 30, 27], **kwargs
    ):
        self.n_clusters = n_clusters
        self.n_channels = n_channels 
        self.n_slaves = n_slaves
        self.area_type = area_type
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.power_list = power_list

        self.channel_select = np.zeros(shape=(n_clusters, n_clusters), dtype=np.int32)
        self.channel_power = np.zeros(shape=(n_clusters,), dtype=np.int32)
        self.pathloss = np.zeros(shape=(n_clusters, n_clusters))
        self.FastFading = np.zeros(shape=(n_clusters, n_clusters, n_channels))
        self.Clusters = [ClusterChannel(n_channels, n_slaves, area_type, fc, hb, hm) for _ in range(n_clusters)]
        
        self.calc_pathloss()
        self.calc_fast_fading()
        self.A, self.B, self.C = calc_pathloss_params(self.fc, self.hb, self.hm, self.area_type)

    def calc_pathloss(self):
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                d = distance(self.Clusters[i].position[0], self.Clusters[j].position[0]) + 1e-3
                self.pathloss[i][j] = max(self.A + self.B * np.log10(d) + self.C, 0)

    def calc_fast_fading(self):
        h = generate_complex_gaussian(size=(self.n_clusters, self.n_clusters, self.n_channels)) / np.sqrt(2)
        self.FastFading = 20 * np.log10(np.abs(h))
    
    def act(self, actions=None, init=False):
        #!实际上和act那里定义的不一样, 传入的actions还要再看看
        master_action = [actions[i][:2*self.n_clusters] for i in range(self.n_clusters)]
        slaves_action = [actions[i][2*self.n_clusters:] for i in range(self.n_clusters)] #! 看看对不对
        if init:
            for i in range(self.n_clusters):
                self.Clusters[i].act(init=True)
            for i in range(self.n_clusters):
                for j in range(self.n_clusters):
                    self.channel_select[i][j] = np.random.randint(self.n_channels)
                    self.channel_power[i][j] = np.random.choice(self.power_list)
        else:
            cnt = 0
            for i in range(self.n_clusters):
                channel, power = slaves_action[i][:self.n_slaves], slaves_action[i][self.n_slaves:]
                cnt += self.Clusters[i].act(channel, power)
            for i in range(self.n_clusters):
                channel, power = master_action[i][:self.n_slaves], master_action[i][self.n_slaves:]
                for j in range(self.n_clusters):
                    self.channel_power[i][j] = power[j] #注意 i -> j 和 j -> i 是不一样的, 双向通信
                    if self.channel_select[i][j] != channel[j]:
                        self.channel_select[i][j] = channel[j]
                        if i != j:
                            cnt += 1
            return cnt

    @property
    def master_position(self):
        return np.array([self.Clusters[i].position[0] for i in range(self.n_clusters)])

    def observe(self):
        _channel, _power, _position = [], [], []
        for i in range(self.n_clusters):
            channel, channel_power, position = self.Clusters[i].observe()
            _channel.append(np.concatenate((channel, self.channel_select[i]), axis=0))
            _power.append(np.concatenate((channel_power, self.channel_power[i]), axis=0))
            _position.append(np.concatenate((position[1:], self.master_position), axis=0))
        return (_channel, _power, _position)
        

class JammerChannel(JammerMoving):
    """
    干扰机的信道类
    """
    def __init__(
        self, n_jammers=3, n_channels=6, jamming_mode='Markov', area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, jammer_power=30, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_jammers = n_jammers
        self.n_channels = n_channels
        self.area_type = area_type
        self.jamming_mode = jamming_mode
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.jammer_power = jammer_power #TODO 注意这里的jammer的功率都是固定的
        self.channels_jammed = np.zeros(shape=(n_channels,), dtype=np.int32) # 被干扰了就是1，没有被干扰就是0
        self.jamming_channel = np.zeros(shape=(n_jammers,)) # 每个干扰机的干扰信道
        self._init_jamming()

    def _init_jamming(self):
        if self.jamming_mode == 'Markov':
            self.transition_matrix = np.random.random(size=(self.n_jammers, self.n_channels))
            self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)
            for i in range(self.n_jammers):
                self.jamming_channel[i] = np.random.choice(self.n_channels, p=self.transition_matrix[i])
                self.channels_jammed[self.jamming_channel[i]] += 1
        elif self.jamming_mode == 'Sweeping':
            for i in range(self.n_jammers):
                self.jamming_channel[i] = i
                self.channels_jammed[i] += 1
        elif self.jamming_mode == 'Random':
            for i in range(self.n_jammers):
                self.jamming_channel[i] = np.random.randint(self.n_channels)
                self.channels_jammed[self.jamming_channel[i]] += 1
        else:
            raise NotImplementedError("The jamming mode is not implemented.")

    def step(self):
        super().step() # 更新位置
        def next_channel(i):
            if self.jamming_mode == 'Markov':
                return np.random.choice(self.n_channels, p=self.transition_matrix[i])
            elif self.jamming_mode == 'Sweeping':
                return (self.jamming_channel[i] + 1) % self.n_channels
            elif self.jamming_mode == 'Random':
                return np.random.randint(self.n_channels)
        
        for i in range(self.n_jammers):
            self.channels_jammed[self.jamming_channel[i]] -= 1
            self.jamming_channel[i] = next_channel(i)
            self.channels_jammed[self.jamming_channel[i]] += 1