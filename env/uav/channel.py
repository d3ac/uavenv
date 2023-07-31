import numpy as np
from env.uav.moving import UAVMoving, JammerMoving

def generate_complex_gaussian(size): # 生成复数高斯噪声
    real_part = np.random.normal(0, 1, size)
    imag_part = np.random.normal(0, 1, size)
    return real_part + 1j*imag_part

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
        self, n_channels=6, n_slaves=3, area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, **kwargs
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
        self.power_list = np.zeros(shape=(n_slaves,), dtype=np.int32)
        self.channel_list = np.zeros(shape=(n_slaves,), dtype=np.int32)
        if n_channels < n_slaves:
            raise ValueError("The number of channels should be greater than the number of slaves.")
        self.calc_pathloss()
        self.calc_fast_fading()
    
    def calc_pathloss(self):
        """
        计算master和slaves的pathloss
        ! 注意实际上使用Okumura-Hata模型有一点不合适
        """
        A, B, C = calc_pathloss_params(self.fc, self.hb, self.hm, self.area_type)
        for i in range(len(self.n_slaves)):
            d = np.sqrt(np.sum(np.square(self.position[i+1] - self.position[0]))) + 1e-3
            self.pathloss[i] = max(A + B * np.log10(d) + C, 0) # 防止出现负数
    
    def calc_fast_fading(self):
        h = generate_complex_gaussian(size=(self.n_slaves, self.n_channels)) / np.sqrt(2) # 每一个slave对于所有的信道都先算出来
        self.FastFading = 20 * np.log10(np.abs(h))

class Channel(object):
    """
    所有簇群的信道类
    """
    def __init__(
        self, n_clusters=3, n_channels=6, n_slaves=3, area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, **kwargs
    ):
        self.n_clusters = n_clusters
        self.n_channels = n_channels 
        self.n_slaves = n_slaves
        self.area_type = area_type
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.pathloss = np.zeros(shape=(n_clusters, n_clusters))
        self.FastFading = np.zeros(shape=(n_clusters, n_clusters, n_channels))
        self.Clusters = [ClusterChannel(n_channels, n_slaves, area_type, fc, hb, hm) for _ in range(n_clusters)]
        self.calc_pathloss()
        self.calc_fast_fading()

    def calc_pathloss(self):
        A, B, C = calc_pathloss_params(self.fc, self.hb, self.hm, self.area_type)
        for i in range(self.n_clusters):
            for j in range(self.n_clusters):
                d = np.sqrt(np.sum(np.square(self.Clusters[i].position[0] - self.Clusters[j].position[0]))) + 1e-3
                self.pathloss[i][j] = max(A + B * np.log10(d) + C, 0)

    def calc_fast_fading(self):
        h = generate_complex_gaussian(size=(self.n_clusters, self.n_clusters, self.n_channels)) / np.sqrt(2)
        self.FastFading = 20 * np.log10(np.abs(h))

class JammerChannel(JammerMoving):
    """
    干扰机的信道类
    """
    def __init__(
        self, n_jammers=3, n_channels=6, jamming_mode='Markov', area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_jammers = n_jammers
        self.n_channels = n_channels
        self.area_type = area_type
        self.jamming_mode = jamming_mode
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.channels_jammed = np.zeros(shape=(n_channels,), dtype=np.int32) # 被干扰了就是1，没有被干扰就是0
        self.jamming_channel = np.zeros(shape=(n_jammers,)) # 每个干扰机的干扰信道
        self._init_jamming_channel()

    def _init_jamming_channel(self):
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