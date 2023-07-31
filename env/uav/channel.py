import numpy as np
from env.uav.moving import UAVMoving

def generate_complex_gaussian(size): # 生成复数高斯噪声
    real_part = np.random.normal(0, 1, size)
    imag_part = np.random.normal(0, 1, size)
    return real_part + 1j*imag_part

class ClusterChannel(UAVMoving):
    """
    一个簇群的信道类
    """
    def __init__(
        self, n_channels=6, n_slaves=3, area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_channels = n_channels
        self.n_slaves = n_slaves
        self.area_type = area_type
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.pathloss = np.zeros(shape=(n_slaves,))
        if n_channels < n_slaves:
            raise ValueError("The number of channels should be greater than the number of slaves.")
        self.calc_pathloss(self.position)
        self.calc_fast_fading()
    
    def calc_pathloss(self, positions):
        """
        计算master和slaves的pathloss, 使用Okumura-Hata模型 https://www.wiley.com/legacy/wileychi/molisch/supp2/appendices/c07_Appendices.pdf
        默认参数计算出的数据 : A:110.2, B:33.8
        ! 注意实际上使用Okumura-Hata模型有一点不合适
        """
        # calculate pathloss parameters
        if self.area_type == "small_and_medium_size_cities":
            C = 0
            ahm = (1.1*np.log10(self.fc)-0.7)*self.hm - (1.56*np.log10(self.fc)-0.8)
        else:
            if self.fc <= 200:
                ahm = 8.29*(np.log10(1.54*self.hm))**2 - 1.1
            else:
                ahm = 3.2*(np.log10(11.75*self.hm))**2 - 4.97
            if self.area_type == "metropolitan_areas":
                C = 0
            if self.area_type == "suburban_environments":
                C = -2*(np.log10(self.fc/28))**2 - 5.4
            if self.area_type == "rural_area":
                C = -4.78*(np.log10(self.fc))**2 + 18.33*np.log10(self.fc) - 40.98
        A = 69.55 + 26.16 * np.log10(self.fc) - 13.82 * np.log10(self.hb) - ahm
        B = 44.9 - 6.55 * np.log10(self.hb)

        for i in range(len(self.n_slaves)):
            d = np.sqrt(np.sum(np.square(positions[i+1] - positions[0]))) + 1e-3
            self.pathloss[i] = max(A + B * np.log10(d) + C, 0) # 防止出现负数
    
    def calc_fast_fading(self):
        h = generate_complex_gaussian(size=(self.n_slaves, self.n_channels)) / np.sqrt(2) # 每一个slave对于所有的信道都先算出来
        self.FastFading = 20 * np.log10(np.abs(h))

class MasterChannel(object):
    def __init__(
        self, n_groups=3, n_channels=6, n_slaves=3, area_type="small_and_medium_size_cities", fc=800*1e6, hb=50, hm=20, **kwargs
    ):
        self.n_groups = n_groups
        self.n_channels = n_channels 
        self.n_slaves = n_slaves
        self.area_type = area_type
        self.fc = fc
        self.hb = hb
        self.hm = hm
        self.Clusters = [ClusterChannel(n_channels, n_slaves, area_type, fc, hb, hm) for _ in range(n_groups)]