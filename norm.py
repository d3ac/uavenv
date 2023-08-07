import numpy as np

class container():
    def __init__(self, max_len):
        self.data = []
        self.max_len = max_len
        self.cnt = 0
        self.sum = 0
        self.sum_square = 0
        self.idx = 0
        self.flag = False

    def _append(self, x):
        if self.flag:
            self.sum -= self.data[self.idx]
            self.sum_square -= self.data[self.idx] ** 2
            self.sum += x
            self.sum_square += x ** 2
            self.data[self.idx] = x
            self.idx = (self.idx + 1) % self.max_len
        else:
            self.data.append(x)
            self.sum += x
            self.sum_square += x ** 2
            self.cnt += 1
            if self.cnt == self.max_len:
                self.flag = True
    
    def append(self, x):
        x.reshape(-1)
        for i in x:
            self._append(i)

    def mean(self):
        return self.sum / self.cnt
    
    def std(self):
        mean = self.mean()
        return np.sqrt(self.sum_square / self.cnt - mean ** 2)



class obs_Normalizer(object):
    def __init__(self, training=True, t=None):
        self.training = training
        self.max_storage = 20000
        if training:
            self.obs_channel = container(self.max_storage)
            self.obs_power = container(self.max_storage)
            self.obs_position = container(self.max_storage)
            self.obs_SNR = container(self.max_storage)
        else:
            self.mean_channel = t.mean_channel
            self.std_channel = t.std_channel
            self.mean_power = t.mean_power
            self.std_power = t.std_power
            self.mean_position = t.mean_position
            self.std_position = t.std_position
            self.mean_SNR = t.mean_SNR
            self.std_SNR = t.std_SNR
        self.cnt = 0

    def update(self):
        self.mean_channel = self.obs_channel.mean()
        self.std_channel = self.obs_channel.std()
        self.mean_power = self.obs_power.mean()
        self.std_power = self.obs_power.std()
        self.mean_position = self.obs_position.mean()
        self.std_position = self.obs_position.std()
        self.mean_SNR = self.obs_SNR.mean()
        self.std_SNR = self.obs_SNR.std()

    def merge_obs(self, channel, power, position, SNR):
        position = np.array(position)
        if self.training:
            self.obs_channel.append(channel)
            self.obs_power.append(power)
            self.obs_position.append(position)
            self.obs_SNR.append(SNR)
            self.update()
        channel = (channel - self.mean_channel) / self.std_channel
        power = (power - self.mean_power) / self.std_power
        position = (position - self.mean_position) / self.std_position
        SNR = (SNR - self.mean_SNR) / self.std_SNR
        return np.concatenate((channel, power, position, SNR), axis=1)


class value_Normalizer(object):
    def __init__(self, training=True, t=None):
        self.training = training
        self.max_storage = 20000
        if training:
            self.obs_value = container(self.max_storage)
        else:
            self.mean_value = t.mean_value
            self.std_value = t.std_value
        self.cnt = 0

    def update(self):
        self.mean_value = self.obs_value.mean()
        self.std_value = self.obs_value.std()

    def merge_obs(self, value):
        if self.training:
            self.obs_value.append(value)
            self.update()
        value = (value - self.mean_value) / self.std_value
        return value