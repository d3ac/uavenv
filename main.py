from env.uav.uav import systemEnv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    env = systemEnv()
    (channel, power, position, SNR), info = env.reset()
    SNR = np.array(SNR).reshape(-1)
    SNR = np.sort(SNR)
    print(SNR)