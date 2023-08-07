from env.uav.uav import systemEnv
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt

def get_random_data():
    r = []
    trange = tqdm.tqdm(range(100000))
    for i in trange:
        action = env.generate_random_actions()
        obs, reward, done, _,  info = env.step(action)
        r.append(reward)
    return np.array(r).reshape(-1)

if __name__ == '__main__':
    env = systemEnv()
    env.reset()
    r = get_random_data()
    # BEGIN: 5f6c7d5d8d5c
    window_size = 100
    plt.plot(pd.Series(r).rolling(window_size).mean())
    # END: 5f6c7d5d8d5c
    plt.show()
    print(np.mean(r), np.std(r), min(r), max(r))
# 119.25917579504261 80.68676041531067            (500000)
# 108.70945564152147 80.81674735557192            (100000)