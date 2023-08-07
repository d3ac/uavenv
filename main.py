from env.uav.uav import systemEnv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    env = systemEnv()
    env.reset()
    r = []
    # for i in range(10):
    #     action = env.generate_random_actions()
    #     obs, reward, done, _,  info = env.step(action)
    #     r.append(reward)
    # r = np.array(r).reshape(-1)
    # print(np.mean(r), np.std(r), min(r), max(r))
    action = env.generate_random_actions()
    obs, reward, done, _,  info = env.step(action)
    print(reward)