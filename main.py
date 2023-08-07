from env.uav.uav import systemEnv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    env = systemEnv()
    env.reset()
    for i in range(10000):
        action = env.generate_random_actions()
        obs, reward, done, _,  info = env.step(action)
        if done[0] == True:
            print('done', i)
            break