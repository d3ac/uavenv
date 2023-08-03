from env.uav.uav import systemEnv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    env = systemEnv()
    env.reset()
    for i in range(10):
        actions = env.generate_random_actions()
        obs, reward, truc, done, info = env.step(actions)
        print(obs)
        break