from env.uav.uav import systemEnv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    env = systemEnv()
    env.reset()
    action = env.generate_random_actions()
    obs, reward, done, _,  info = env.step(action)
    print(obs.shape)
    print(env.observation_space)