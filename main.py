from UAVenv.uav.uav import systemEnv
import pandas as pd
import numpy as np
import tqdm

def get_random_data(n):
    r = []
    trange = tqdm.tqdm(range(n))
    for i in trange:
        action = env.generate_random_actions()
        obs, reward, done, _,  info = env.step(action)
        r.append(reward)
        if done.all():
            env.reset()
    r = np.array(r).reshape(-1)
    print(np.mean(r), np.std(r))

if __name__ == '__main__':
    env = systemEnv()
    env.reset()
    get_random_data(5000)