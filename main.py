from UAVenv.uav.uav import systemEnv
import pandas as pd
import numpy as np
import tqdm

def get_random_data(env, n):
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
    env1 = systemEnv()
    env1.reset(None)
    rewards = []
    for i in range(10):
        action = env1.generate_random_actions()
        obs, reward, done, _, info = env1.step(action)
        rewards.append(reward.mean())
    print(rewards)