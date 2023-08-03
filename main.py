from env.uav.uav import systemEnv
import pandas as pd
import numpy as np


if __name__ == '__main__':
    env = systemEnv()
    (channel, power, position, SNR), info = env.reset()
    actions = []
    for i in range(env.n_clusters):
        channel = np.random.randint(0, env.n_channels, (env.n_clusters,))
        power = np.random.randint(0, len(env.power_list), (env.n_clusters,))
        master = np.concatenate((channel, power), axis=0)
        
        channel = np.random.randint(0, env.n_channels, (env.n_slaves,))
        power = np.random.randint(0, len(env.power_list), (env.n_slaves,))
        slaves = np.concatenate((channel, power), axis=0)
        actions.append(np.concatenate((master, slaves), axis=0))
    actions = np.array(actions)

    observation, reward, truncated, done, info = env.step(actions)
    print(observation.shape)
    print(reward)
    print(truncated)
    print(done)
    print(info)