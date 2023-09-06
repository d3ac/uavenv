from UAVenv.uav.uav import systemEnv
import pandas as pd
import numpy as np
import tqdm
import matplotlib.pyplot as plt

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
    env = systemEnv()
    env.reset(None)
    # get_random_data(env, 100000)
    # position = env.channel.Clusters[0].position
    # print(position)
    # x = position[1:,0]
    # y = position[1:,1]
    # z = position[1:,2]
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x,y,z)
    # plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(env.n_clusters):
        position = env.channel.Clusters[i].position
        x = position[:,0]
        y = position[:,1]
        z = position[:,2]
        ax.scatter(x,y,z)
    plt.show()

    while 1:
        action = env.generate_random_actions()
        obs, reward, done, _,  info = env.step(action)
        if done.all():
            env.reset()
            break
        print(1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(env.n_clusters):
        position = env.channel.Clusters[i].position
        x = position[:,0]
        y = position[:,1]
        z = position[:,2]
        ax.scatter(x,y,z)
    plt.show()