import numpy as np
import random

def distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1-p2)))

class UAVMoving(object):
    """
    实现一个uav簇群的移动
    """
    def __init__(
        self, n_slaves=3, init_position=None, xlim=1000, ylim=1000, zlim_max=200, zlim_min=50, max_radius=50,
        master_velocity=10, slave_velocity=10, moving_factor=0.1, dt=0.1, **kwargs
    ):
        # 储存位置信息
        self.n_slaves = n_slaves
        self.position = np.zeros(shape=(n_slaves+1, 3))
        self.azimuth = np.zeros(shape=(n_slaves+1,))
        self.elevation = np.zeros(shape=(n_slaves+1,))
        self.velocity = np.zeros(shape=(n_slaves+1,))
        # 移动参数
        self.master_velocity = master_velocity # 簇头移动速度
        self.slave_velocity = slave_velocity # 从机移动速度
        self.moving_factor = moving_factor
        self.dt = dt
        # 移动限制
        self.xlim = xlim
        self.ylim = ylim
        self.zlim_max = zlim_max
        self.zlim_min = zlim_min
        self.max_radius = max_radius
        if zlim_min < max_radius:
            raise ValueError("zlim_min must be greater than max_radius")
        self.init_position = init_position
        self.reset_UavMoving()
    
    def reset_UavMoving(self):
        self._init_position(self.init_position)
        self._init_moving_params()

    def _init_position(self, init_potision=None):
        self.position[0] = init_potision if init_potision is not None else (random.uniform(0,self.xlim),random.uniform(0,self.ylim),random.uniform(self.zlim_min,self.zlim_max))
        for i in range(1, self.n_slaves+1):
            azimuth =  2 * np.pi * np.random.random()
            elevation = np.pi * np.random.random()
            radius = self.max_radius * np.cbrt(np.random.random()) # 开立方根生成的半径才能保证在立方体内均匀分布
            x = radius * np.cos(azimuth) * np.sin(elevation)
            y = radius * np.sin(azimuth) * np.sin(elevation)
            z = radius * np.cos(elevation)
            self.position[i] = (x,y,z)
    
    def _init_moving_params(self):
        for i in range(self.n_slaves+1):
            self.azimuth[i] = 2 * np.pi * np.random.random()
            self.elevation[i] = np.pi * np.random.random()
            self.velocity[i] = 2 * self.slave_velocity * np.random.random()
        self.velocity[0] = 2 * self.master_velocity * np.random.random()

    def clip_to_master(self, slave_position, master_position):
        if distance(slave_position, master_position) < self.max_radius:
            return slave_position
        else:
            position = slave_position - master_position
            position = position / (distance(position, np.zeros(shape=(3,))) + 1e-3)
            position = position * self.max_radius
            return position + master_position

    def position_step(self):
        # master
        def rand():
            return np.random.random() - 0.5
        self.azimuth[0] = np.clip(self.azimuth[0] + 2*np.pi*rand()*self.moving_factor, 0, 2*np.pi)
        self.elevation[0] = np.clip(self.elevation[0] + np.pi*rand()*self.moving_factor, 0, np.pi)
        self.velocity[0] = np.clip(self.velocity[0] + self.master_velocity*rand()*self.moving_factor, 0,  2 * self.master_velocity)
        self.position[0][0] += self.velocity[0] * np.cos(self.azimuth[0]) * np.sin(self.elevation[0]) * self.dt
        self.position[0][1] += self.velocity[0] * np.sin(self.azimuth[0]) * np.sin(self.elevation[0]) * self.dt
        self.position[0][2] += self.velocity[0] * np.cos(self.elevation[0]) * self.dt
        # slave
        for i in range(1, self.n_slaves+1):
            self.azimuth[i] = np.clip(self.azimuth[i] + 2*np.pi*rand()*self.moving_factor, 0, 2*np.pi)
            self.elevation[i] = np.clip(self.elevation[i] + np.pi*rand()*self.moving_factor, 0, np.pi)
            self.velocity[i] = np.clip(self.velocity[i] + self.slave_velocity*rand()*self.moving_factor, 0,  2 * self.slave_velocity)
            x = self.position[i][0] + self.velocity[i] * np.cos(self.azimuth[i]) * np.sin(self.elevation[i]) * self.dt
            y = self.position[i][1] + self.velocity[i] * np.sin(self.azimuth[i]) * np.sin(self.elevation[i]) * self.dt
            z = self.position[i][2] + self.velocity[i] * np.cos(self.elevation[i]) * self.dt
            self.position[i] = (np.clip(x, 0, self.xlim), np.clip(y, 0, self.ylim), np.clip(z, self.zlim_min, self.zlim_max))
            self.position[i] = self.clip_to_master(self.position[i], self.position[0])


class JammerMoving(object):
    def __init__(
        self,n_jammers=3, init_position=None, xlim=1000, ylim=1000, zlim_max=200, zlim_min=50,
        jammer_velocity=10, moving_factor=0.1, dt=0.1
    ):
        self.n_jammers = n_jammers
        self.position = np.zeros(shape=(n_jammers, 3))
        self.azimuth = np.zeros(shape=(n_jammers,))
        self.elevation = np.zeros(shape=(n_jammers,))
        self.velocity = np.zeros(shape=(n_jammers,))
        # 移动参数
        self.jammer_velocity = jammer_velocity
        self.moving_factor = moving_factor
        self.dt = dt
        # 移动限制
        self.xlim = xlim
        self.ylim = ylim
        self.zlim_max = zlim_max
        self.zlim_min = zlim_min
        self.init_position = init_position
        self.reset_JammerMoving()

    def reset_JammerMoving(self):
        self._init_position(self.init_position)
        self._init_moving_params()
    
    def _init_position(self, init_potision=None):
        if init_potision is None:
            for i in range(self.n_jammers):
                self.position[i] = (random.uniform(0,self.xlim),random.uniform(0,self.ylim),random.uniform(self.zlim_min,self.zlim_max))
        else:
            self.position = init_potision
    
    def _init_moving_params(self):
        for i in range(self.n_jammers):
            self.azimuth[i] = 2 * np.pi * np.random.random()
            self.elevation[i] = np.pi * np.random.random()
            self.velocity[i] = 2 * self.jammer_velocity * np.random.random()
    
    def position_step(self):
        for i in range(self.n_jammers):
            def rand():
                return np.random.random() - 0.5
            self.azimuth[i] = np.clip(self.azimuth[i] + 2*np.pi*rand()*self.moving_factor, 0, 2*np.pi)
            self.elevation[i] = np.clip(self.elevation[i] + np.pi*rand()*self.moving_factor, 0, np.pi)
            self.velocity[i] = np.clip(self.velocity[i] + self.jammer_velocity*rand()*self.moving_factor, 0,  2 * self.jammer_velocity)
            x = self.position[i][0] + self.velocity[i] * np.cos(self.azimuth[i]) * np.sin(self.elevation[i]) * self.dt
            y = self.position[i][1] + self.velocity[i] * np.sin(self.azimuth[i]) * np.sin(self.elevation[i]) * self.dt
            z = self.position[i][2] + self.velocity[i] * np.cos(self.elevation[i]) * self.dt
            self.position[i] = (np.clip(x, 0, self.xlim), np.clip(y, 0, self.ylim), np.clip(z, self.zlim_min, self.zlim_max))