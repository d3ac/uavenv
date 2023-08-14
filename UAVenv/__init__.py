from gym.envs.registration import register
from UAVenv.uav.uav import systemEnv

register(
    'uav-v0',
    entry_point='UAVenv.uav.uav:systemEnv',
)