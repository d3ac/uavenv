from gym.envs.registration import register

register(
    'uav-v0',
    entry_point='UAVenv.uav.uav:systemEnv',
)