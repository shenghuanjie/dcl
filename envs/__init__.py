from gym.envs.registration import register

register(
    id='CustomContinuousMountain-v2',
    entry_point='envs.custom_continuous_mountain_car:Continuous_MountainCarEnv',
)

register(
    id='CustomLunarLander-v1',
    entry_point='envs.custom_lunar_lander:LunarLanderContinuous',
)
