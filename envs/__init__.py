from gym.envs.registration import register

register(
    id='CustomContinuousMountain-v1',
    entry_point='envs.custom_continuous_mountain_car_v1:Continuous_MountainCarEnv',
)

register(
    id='CustomContinuousMountain-v2',
    entry_point='envs.custom_continuous_mountain_car_v2:Continuous_MountainCarEnv',
)

register(
    id='CustomContinuousMountain-v3',
    entry_point='envs.custom_continuous_mountain_car_v3:Continuous_MountainCarEnv',
)

register(
    id='CustomLunarLander-v1',
    entry_point='envs.custom_lunar_lander:LunarLanderContinuous',
)


