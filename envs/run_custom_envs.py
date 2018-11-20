import sys
sys.path.insert(0, '..\\dcl')

from envs.custom_humanoid_env import *
import time


def test_toddler():
    env = CustomHumanoidEnv()
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        # print(action, reward)


def test_humanoid():
    import gym
    env = gym.make("Humanoid-v2")
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        # print(action.shape, reward)


def test_custom_lunar_lander():
    from envs.custom_lunar_lander import LunarLanderContinuous
    env = LunarLanderContinuous()
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        if done:
            env.reset()
        # #print(action, reward)


def test_mountain_car():
    from envs.custom_mountain_car import MountainCarEnv
    env = MountainCarEnv(magnitude=0)
    env.reset()
    while True:
        # action = env.action_space.sample()
        action = 2
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        if done:
            env.reset()


def test_continuous_mountain_car():
    from envs.custom_continuous_mountain_car import Continuous_MountainCarEnv
    env = Continuous_MountainCarEnv(weight=0.001)
    env.reset()
    while True:
        # action = env.action_space.sample()
        action = [1.0]
        obs, reward, done, _ = env.step(action)
        env.render()
        print(obs, action)
        time.sleep(1e-3)
        if done:
            env.reset()


def test_continuous_mountain_car_org():
    from envs.custom_continuous_mountain_car_org import Continuous_MountainCarEnv
    env = Continuous_MountainCarEnv()
    env.reset()
    while True:
        # action = env.action_space.sample()
        action = [0.5]
        obs, reward, done, _ = env.step(action)
        env.render()
        print(obs, action)
        time.sleep(1e-3)
        if done:
            env.reset()


if __name__ == '__main__':

    envs = {'toddler': test_toddler,
            'humanoid': test_humanoid,
            'lunar_lander': test_custom_lunar_lander,
            'car': test_mountain_car,
            'car_cont': test_continuous_mountain_car,
            'car_cont_org': test_continuous_mountain_car_org}

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-ev', choices='toddler, humanoid, lunar_lander, car, car_cont, car_cont_org',
                        default='car_cont')
    args = parser.parse_args()

    envs[args.env_name]()

