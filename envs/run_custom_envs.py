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
    from envs.custom_continuous_mountain_car_v2 import Continuous_MountainCarEnv
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
    from envs.custom_continuous_mountain_car_v1 import Continuous_MountainCarEnv
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


def test_continuous_mountain_car_openai():
    import gym
    env = gym.make('MountainCarContinuous-v0')
    env.reset()
    while True:
        # action = env.action_space.sample()
        action = [1.0]
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        if done:
            env.reset()


def test_continuous_mountain_car_weights():
    import gym
    import envs.logz as logz

    logdir = 'data/test-mountaincar-weights_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logz.configure_output_dir(logdir)

    env = gym.make('CustomContinuousMountain-v2')
    weights = np.arange(0.0017, 0.001701, step=0.0000001)
    for weight in weights:
        env.reset(weight=weight)
        print('input weight: ', weight)
        print('actual weight: ', env.parameters['weight'])
        step = 0
        reward = np.nan
        while step < 10000:
            # action = env.action_space.sample()
            action = [1.0]
            obs, reward, done, _ = env.step(action)
            env.render()
            time.sleep(1e-3)
            if done:
                break
            step += 1
        logz.log_tabular('weight: ' + str(weight), reward)

    logz.dump_tabular()


def test_continuous_mountain_car_v3():
    import gym
    env = gym.make('CustomContinuousMountain-v3')
    env.reset()
    while True:
        # action = env.action_space.sample()
        action = [1.0]
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        if done:
            env.reset()

def test_arm3d_disc_env():
    from rllab_curriculum.curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv
    env = Arm3dDiscEnv()
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        if done:
            env.reset()

if __name__ == '__main__':
    envs = {'toddler': test_toddler,
            'humanoid': test_humanoid,
            'lunar_lander': test_custom_lunar_lander,
            'car': test_mountain_car,
            'car_cont_v2': test_continuous_mountain_car,
            'car_cont_v1': test_continuous_mountain_car_org,
            'car_cont_v0': test_continuous_mountain_car_openai,
            'car_cont_v3': test_continuous_mountain_car_v3,
            'car_cont_weights_test': test_continuous_mountain_car_weights,
            'arm3d_disc_env':test_arm3d_disc_env}

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-ev', choices='toddler, humanoid, lunar_lander, car, '
                                                     'car_cont_v0, car_cont_v1, car_cont_v2, '
                                                     'car_cont_v3, car_cont_weights_test'
                                                     'arm3d_disc_env',
                        default='car_cont_v3')
    args = parser.parse_args()
    envs[args.env_name]()

