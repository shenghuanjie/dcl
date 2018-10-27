from CustomHumanoid.custom_humanoid_env import *
import time


def test_toddler():
    env = CustomHumanoidEnv()
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        print(action, reward)


def test_humanoid():
    import gym
    env = gym.make("Humanoid-v2")
    env.reset()
    while True:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        time.sleep(1e-3)
        print(action.shape, reward)


if __name__ == '__main__':
    # test_toddler()
    test_humanoid()
