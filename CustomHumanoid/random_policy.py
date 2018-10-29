from custom_humanoid_env import *
import time
import logz
import gym


def run(env_name, logdir):
    if env_name == "toddler":
        env = CustomHumanoidEnv()
    else:
        env = gym.make("Humanoid-v2")
    env.reset()
    logz.configure_output_dir(logdir)
    for itr in range(100):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        time.sleep(0.1)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("AverageReturn", reward)
        logz.dump_tabular()

def main():
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    logdir = 'random_policy_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)
    for e in range(3):
        seed = 1 + 10*e
        print('Running experiment with seed %d'%seed)
        logdir2 = os.path.join(logdir,'%d'%seed)
        run("toddler", logdir2)



if __name__ == '__main__':
    main()
