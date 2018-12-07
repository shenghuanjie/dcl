import sys

sys.path.insert(0, '..\\dcl')

import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import sac.logz as logz
import envs
import sac.nn as nn
import json
from tensorflow.keras.models import load_model
from tensorflow.keras import layers


def get_file(path, extension='h5'):
    files = os.listdir(path)
    for file in files:
        _, file_extension = os.path.splitext(file)
        if '.' + extension == file_extension:
            return file
    return None


def test_run(args, dir, max_steps=2000):
    with open(os.path.join(dir, 'params.json')) as f:
        json_params = json.load(f)
    env = gym.make(json_params['env_name'])
    # Observation and action sizes
    ac_dim = env.action_space.n \
        if isinstance(env.action_space, gym.spaces.Discrete) \
        else env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    tf.reset_default_graph()

    policy = nn.GaussianPolicy(
        action_dim=ac_dim,
        reparameterize=json_params['algorithm_params']['reparameterize'],
        **json_params['policy_params'])
    policy.build([None, obs_dim])
    # saver = tf.train.Saver()

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True  # may need if using GPU
    sess = tf.Session(config=tf_config)
    with sess.as_default():
        # policy = saver.restore(sess, os.path.join(dir, 'policy.h5'))
        # policy = load_model(os.path.join(dir, 'policy.h5'),
        #                    custom_objects={'GaussianPolicy': nn.GaussianPolicy,
        #                                    'DistributionLayer': nn.DistributionLayer})
        policy.load_weights(os.path.join(dir, 'policy.h5'))
        for e in range(args.n_experiments):
            seed = args.seed + 10 * e
            print('Running experiment with seed %d' % seed)

            tf.set_random_seed(args.seed)
            np.random.seed(args.seed)
            env.seed(args.seed)

            uid = 'seed_' + str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
            logz.configure_output_dir(dir, file='test-run' + '_' + uid + '.txt', check=False)

            env = gym.wrappers.Monitor(env, args.exp_name, force=True, uid=uid)
            obs = env.reset()
            for istep in range(max_steps):
                action = policy.eval(obs)
                obs, reward, done, _ = env.step(action)
                if args.render:
                    env.render()
                time.sleep(1e-3)
                logz.log_tabular('step', istep)
                for i, ob in obs:
                    logz.log_tabular('observation_' + str(i), obs)
                for j, act in action:
                    logz.log_tabular('action_' + str(j), act)
                logz.log_tabular('reward', reward)
                if done:
                    break
                logz.dump_tabular()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    for root, dir, files in os.walk(args.exp_name):
        if 'policy.h5' in files:
            test_run(args, root)


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.FATAL)
    main()
