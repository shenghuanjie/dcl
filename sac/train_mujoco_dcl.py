import sys

sys.path.insert(0, '..\\dcl')

import argparse
import gym
import sac.logz as logz
import numpy as np
import os
import tensorflow as tf
import time

import sac.nn as nn
from sac.sac import SAC
import sac.utils as utils

from multiprocessing import Process

# import CustomHumanoid
from envs.custom_humanoid_env import CustomHumanoidEnv
from envs.custom_lunar_lander import LunarLanderContinuous
from envs.custom_lunar_lander import GLOBAL_PARAMS


def make_dict_list(paras, nstep, type=float):
    return_list = []
    for istep in range(nstep):
        temp_dict = {}
        for para in paras:
            para2 = type(para[2])
            para1 = type(para[1])
            if para1 <= para2:
                temp_dict[para[0]] = type((para2 - para1) / nstep * istep + para1)
            else:
                temp_dict[para[0]] = type(para1 - (para1 - para2) / nstep * istep)
        return_list.append(temp_dict)
    return return_list


def train_SAC(env_name, exp_name, seed, logdir,
              two_qf=False, reparam=False, nepochs=50, nsteps=10,
              paras_dict=({'LEG_AWAY', 10, 110},), exp_replay=False):
    alpha = {
        'Ant-v2': 0.1,
        'HalfCheetah-v2': 0.2,
        'Hopper-v2': 0.2,
        'Humanoid-v2': 0.05,
        'Walker2d-v2': 0.2,
        'Toddler': 0.05,
        'Adult': 0.05,
        'LunarLander': 0.2
    }.get(env_name, 0.2)

    algorithm_params = {
        'alpha': alpha,
        'batch_size': 256,
        'discount': 0.99,
        'learning_rate': 1e-3,
        'reparameterize': reparam,
        'tau': 0.01,
        'epoch_length': 1000,
        'n_epochs': nepochs,  # 500
        'two_qf': two_qf,
    }
    sampler_params = {
        'max_episode_length': 1000,
        'prefill_steps': 1000,
    }
    replay_pool_params = {
        'max_size': 1e6,
    }

    value_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    q_function_params = {
        'hidden_layer_sizes': (128, 128),
    }

    policy_params = {
        'hidden_layer_sizes': (128, 128),
    }

    if env_name == 'Toddler' or env_name == 'Adult':
        env = CustomHumanoidEnv(template=env_name)
    elif env_name == 'LunarLander':
        env = LunarLanderContinuous()
    else:
        env = gym.envs.make(env_name)

    assert (env_name == 'LunarLander')
    envs = []
    for istep in range(nsteps):
        para_dict = paras_dict[istep]
        for k, v in para_dict.items():
            this_type = type(GLOBAL_PARAMS[k])
            para_dict[k] = this_type(v)
        envs.append(LunarLanderContinuous(**para_dict))
        # envs.append(LunarLanderContinuous(LEG_AWAY=int(10+istep*10),
        #                                   LEG_SPRING_TORQUE=int(100-istep*10)))

    logz.configure_output_dir(logdir)
    params = {
        'exp_name': exp_name,
        'env_name': env_name,
        'algorithm_params': algorithm_params,
        'sampler_params': sampler_params,
        'replay_pool_params': replay_pool_params,
        'value_function_params': value_function_params,
        'q_function_params': q_function_params,
        'policy_params': policy_params
    }
    logz.save_params(params)

    # Observation and action sizes
    ac_dim = env.action_space.n \
        if isinstance(env.action_space, gym.spaces.Discrete) \
        else env.action_space.shape[0]

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)
    env.seed(seed)

    q_function = nn.QFunction(name='q_function', **q_function_params)
    if algorithm_params.get('two_qf', False):
        q_function2 = nn.QFunction(name='q_function2', **q_function_params)
    else:
        q_function2 = None
    value_function = nn.ValueFunction(
        name='value_function', **value_function_params)
    target_value_function = nn.ValueFunction(
        name='target_value_function', **value_function_params)
    policy = nn.GaussianPolicy(
        action_dim=ac_dim,
        reparameterize=algorithm_params['reparameterize'],
        **policy_params)

    samplers = []
    replay_pools = []
    replay_pool = None
    sampler = None
    for istep in range(nsteps):
        env = envs[istep]
        env.seed(seed)
        if exp_replay:
            if replay_pool is None:
                replay_pool = utils.ExperienceReplayPool(
                    observation_shape=env.observation_space.shape,
                    action_shape=(ac_dim,),
                    **replay_pool_params)
            if sampler is None:
                sampler = utils.SimpleSampler(**sampler_params)
        else:
            replay_pool = utils.SimpleReplayPool(
                observation_shape=env.observation_space.shape,
                action_shape=(ac_dim,),
                **replay_pool_params)
            sampler = utils.SimpleSampler(**sampler_params)
        sampler.initialize(env, policy, replay_pool)
        samplers.append(sampler)
        replay_pools.append(replay_pool)

    algorithm = SAC(**algorithm_params)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True  # may need if using GPU
    with tf.Session(config=tf_config):
        algorithm.build(
            env=env,
            policy=policy,
            q_function=q_function,
            q_function2=q_function2,
            value_function=value_function,
            target_value_function=target_value_function)
        # algorithm_params.get('n_epochs', 1000)
        epoch = 0
        for istep in range(nsteps):
            sampler = samplers[istep]
            replay_pool = replay_pools[istep]
            for _ in algorithm.train(sampler, n_epochs=algorithm_params.get('n_epochs', 100)):
                logz.log_tabular('Iteration', epoch)
                for k, v in algorithm.get_statistics().items():
                    logz.log_tabular(k, v)
                for k, v in replay_pool.get_statistics().items():
                    logz.log_tabular(k, v)
                for k, v in sampler.get_statistics().items():
                    logz.log_tabular(k, v)
                logz.dump_tabular()
                epoch += 1
            if exp_replay:
                replay_pool.deprecate()


def train_func(args, logdir, seed, paras_dict):
    train_SAC(
        env_name=args.env_name,
        exp_name=args.exp_name,
        seed=seed,
        logdir=os.path.join(logdir, '%d' % seed),
        two_qf=args.two_qf,
        reparam=args.reparam,
        nepochs=args.n_epochs,
        nsteps=args.n_steps,
        paras_dict=paras_dict,
        exp_replay=args.exp_replay
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Toddler')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--two_qf', action='store_true')
    parser.add_argument('--reparam', action='store_true')
    parser.add_argument('--exp_replay', action='store_true')
    parser.add_argument('--n_steps', '-s', type=int, default=10)
    parser.add_argument('--n_epochs', '-ep', type=int, default=50)
    parser.add_argument('--paras', type=str, default='LEG_AWAY,10,110')
    args = parser.parse_args()

    paras = [para.split(',') for para in args.paras.split(';')]
    paras_dict = make_dict_list(paras, args.n_steps)

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = 'sac_' + args.env_name + '_' + args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)

    processes = []

    for e in range(args.n_experiments):
        seed = args.seed + 10 * e
        print('Running experiment with seed %d' % seed)

        """
        def train_func():
            train_SAC(
                env_name=args.env_name,
                exp_name=args.exp_name,
                seed=seed,
                logdir=os.path.join(logdir, '%d' % seed),
            )
        """
        # inputs = {'args': args, 'logdir': logdir, 'seed': seed}
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
        p = Process(target=train_func, args=(args, logdir, seed, paras_dict))
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.FATAL)
    main()
