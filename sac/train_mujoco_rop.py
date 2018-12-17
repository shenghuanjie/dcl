import sys
sys.path.insert(0, '../dcl')
sys.path.insert(0, './rllab_curriculum')

import os
import time
import argparse
import gym
import sac.logz as logz
import numpy as np
import tensorflow as tf

import sac.nn as nn
from sac.sac import SAC
import sac.utils as utils

from multiprocessing import Process

# import CustomHumanoid
from envs.custom_humanoid_env import CustomHumanoidEnv
from envs.custom_lunar_lander import LunarLanderContinuous
from envs.custom_continuous_mountain_car_v2 import Continuous_MountainCarEnv
from rllab_curriculum.curriculum.envs.arm3d.arm3d_disc_env import Arm3dDiscEnv

from rllab_curriculum.new.edit_xml import edit_xml


def make_dict_list(paras, nstep, type=float):
    if paras is None:
        return None
    return_list = []
    for istep in range(nstep):
        temp_dict = {}
        for para in paras:
            if para is None or len(para) < 2:
                continue
            para2 = type(para[2])
            para1 = type(para[1])
            if para1 <= para2:
                temp_dict[para[0]] = type((para2 - para1) / nstep * istep + para1)
            else:
                temp_dict[para[0]] = type(para1 - (para1 - para2) / nstep * istep)
        return_list.append(temp_dict)
    return return_list


def test_run(env, policy, sess, expt_dir='/tmp/cs294-112_project/', max_steps=2000):
    # env = gym.wrappers.Monitor(env, os.path.join(expt_dir, "gym"), force=True)
    # env.episode_id += 1
    obs = env.reset()
    with sess.as_default():
        for _ in range(max_steps):
            action = policy.eval(obs)
            obs, reward, done, _ = env.step(action)
            env.render()
            time.sleep(1e-3)
            if done:
                break


def train_SAC(env_name, exp_name, seed, logdir,
              two_qf=False, reparam=False, nepochs=50, nsteps=10,
              paras_dict=None, exp_replay=False):
    alpha = {
        'Ant-v2': 0.1,
        'HalfCheetah-v2': 0.2,
        'Hopper-v2': 0.2,
        'Humanoid-v2': 0.05,
        'Walker2d-v2': 0.2,
        'Toddler': 0.05,
        'Adult': 0.05,
        'LunarLander': 0.2,
        'CustomContinuousMountain-v2': 0.2,
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
    elif env_name == 'Arm3dDiscEnv':
        env = Arm3dDiscEnv()
    else:
        env = gym.make(env_name)

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

    replay_pool = utils.ExperienceReplayPool(
        observation_shape=env.observation_space.shape,
        action_shape=(ac_dim,),
        **replay_pool_params)
    sampler = utils.ExperienceSampler(**sampler_params)
    sampler.initialize(env, policy, replay_pool)

    algorithm = SAC(**algorithm_params)

    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
    tf_config.gpu_options.allow_growth = True  # may need if using GPU
    sess = tf.Session(config=tf_config)
    with sess.as_default():
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

            if env_name == 'Arm3dDiscEnv':
                edit_xml(value=list(paras_dict[istep].values())[0])
                sampler.env = Arm3dDiscEnv()
            else:
                env.reset(**paras_dict[istep])

            for _ in algorithm.train(sampler, n_epochs=algorithm_params.get('n_epochs', 100)):
                logz.log_tabular('Iteration', epoch)
                for k, v in algorithm.get_statistics().items():
                    logz.log_tabular(k, v)
                for k, v in replay_pool.get_statistics().items():
                    logz.log_tabular(k, v)
                for k, v in sampler.get_statistics().items():
                    logz.log_tabular(k, v)
                for k, v in paras_dict[istep].items():
                    logz.log_tabular(k, v)
                logz.dump_tabular()
                epoch += 1
            if exp_replay:
                replay_pool.deprecate()
            else:
                replay_pool.dump()
            sampler.clear()
    return env, policy, sess


def train_func(args, logdir, seed, paras_dict):
    return train_SAC(
        env_name=args.env_name,
        exp_name=args.exp_name,
        seed=seed,
        logdir=logdir,
        two_qf=args.two_qf,
        reparam=args.reparam,
        nepochs=args.n_epochs,
        nsteps=args.n_steps,
        paras_dict=paras_dict,
        exp_replay=args.exp_replay
    )


def test_or_save(args, logdir, seed, para_dict):
    env, policy, sess = train_func(args, logdir, seed, para_dict)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    if args.test:
        test_run(env, policy, sess, expt_dir=logdir)
    if args.save:
        with sess.as_default():
            policy.save(os.path.join(logdir, 'policy.h5'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='CustomContinuousMountain-v2')
    parser.add_argument('--exp_name', type=str, default='Test')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--two_qf', action='store_true')
    parser.add_argument('--reparam', action='store_true')
    parser.add_argument('--exp_replay', action='store_true')
    parser.add_argument('--n_steps', '-s', type=int, default=10)
    parser.add_argument('--n_epochs', '-ep', type=int, default=50)
    parser.add_argument('--paras', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    assert args.paras is not None
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
        logdir_seed = os.path.join(logdir, '%d' % seed)

        """
        def train_func():
            train_SAC(
                env_name=args.env_name,
                exp_name=args.exp_name,
                seed=seed,
                logdir=os.path.join(logdir, '%d' % seed),
            )
        """
        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
        if args.test or args.save:
            test_or_save(args, logdir_seed, seed, paras_dict)
        else:
            p = Process(target=train_func, args=(args, logdir_seed, seed, paras_dict))
            p.start()
            processes.append(p)
            # if you comment in the line below, then the loop will block
            # until this process finishes
            # p.join()

    for p in processes:
        p.join()

    # print(return_dict)
    # test_run(return_dict['env'], return_dict['policy'])


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.logging.set_verbosity(tf.logging.FATAL)
    main()
