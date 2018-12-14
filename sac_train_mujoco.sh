#!/usr/bin/env bash

python sac/train_mujoco_dcl.py --env_name CustomContinuousMountain-v3 --exp_name cl_w0.0005-0.0025_ep2_s20 -e 3 --two_qf --reparam -ep 2 -s 20 --paras weight,0.0005,0.0025 --exp_replay --save

python sac/train_mujoco_dcl.py --env_name CustomContinuousMountain-v3 --exp_name cl_w0.0005-0.0025_ep4_s10 -e 3 --two_qf --reparam -ep 4 -s 10 --paras weight,0.0005,0.0025 --exp_replay --save

python sac/train_mujoco.py --env_name CustomContinuousMountain-v3 --exp_name sac_w0.0024 -e 3 --two_qf --reparam -ep 40 --para weight,0.0024 --exp_replay --save

python sac/train_mujoco.py --env_name CustomContinuousMountain-v3 --exp_name sac_w0.0005 -e 3 --two_qf --reparam -ep 40 --para weight,0.0005 --exp_replay --save