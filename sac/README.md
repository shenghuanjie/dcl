# To run the experiment use the following commands

## To run curriculum learning, use the following command
python sac/train_mujoco_dcl.py --env_name CustomContinuousMountain-v2 --exp_name cl_sac_w0.0001-0.001 -e 3 --two_qf --reparam -ep 2 --s 10 --para weight,0.0001,0.001 --exp_replay
### To visualize the final learned model, use add --test to the command above.
python sac/train_mujoco_dcl.py --env_name CustomContinuousMountain-v2 --exp_name cl_sac_w0.0001-0.001 -e 3 --two_qf --reparam -ep 2 --s 10 --para weight,0.0001,0.001 --exp_replay --test

## To run the hard task, use the following command
python sac/train_mujoco.py --env_name CustomContinuousMountain-v2 --exp_name sac_w0.001 -e 3 --two_qf --reparam -ep 20 --para weight,0.001
### To visualize the final learned model, run the following command
python sac/train_mujoco_render.py --env_name CustomContinuousMountain-v2 --exp_name sac_w0.001 -e 1 --two_qf --reparam -ep 20 --para weight,0.001 --seed 21 --test

## To run the easy task, use the following command
python sac/train_mujoco.py --env_name CustomContinuousMountain-v2 --exp_name sac_w0.0001 -e 3 --two_qf --reparam -ep 20 --para weight,0.0001

# To plot the result, run:
python sac/plot.py sac/data/sac_CustomContinuousMountain-v2_sac_w0.0001_25-11-2018_15-00-20 sac/data/sac_CustomContinuousMountain-v2_cl_sac_w0.0001-0.001_25-11-2018_11-08-57 sac/data/sac_Continuous_MountainCar_sac_w0.001_18-11-2018_18-51-40 --value AverageReturn