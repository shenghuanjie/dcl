### To run the experiment use the following commands

#### make sure your current path is `dcl`

#### To run sac WITHOUT curriculum learning, use:
python sac/train_mujoco.py --env_name CustomContinuousMountain-v3 --exp_name cl_w0.001 -e 1 --two_qf --reparam -ep 20 --para weight,0.001 --save

#### To run curriculum learning, use:
python sac/train_mujoco_dcl.py --env_name CustomContinuousMountain-v3 --exp_name cl_w0.001-0.002 -e 1 --two_qf --reparam --exp_replay -ep 2 -s 10 --paras weight,0.001,0.002 --save

#### To plot the result, run:
python sac/plot.py sac/data/sac_CustomContinuousMountain-v2_sac_w0.0001_25-11-2018_15-00-20 sac/data/sac_CustomContinuousMountain-v2_cl_sac_w0.0001-0.001_25-11-2018_11-08-57 sac/data/sac_Continuous_MountainCar_sac_w0.001_18-11-2018_18-51-40 --value AverageReturn

#### To run the saved model, run:
python sac/test_mujoco.py sac/data/sac_CustomContinuousMountain-v3_test_save_06-12-2018_00-08-30 --render
