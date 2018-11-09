import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
import os.path as osp
import os
from humanoid_builder import humanoid_xml_builder


def mass_center(model, data):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


def angle_from_quaternion(quat):
    # z' from quaternion [a,b,c,d] is
    # [ 2bd + 2ac,
    #   2cd - 2ab,
    #   a**2 - b**2 - c**2 + d**2
    # ]
    # so inner product with z = [0,0,1] is
    # z' dot z = a**2 - b**2 - c**2 + d**2
    a, b, c, d = quat[0], quat[1], quat[2], quat[3]
    return np.arccos(a**2 - b**2 - c**2 + d**2)


class CustomHumanoidEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, template='baby', override=False, obs_kind=1, **kwargs):
        self.head_idx = 0   # temp definition to avoid error in model init
        self.obs_kind = obs_kind
        if template == 'baby':
            #adult
            # template_kwargs = dict(gut_thickness=1.0,#1.5
            #                        butt_thickness=0.9, #1.4
            #                        lwaist_z=0.85,
            #                        leg_length=1.20, #0.62
            #                        leg_thickness=1.2, #1.4
            #                        arm_length=1.2, #0.7
            #                        arm_thickness=1.3, #1.7
            #                        hand_type=1, #2
            #                        hand_thickness=1.2, #1.7
            #                        torso_thickness=1.2, #1.4
            #                        span=0.1,
            #                        neck=1.4, #1.2
            #                        elbow_scale=3,
            #                        shoulder_scale=2,
            #                        ab_scale=4, #4
            #                        hip_scale=2.0, #1.6
            #                        knee_scale=1.6,
            #                        neck_scale=1.3,#1.5
            #                        foot_scale=10, #4
            #                        foot_type=1) #2

            # toddler
            template_kwargs = dict(gut_thickness=1.5,
                                   butt_thickness=1.4,
                                   lwaist_z=0.85,
                                   leg_length=0.62,
                                   leg_thickness=1.4,
                                   arm_length=0.7,
                                   arm_thickness=1.7,
                                   hand_type=1,
                                   hand_thickness=1.7,
                                   torso_thickness=1.4,
                                   span=0.9,
                                   neck=1.2,
                                   elbow_scale=3,
                                   shoulder_scale=2,
                                   ab_scale=4,
                                   hip_scale=1.6,
                                   knee_scale=1.6,
                                   neck_scale=1.5,
                                   foot_scale=4,
                                   foot_type=1)
            for k,v in template_kwargs.items():
                if not(override and k in kwargs):
                    kwargs[k] = v
        fullpath = humanoid_xml_builder(**kwargs)
        mujoco_env.MujocoEnv.__init__(self, fullpath, 5)
        self.head_idx = np.argmax([name == b'head' for name in self.model.body_names])
        utils.EzPickle.__init__(self, obs_kind=obs_kind, **kwargs)
        os.remove(fullpath)

    def _get_obs(self):
        data = self.data
        if not(hasattr(self, 'obs_kind')):
            self.obs_kind = 1
        if self.obs_kind==1:
            return np.concatenate([data.qpos.flat,
                                   data.qvel.flat,
                                   data.cvel.flat,
                                   data.cfrc_ext.flat])
        else:
            return np.concatenate([data.xanchor.flat,
                                   data.qvel.flat])

    def step(self, a):
        # MY OWN CODE
        pos_before = mass_center(self.model, self.data)
        self.do_simulation(a, self.frame_skip)
        # MY OWN CODE
        pos_after = mass_center(self.model, self.data)
        alive_bonus = 5.0
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(self.data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(self.data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus
        qpos = self.data.qpos
        done = bool((qpos[2] < 1.0) or (qpos[2] > 2.0))
        return self._get_obs(), reward, done, None

    def reset_model(self):
        c = 0.01 #0.35
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0#1
        self.viewer.cam.distance = self.model.stat.extent * 2#1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -30#-20
