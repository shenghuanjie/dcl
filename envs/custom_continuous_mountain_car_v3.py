# -*- coding: utf-8 -*-
"""
@author: Olivier Sigaud

A merge between two sources:

* Adaptation of the MountainCar Environment from the "FAReinforcement" library
of Jose Antonio Martin H. (version 1.0), adapted by  'Tom Schaul, tom@idsia.ch'
and then modified by Arnaud de Broissia

* the OpenAI/gym MountainCar environment
itself from 
http://incompleteideas.net/sutton/MountainCar/MountainCar1.cp
permalink: https://perma.cc/6Z2N-PFWC
"""

import math

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

import os


class Continuous_MountainCarEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, **kwargs):
        # my own code
        self.parameters = {'offset': 0.1,
                           'magnitude': 0.45,
                           'weight': 0.0017,
                           'frequency': 3,
                           'max_step': 120,
                           'goal_positions': [-1.5, 0.5],
                           'max_speed': 0.07,
                           'power': 0.0015,
                           'initial_position': 0,
                           'initial_position_range': 0.1,
                           'max_position': 0.7,
                           'min_position': -1.7,
                           'min_action': -1.0,
                           'max_action': 1.0}
        # The mountain car cannot reach the goal directly if the weight is greater than 0.0017
        # self.weight_scalar = 1e-4
        self.num_step = 0
        self.set_paras(**kwargs)

        self.low_state = np.array([self.parameters['min_position'], -self.parameters['max_speed']])
        self.high_state = np.array([self.parameters['max_position'], self.parameters['max_speed']])

        self.viewer = None

        self.action_space = spaces.Box(low=self.parameters['min_action'], high=self.parameters['max_action'], shape=(1,))
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state)

        self.seed()
        self.reset()

    def set_paras(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.parameters.keys():
                datatype = type(self.parameters[k])
                self.parameters[k] = datatype(v)
        # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        if not isinstance(self.parameters['goal_positions'], list) or len(self.parameters['goal_positions']) < 2:
            if self.parameters['goal_positions'] < self.parameters['initial_position']:
                self.parameters['goal_positions'] = [self.parameters['goal_positions'],
                                                     self.parameters['max_position'] + 1]
            else:
                self.parameters['goal_positions'] = [self.parameters['min_positions'] - 1,
                                                     self.parameters['max_position']]
        else:
            self.parameters['goal_positions'] = self.parameters['goal_positions'][0:2]
        self.parameters['max_speed'] = self.parameters['max_speed']

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        degree = math.atan(math.cos(self.parameters['frequency'] * position) * self.parameters['frequency'])
        velocity += (force * self.parameters['power'] - self.parameters['weight'] * math.sin(degree)) * math.cos(degree)

        if velocity > self.parameters['max_speed']: velocity = self.parameters['max_speed']
        if velocity < -self.parameters['max_speed']: velocity = -self.parameters['max_speed']
        position += velocity
        if position > self.parameters['max_position']:
            position = self.parameters['max_position']
        if position < self.parameters['min_position']:
            position = self.parameters['min_position']
        if position == self.parameters['min_position'] and velocity < 0:
            # velocity = 0
            velocity = -velocity # bounce off the wall

        # reach either side would end the simulation.
        done = bool(position >= self.parameters['goal_positions'][1] or
                    position <= self.parameters['goal_positions'][0])

        reward = 0
        # self.parameters - self.num_step
        self.num_step += 1
        if done:
            reward += 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}

    def reset(self, **kwargs):
        self.set_paras(**kwargs)
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state)

    #    def get_state(self):
    #        return self.state

    def _height(self, xs):
        return np.sin(self.parameters['frequency'] * xs) * self.parameters['magnitude'] + \
               self.parameters['magnitude'] + self.parameters['offset']

    def render(self, mode='human'):

        world_width = self.parameters['max_position'] - self.parameters['min_position']
        ratio = (2 * self.parameters['magnitude'] + self.parameters['offset']) / world_width * 2

        screen_width = 600
        screen_height = int(400 * ratio) + 50
        if screen_height < 400:
            screen_height = 400

        scale = screen_width / world_width
        carwidth = 40
        carheight = 20

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.parameters['min_position'], self.parameters['max_position'], 100)
            ys = self._height(xs)
            xys = list(zip((xs - self.parameters['min_position']) * scale, ys * scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
            car = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight / 2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(carwidth / 4, clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight / 2.5)
            backwheel.add_attr(rendering.Transform(translation=(-carwidth / 4, clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            for goal_position in self.parameters['goal_positions']:
                flagx = (goal_position - self.parameters['min_position']) * scale
                flagy1 = self._height(goal_position) * scale
                flagy2 = flagy1 + 50
                flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
                self.viewer.add_geom(flagpole)
                flag = rendering.FilledPolygon([(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])
                flag.set_color(.8, .8, 0)
                self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation((pos - self.parameters['min_position']) * scale, self._height(pos) * scale)
        self.cartrans.set_rotation(math.atan(math.cos(self.parameters['frequency'] * pos)
                                   * self.parameters['magnitude']
                                   * self.parameters['frequency']))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()
