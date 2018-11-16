import math

import Box2D
import gym
import numpy as np
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from gym import spaces
from gym.utils import seeding

# Rocket trajectory optimization is a classic topic in Optimal Control.
#
# According to Pontryagin's maximum principle it's optimal to fire engine full throttle or
# turn it off. That's the reason this environment is OK to have discreet actions (engine on or off).
#
# Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Solved is 200 points.
#
# Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land
# on its first attempt. Please see source code for details.
#
# Too see heuristic landing, run:
#
# python gym/envs/box2d/custom_lunar_lander.py
#
# To play yourself, run:
#
# python examples/agents/keyboard_agent.py LunarLander-v0
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.

# Modified by Sid Reddy (sgr@berkeley.edu) on 8/14/18
#
# Changelog:
# - different discretization scheme for actions
# - different terminal rewards
# - different observations
# - randomized landing site
#
# You can create an env object using `gym.make('LunarLanderContinuous-v2')`,
# and it will use the discrete action space specified in this file, even though
# the env is called "Continuous".
#
# A good agent should be able to achieve >150 reward.

GLOBAL_PARAMS = {

    'MAX_NUM_STEPS': 1000,

    'N_OBS_DIM': 9,
    'N_ACT_DIM': 6,  # num discrete actions

    'FPS': 50,
    'SCALE': 30.0,  # affects how fast-paced the game is, forces should be adjusted as well

    'MAIN_ENGINE_POWER': 13.0,
    'SIDE_ENGINE_POWER': 0.6,

    'INITIAL_RANDOM': 1000.0,  # Set 1500 to make game harder

    'LANDER_POLY': [
        (-14, +17), (-17, 0), (-17, -10),
        (+17, -10), (+17, 0), (+14, +17)
    ],
    'LEG_AWAY': 20,
    'LEG_DOWN': 18,
    'LEG_W': 2, 'LEG_H': 8,
    'LEG_SPRING_TORQUE': 40,  # 40 is too difficult for human players, 400 a bit easier

    'SIDE_ENGINE_HEIGHT': 14.0,
    'SIDE_ENGINE_AWAY': 12.0,

    'VIEWPORT_W': 600,
    'VIEWPORT_H': 400,

    'THROTTLE_MAG': 0.75,  # discretized 'on' value for thrusters
    'NOOP': 1,  # don't fire main engine, don't steer

    # my code
    'CHUNKS': 11 # range [3, 41]
}


def disc_to_cont(action, THROTTLE_MAG):  # discrete action -> continuous action
    if type(action) == np.ndarray:
        return action
    # main engine
    if action < 3:
        m = -THROTTLE_MAG
    elif action < 6:
        m = THROTTLE_MAG
    else:
        raise ValueError
    # steering
    if action % 3 == 0:
        s = -THROTTLE_MAG
    elif action % 3 == 1:
        s = 0
    else:
        s = THROTTLE_MAG
    return np.array([m, s])


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander == contact.fixtureA.body or self.env.lander == contact.fixtureB.body:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLander(gym.Env):
    continuous = False
    parameters = dict(GLOBAL_PARAMS)

    def __init__(self, **kwargs):

        for ikey in kwargs.keys():
            self.parameters[ikey] = kwargs[ikey]

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': self.parameters['FPS']
        }
        self._seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.particles = []

        self.prev_reward = None

        high = np.array([np.inf] * self.parameters['N_OBS_DIM'])  # useful range is -1 .. +1, but spikes can be higher
        self.observation_space = spaces.Box(-high, high)

        self.action_space = spaces.Discrete(self.parameters['N_ACT_DIM'])

        self.curr_step = None

        self._reset()

    # override
    def reset(self):
        return self._reset()

    # override
    def step(self, action):
        return self._step(action)

    # override
    def seed(self, seed=None):
        return self._seed(seed)

    # override
    def render(self, mode='human', close=False):
        return self._render(mode, close)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.moon: return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.moon = None
        self.world.DestroyBody(self.lander)
        self.lander = None
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])

    def _reset(self):
        self.curr_step = 0

        self._destroy()
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        W = self.parameters['VIEWPORT_W'] / self.parameters['SCALE']
        H = self.parameters['VIEWPORT_H'] / self.parameters['SCALE']

        # terrain
        CHUNKS = self.parameters['CHUNKS'] # 11
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        # randomize helipad x-coord
        helipad_chunk = np.random.choice(range(1, CHUNKS - 1))

        self.helipad_x1 = chunk_x[helipad_chunk - 1]
        self.helipad_x2 = chunk_x[helipad_chunk + 1]

        self.helipad_y = H / 4
        height[helipad_chunk - 2] = self.helipad_y
        height[helipad_chunk - 1] = self.helipad_y
        height[helipad_chunk + 0] = self.helipad_y
        height[helipad_chunk + 1] = self.helipad_y
        height[helipad_chunk + 2] = self.helipad_y
        smooth_y = [0.33 * (height[i - 1] + height[i + 0] + height[i + 1]) for i in range(CHUNKS)]

        self.moon = self.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys = []
        for i in range(CHUNKS - 1):
            p1 = (chunk_x[i], smooth_y[i])
            p2 = (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(
                vertices=[p1, p2],
                density=0,
                friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

        self.moon.color1 = (0.0, 0.0, 0.0)
        self.moon.color2 = (0.0, 0.0, 0.0)

        initial_y = self.parameters['VIEWPORT_H'] / self.parameters['SCALE']  # *0.75
        self.lander = self.world.CreateDynamicBody(
            position=(self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / self.parameters['SCALE'],
                                              y / self.parameters['SCALE'])
                                             for x, y in self.parameters['LANDER_POLY']]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.lander.color1 = (0.5, 0.4, 0.9)
        self.lander.color2 = (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter((
            self.np_random.uniform(-self.parameters['INITIAL_RANDOM'], self.parameters['INITIAL_RANDOM']),
            self.np_random.uniform(-self.parameters['INITIAL_RANDOM'], self.parameters['INITIAL_RANDOM'])
        ), True)

        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2
                          - i * self.parameters['LEG_AWAY'] / self.parameters['SCALE'], initial_y),
                angle=(i * 0.05),
                fixtures=fixtureDef(
                    shape=polygonShape(box=(self.parameters['LEG_W'] / self.parameters['SCALE'],
                                            self.parameters['LEG_H'] / self.parameters['SCALE'])),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.5, 0.4, 0.9)
            leg.color2 = (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * self.parameters['LEG_AWAY'] / self.parameters['SCALE'],
                              self.parameters['LEG_DOWN'] / self.parameters['SCALE']),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=self.parameters['LEG_SPRING_TORQUE'],
                motorSpeed=+0.3 * i  # low enough not to jump back into the sky
            )
            if i == -1:
                rjd.lowerAngle = +0.9 - 0.5  # Yes, the most esoteric numbers here, angles legs have freedom to travel within
                rjd.upperAngle = +0.9
            else:
                rjd.lowerAngle = -0.9
                rjd.upperAngle = -0.9 + 0.5
            leg.joint = self.world.CreateJoint(rjd)
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        return self._step(self.parameters['NOOP'])[0]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / self.parameters['SCALE'], pos=(0, 0)),
                density=mass,
                friction=0.1,
                categoryBits=0x0100,
                maskBits=0x001,  # collide only with ground
                restitution=0.3)
        )
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all):
        while self.particles and (all or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    def _step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid " % (action,type(action))
        if type(action) in [int, np.int64, np.int32]:
            action = disc_to_cont(action, self.parameters['THROTTLE_MAG'])

        # Engines
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / self.parameters['SCALE'] for _ in range(2)]

        m_power = 0.0
        if (self.continuous and action[0] > 0.0) or (not self.continuous and action == 2):
            # Main engine
            if self.continuous:
                m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5  # 0.5..1.0
                assert 0.5 <= m_power <= 1.0
            else:
                m_power = 1.0
            ox = tip[0] * (4 / self.parameters['SCALE'] + 2 * dispersion[0]) + side[0] * dispersion[
                1]  # 4 is move a bit downwards, +-2 for randomness
            oy = -tip[1] * (4 / self.parameters['SCALE'] + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            p = self._create_particle(3.5, impulse_pos[0], impulse_pos[1],
                                      m_power)  # particles are just a decoration, 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse((ox * self.parameters['MAIN_ENGINE_POWER'] * m_power,
                                  oy * self.parameters['MAIN_ENGINE_POWER'] * m_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * self.parameters['MAIN_ENGINE_POWER'] * m_power,
                                            -oy * self.parameters['MAIN_ENGINE_POWER'] * m_power),
                                           impulse_pos, True)

        s_power = 0.0
        if (self.continuous and np.abs(action[1]) > 0.5) or (not self.continuous and action in [1, 3]):
            # Orientation engines
            if self.continuous:
                direction = np.sign(action[1])
                s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
                assert 0.5 <= s_power <= 1.0
            else:
                direction = action - 2
                s_power = 1.0
            ox = tip[0] * dispersion[0] + side[0] * \
                 (3 * dispersion[1] + direction * self.parameters['SIDE_ENGINE_AWAY'] / self.parameters['SCALE'])
            oy = -tip[1] * dispersion[0] - side[1] * \
                 (3 * dispersion[1] + direction * self.parameters['SIDE_ENGINE_AWAY'] / self.parameters['SCALE'])
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / self.parameters['SCALE'],
                           self.lander.position[1] + oy + tip[1] * self.parameters['SIDE_ENGINE_HEIGHT'] /
                           self.parameters['SCALE'])
            p = self._create_particle(0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse((ox * self.parameters['SIDE_ENGINE_POWER'] * s_power,
                                  oy * self.parameters['SIDE_ENGINE_POWER'] * s_power), impulse_pos,
                                 True)
            self.lander.ApplyLinearImpulse((-ox * self.parameters['SIDE_ENGINE_POWER'] * s_power,
                                            -oy * self.parameters['SIDE_ENGINE_POWER'] * s_power),
                                           impulse_pos, True)

        # perform normal update
        self.world.Step(1.0 / self.parameters['FPS'], 6 * 30, 2 * 30)

        pos = self.lander.position
        vel = self.lander.linearVelocity
        helipad_x = (self.helipad_x1 + self.helipad_x2) / 2
        state = [
            (pos.x - self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2)
            / (self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2),
            (pos.y - (self.helipad_y + self.parameters['LEG_DOWN'] / self.parameters['SCALE']))
            / (self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2),
            vel.x * (self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2) / self.parameters['FPS'],
            vel.y * (self.parameters['VIEWPORT_H'] / self.parameters['SCALE'] / 2) / self.parameters['FPS'],
            self.lander.angle,
            20.0 * self.lander.angularVelocity / self.parameters['FPS'],
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (helipad_x - self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2)
            / (self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2)
        ]
        assert len(state) == self.parameters['N_OBS_DIM']

        self.curr_step += 1

        reward = 0
        shaping = 0
        dx = (pos.x - helipad_x) / (self.parameters['VIEWPORT_W'] / self.parameters['SCALE'] / 2)
        shaping += -100 * np.sqrt(state[2] * state[2] + state[3] * state[3]) - 100 * abs(state[4])
        shaping += -100 * np.sqrt(dx * dx + state[1] * state[1]) + 10 * state[6] + 10 * state[7]
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        reward -= m_power * 0.30  # less fuel spent is better, about -30 for heurisic landing
        reward -= s_power * 0.03

        oob = abs(state[0]) >= 1.0
        timeout = self.curr_step >= self.parameters['MAX_NUM_STEPS']
        not_awake = not self.lander.awake

        at_site = self.helipad_x1 <= pos.x <= self.helipad_x2 and state[1] <= 0
        grounded = self.legs[0].ground_contact and self.legs[1].ground_contact
        landed = at_site and grounded

        done = self.game_over or oob or not_awake or timeout or landed
        if done:
            if self.game_over or oob:
                reward = -100
                self.lander.color1 = (255, 0, 0)
            elif at_site:
                reward = +100
                self.lander.color1 = (0, 255, 0)
            elif timeout:
                self.lander.color1 = (255, 0, 0)
        info = {}

        return np.array(state), reward, done, info

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.parameters['VIEWPORT_W'], self.parameters['VIEWPORT_H'])
            self.viewer.set_bounds(0, self.parameters['VIEWPORT_W'] / self.parameters['SCALE'],
                                   0, self.parameters['VIEWPORT_H'] / self.parameters['SCALE'])

        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))
            obj.color2 = (max(0.2, 0.2 + obj.ttl), max(0.2, 0.5 * obj.ttl), max(0.2, 0.5 * obj.ttl))

        self._clean_particles(False)

        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20, color=obj.color2, filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        for x in [self.helipad_x1, self.helipad_x2]:
            flagy1 = self.helipad_y
            flagy2 = flagy1 + 50 / self.parameters['SCALE']
            self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(1, 1, 1))
            self.viewer.draw_polygon([(x, flagy2), (x, flagy2 - 10 / self.parameters['SCALE']),
                                      (x + 25 / self.parameters['SCALE'], flagy2 - 5 / self.parameters['SCALE'])],
                                     color=(0.8, 0.8, 0))

        clock_prog = self.curr_step / self.parameters['MAX_NUM_STEPS']
        self.viewer.draw_polyline(
            [(0, 0.05 * self.parameters['VIEWPORT_H'] / self.parameters['SCALE']),
             (clock_prog * self.parameters['VIEWPORT_W'] / self.parameters['SCALE'],
              0.05 * self.parameters['VIEWPORT_H'] / self.parameters['SCALE'])],
            color=(255, 0, 0), linewidth=5)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


class LunarLanderContinuous(LunarLander):
    continuous = True


def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0] * 0.5 + s[
        2] * 1.0  # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ > 0.4: angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55 * np.abs(s[0])  # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
    # print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5
    # print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]:  # legs have contact
        angle_todo = 0
        hover_todo = -(s[3]) * 0.5  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


if __name__ == "__main__":
    # env = LunarLander()
    env = LunarLanderContinuous()
    s = env.reset()
    total_reward = 0
    steps = 0
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        env.render()
        total_reward += r
        if steps % 20 == 0 or done:
            print(["{:+0.2f}".format(x) for x in s])
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
