from custom_humanoid_env import *
import time
env = CustomHumanoidEnv()
env.reset()
while True:
	env.step(env.action_space.sample())
	env.render()
	time.sleep(1e-3)