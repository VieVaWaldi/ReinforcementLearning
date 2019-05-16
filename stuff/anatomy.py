import random

class Environment:

	def __init__(self):
		self.steps_left = 10

	def get_observation(self):
		""" Gives current environment state to agent """
		return [0.0, 0.0, 0.0]

	def get_actions(self):
		""" Possible actions of agent """
		return [0, 1]

	def is_done(self):
		""" End of episode """
		return self.steps_left == 0

	def action(self, action):
		""" Handles agents action and returns a reward for this action """
		if self.is_done():
			raise Exception('Game over')
		self.steps_left -= 1
		return random.random() # random reward

class Agent:

	def __init__(self):
		self.total_reward = 0.0

	def step(self, env):
		""" Allow agent to observe, think about action, take action and get reward """
		current_obs = env.get_observation()
		actions = env.get_actions()
		reward = env.action(random.choice(actions))
		self.total_reward += reward

if __name__ == "__main__":
	env = Environment()
	agent = Agent()

	while not env.is_done():
		agent.step(env)
	print("Total reward = {}".format(agent.total_reward))

print('Job Done!')