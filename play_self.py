from flappyb.environment import Environment
from numpy.random import choice


class Agent:

    def __init__(self):
        self.total_reward = 0

    def step(self, env):
        current_obs = env.get_observation_space()                 # emtpy for now
        action = env.get_action_random()
        obs, reward, is_done, _ = env.step(action)
        self.total_reward += reward


# HUMAN PLAYS

# env = Environment(True, 24)
# env.run_human_game()

# RANDOM AGENT

agent = Agent()
env = Environment(True, 10)

for i in range(10):
	env.reset()
	while not env.is_done:
	    agent.step(env)

print("Total reward = {}".format(agent.total_reward))
