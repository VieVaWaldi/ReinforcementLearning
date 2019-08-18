from environment.environment import Environment


class Agent:

    def __init__(self):
        self.total_reward = 0

    def step(self, env):
        env.get_observation_space()
        action = env.get_action_random()
        obs, reward, is_done, _ = env.step(action)
        self.total_reward += reward


# HUMAN PLAYS
env = Environment(draw=True, fps=20, debug=True, dist_to_pipe=40,
                  dist_between_pipes=150, obs_this_pipe=True)
env.run_human_game()


# RANDOM AGENT
# agent = Agent()
# env = Environment(True, 10)

# for i in range(10):
# 	env.reset()
# 	while not env.is_done:
# 	    agent.step(env)

# print("Total reward = {}".format(agent.total_reward))
