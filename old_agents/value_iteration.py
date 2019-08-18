# This sucks for flappy B
# CURRENTLY NO ROUNDING 
# saving not needed because it just sucks
# Take this for presentation, not q-learning or q-iteration

import collections
from tensorboardX import SummaryWriter
from flappyb.environment import Environment
import random
import numpy as np

GAMMA = .9
TEST_EPISODES = 5
TRAINING_STEPS = 3000

# NAME = 'gamma=0.9-trainingsteps:3000-rounding=None'
NAME = 'gamma=0.9-trainingsteps:3000-rounding=2'
WRITE = True
DRAW_TRAINING = False
DRAW = False


class Agent:
    def __init__(self):
        self.env = Environment(DRAW_TRAINING)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.get_action_random()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
        return action_value

    def select_action(self, state):
        best_action, best_value = None, None
        # for action in range(self.env.action_space.n):
        for action in range(self.env.get_action_size()):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    def value_iteration(self):
        for state in range(self.env.get_observation_size()):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.get_action_size())]
            self.values[state] = max(state_values)


if __name__ == "__main__":
    test_env = Environment(DRAW)
    agent = Agent()
    writer = None
    if WRITE:
        writer = SummaryWriter(comment='v_iteration/{}'.format(NAME))

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        print('#', iter_no)
        agent.play_n_random_steps(TRAINING_STEPS)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        if WRITE:
            writer.add_scalar("reward", reward, iter_no)
        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward
        if reward > 500:
            print("Solved in %d iterations!" % iter_no)
            break
    if WRITE:
        writer.close()
