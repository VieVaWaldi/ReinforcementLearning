# value_iteration does the same but converges faster

import gym
from flappyb.environment import Environment
import collections
from tensorboardX import SummaryWriter
import random
import numpy as np

GAMMA = 0.9
TEST_EPISODES = 5
TRAINING_STEPS = 3000

WRITE = False
DRAW_TRAINING = False
DRAW = False
NAME = 'q-iteration-gamma:0.2-trainingsteps:3000-newenv-roundto:1'


class Agent:
    def __init__(self):
        # self.env = gym.make(ENV_NAME)
        self.env = Environment(DRAW_TRAINING)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        # rand = random.uniform(0.2, 0.8)     # more or less and he does nothing
        for _ in range(count):
            # if _ % 1000 == 0:
            #     rand = random.uniform(0.2, 0.8)
            #     print(rand)
            # action = np.random.choice((0, 1), 1, p=(rand, 1 - rand))
            # action = action.item(0)
            action = self.env.get_action_random()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state
            # print(len(self.transits))

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.get_action_size()):
            action_value = self.values[(state, action)]
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
            for action in range(self.env.get_action_size()):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count / total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
                self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = Environment(DRAW)
    agent = Agent()
    writer = None
    if WRITE:
        writer = SummaryWriter(comment=NAME)

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
        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break
    if WRITE:
        writer.close()
