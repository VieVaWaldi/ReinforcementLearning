# Value iteration is not Q-Learning
import gym
import collections
from tensorboardX import SummaryWriter
from flappyb.environment import Environment
from numpy.random import choice

# GAMMA = 0.9
TEST_EPISODES = 20
RANDOM_STEPS = 10000

NAME = 'flappy-bird-value-iteration-10000-rand-steps-2-decimals'
MODEL_PATH = 'models/{}.pt'.format(NAME)
WRITE = True
DRAW = False

# Contains tabled and functions
class Agent:
    def __init__(self):
        self.env = Environment(False)
        self.state = self.env.reset()

        self.rewards = collections.defaultdict(float)
        self.transits = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    # Gather random experience and update tables
    # we can only learn on full episodes (unlike cross entropy)
    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.get_action_random()  # perform random action
            # action = choice((0,1), 1, p=(0.25, 0.75))
            # action = action.item(0)
            new_state, reward, is_done, _ = self.env.step(action)

            self.rewards[(self.state, action, new_state)] = reward  # reward = s, a, ns
            self.transits[(self.state, action)][new_state] += 1  # remembers occurence
            self.state = self.env.reset() if is_done else new_state

    # Simple probability calculation for the best states according to reward -> all goes in table
    def calc_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        total = sum(target_counts.values())
        action_value = 0.0
        for tgt_state, count in target_counts.items():
            reward = self.rewards[(state, action, tgt_state)]
            # action_value += (count / total) * (reward + GAMMA * self.values[tgt_state])
            action_value += (count / total) * (reward + self.values[tgt_state])
        return action_value

    # Uses calculated actions, searches the best for a given state and chooses it
    # play_n_random_steps is random, thus exploration is guaranteed
    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.get_action_size()):
            action_value = self.calc_action_value(state, action)
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        # print('Best action = {}'.format(best_action))
        return best_action

    # Not written into the table, cuz no randomness. Takes another env ?
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

    # Loop over every state, calculate values, update table with max value
    def value_iteration(self):
        for state in range(self.env.get_observation_size()):
            state_values = [self.calc_action_value(state, action) for action in range(self.env.get_action_size())]
            self.values[state] = max(state_values)


# Training loop whoop whoop
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
        print('#{}'.format(iter_no))
        agent.play_n_random_steps(RANDOM_STEPS)  # create data for table
        agent.value_iteration()  # use table to choose best action

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)  # evaluate agent
        reward /= TEST_EPISODES
        print('Current reward: {}'.format(reward))
        if WRITE:
            writer.add_scalar('reward', reward, iter_no)
        if reward > best_reward:
            print('Best reward update %.3f -> %.3f' % (best_reward, reward))
            best_reward = reward
        if iter_no > 5000:
            print('Job Done!')
            break
    if WRITE:
        writer.close()
