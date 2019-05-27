import gym 
import collections
from tensorboardX import SummaryWriter
from flappyb.environment import Environment

GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20

NAME = 'q-learning'
WRITE = False
DRAW_TRAINING = False
DRAW = False

class Agent:
    def __init__(self):
        self.env = Environment(DRAW_TRAINING)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)    # less memory wasted, only store q-values

    # get s, a, r ,ns
    def sample_env(self):
        action = self.env.get_action_random()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return old_state, action, reward, new_state

    # iterate over all action values and return the best one
    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.get_action_size()):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    # q-value is calculated for s, a and stored in table
    def value_update(self, s, a, r, next_s):
        best_v, _ = self.best_value_and_action(next_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1 - ALPHA) + new_val * ALPHA

    # value table is not altered, only measures agent
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            _, action = self.best_value_and_action(state)
            new_state, reward, is_done, _ = env.step(action)
            total_reward += reward
            if is_done: 
                break
            state = new_state
        return total_reward

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
        s, a, r, next_s = agent.sample_env()
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        if WRITE:
            writer.add_scalar('reward', reward, iter_no)
        if reward > best_reward:
            print('Best reward updated %.3f => %.3f' %(best_reward, reward))
            best_reward = reward
        if reward > 0.9:
            print('Solved in %d iterations' %iter_no)
            break
    if WRITE:
        writer.close()