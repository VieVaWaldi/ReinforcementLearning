# READ ME PEASE

# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
# LOSSS FUNCTIONS: https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
# BESST OPTIMIZER: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

# try this :()
# from keras.backend import manual_variable_initialization
# manual_variable_initialization(True)

import random
import numpy as np
from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

from flappyb.environment import Environment

from tensorboardX import SummaryWriter

GAMMA = 0.9
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99999

NAME = 'dqn-expdecay=0.99995-gamma=.9-batchsize=20-nn=512x512'
WRITE = False
DRAW = False
SAVE_MODEL = False

class DQNSolver:

    def __init__(self, observation_space, action_space, model = None):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model is None:
            print('new model')
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(observation_space,), activation="relu"))
            self.model.add(Dense(512, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))       # Linear sucks? maybe try softmax
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        else:
            print('saved model loaded')
            self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def act_free(self, state):
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def learn_flappyb():
    env = Environment(DRAW)
    writer = None
    if WRITE:
    	writer = SummaryWriter(comment=NAME)
    observation_space = env.get_observation_size()
    action_space = env.get_action_size()
    
    # model = load_model('models/dqn/{}.h5'.format(NAME))
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        reward_score = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step_buffer(action)
            # reward = reward if not terminal else -reward
            reward_score += reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " + str(dqn_solver.exploration_rate) + ", score: " + str(reward_score))
                if WRITE:
                	writer.add_scalar("reward", reward_score, run)
                break
            dqn_solver.experience_replay()
        if (run % 30 == 0) and SAVE_MODEL:
            # dqn_solver.model.model.save('models/dqn/{}.h5'.format(NAME))
            dqn_solver.model.save('models/dqn/{}.h5'.format(NAME))
            pass
    if WRITE:
    	writer.close()

def play_flappyb():
    env = Environment(True)

    observation_space = env.get_observation_size()
    action_space = env.get_action_size()
    
    model = keras.models.load_model('models/dqn/{}.h5'.format(NAME))
    dqn_solver = DQNSolver(observation_space, action_space, model)

    for i in range(5):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        is_done = False
        while not is_done:
            action = dqn_solver.act_free(state)
            state_next, reward, terminal, info = env.step_buffer(action)
            is_done = terminal
            state = np.reshape(state_next, [1, observation_space])

if __name__ == "__main__":
    # learn_flappyb()
    play_flappyb()
