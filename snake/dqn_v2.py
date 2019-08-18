import random
import numpy as np
from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

from environment.environment import SnakeEnvironment

from tensorboardX import SummaryWriter

GAMMA = 0.9                     # try .99
LEARNING_RATE = 0.001           # default is 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99995

# PARAMETERS ##################################################################
LEARN = True                   # False if using a trained model

NAME = 'first-attempt'
WRITE = True                    # Only for training
DRAW = True                     # Only for training
SAVE_MODEL = True               # Only for training

# Here you can load trained models:
LOAD_NAME = 'fist-attempt'
###############################################################################


class DQNSolver:

    def __init__(self, observation_space, action_space, model=None):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model is None:
            print('new model')
            self.model = Sequential()
            # andere aktivierungs funktion
            self.model.add(Dense(512, input_shape=(
                observation_space,), activation="relu"))
            self.model.add(Dense(512, activation="relu"))
            # self.model.add(Dropout(0.85))
            # self.model.add(Dense(512, activation="relu"))
            # Linear sucks? maybe try softmax
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(
                lr=LEARNING_RATE))    # Try learning rate deacy
            # self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_WITH_DECAY, decay=1e-6))
        else:
            print('saved model loaded')
            self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, env):
        if np.random.rand() < self.exploration_rate:
            return env.get_action_random()
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
                q_update = (reward + GAMMA *
                            np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def learn_snake():
    env = SnakeEnvironment(draw=DRAW, fps=20, debug=False)
    writer = None
    if WRITE:
        writer = SummaryWriter(comment=NAME)
    observation_space = env.get_observation_size()
    action_space = env.get_action_size()

    # model = load_model('models/dqn/{}.h5'.format(LOAD_NAME))
    dqn_solver = DQNSolver(observation_space, action_space)  # , model)
    run = 0

    if SAVE_MODEL:
        name = '{}-PART={}'.format(NAME, run)
        dqn_solver.model.save('models/dqn/{}.h5'.format(name))
    while True:
        run += 1
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        step = 0
        reward_score = 0

        while True:
            step += 1
            action = dqn_solver.act(state, env)
            state_next, reward, terminal, info = env.step(action)
            reward_score += reward
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                print("Run: " + str(run) + ", exploration: " +
                      str(dqn_solver.exploration_rate) + ", score: " +
                      str(reward_score))
                if WRITE:
                    writer.add_scalar("reward", reward_score, run)
                break
            dqn_solver.experience_replay()
        if (run % 100 == 0) and SAVE_MODEL:
            name = '{}-PART={}'.format(NAME, run)
            dqn_solver.model.save('models/dqn/{}.h5'.format(name))
    if WRITE:
        writer.close()


def play_snake():
    env = SnakeEnvironment(draw=True, fps=1, debug=True)

    observation_space = env.get_observation_size()
    action_space = env.get_action_size()

    model = keras.models.load_model('models/dqn/{}.h5'.format(LOAD_NAME))
    dqn_solver = DQNSolver(observation_space, action_space, model)

    for i in range(20):
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        is_done = False
        while not is_done:
            action = dqn_solver.act_free(state)
            # action = env.get_action_random()
            state_next, reward, terminal, info = env.step(action)
            is_done = terminal
            state = np.reshape(state_next, [1, observation_space])


if __name__ == "__main__":
    if LEARN:
        learn_snake()
    else:
        play_snake()

    print('Jobe Done!')
