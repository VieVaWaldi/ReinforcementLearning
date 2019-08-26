import random
import numpy as np
from collections import deque
import os

# FORCE CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from keras.models import load_model

from environment.environment import SnakeEnvironment

# from tensorboardX import SummaryWriter

# ToDo:
# try negative reward for sidestep, after n useless steps
# make buffer
# understand why score sometimes is so huge

GAMMA = 0.9                     # try .99
LEARNING_RATE = 0.001           # default is 0.001

MEMORY_SIZE = 800000
BATCH_SIZE = 20

EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99995

# PARAMETERS ##################################################################
LEARN = False                    # False if using a trained model

NAME = '5*5-new-paras-512-obs'
WRITE = False                   # Only for training
DRAW = False                    # Only for training
SAVE_MODEL = True               # Only for training

# Here you can load trained models:
# LOAD_NAME = '5*5-first-attempt-PART=6000'
# LOAD_NAME = '5*5-bad-when-no-apple-PART=21800'
# LOAD_NAME = '5*5-new-paras-i1024o-PART=22000'
LOAD_NAME = '5*5-new-paras-512-obs-PART=100000'
###############################################################################


class DQNSolver:

    def __init__(self, observation_space, action_space, model=None):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model is None:
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(
                observation_space,), activation="relu"))
            self.model.add(Dense(512, activation="relu"))
            # self.model.add(Dropout(0.85))
            self.model.add(Dense(self.action_space, activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(
                lr=LEARNING_RATE))    # Try learning rate deacy
            # self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_WITH_DECAY, decay=1e-6))
        else:
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
    env = SnakeEnvironment(draw=DRAW, fps=1, debug=False)
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
            with open("{}.txt".format(NAME),"a") as f:
                f.write("#Run {}, Score: {}\n".format(run, reward_score))
    if WRITE:
        writer.close()


def play_snake(load_name=""):
    env = SnakeEnvironment(draw=True, fps=40, debug=False)


    observation_space = env.get_observation_size()
    action_space = env.get_action_size()

    model = None
    if load_name is not "":
        model = keras.models.load_model('models/dqn/{}.h5'.format(load_name))
    else:
        model = keras.models.load_model('models/dqn/{}.h5'.format(LOAD_NAME))

    dqn_solver = DQNSolver(observation_space, action_space, model)
    reward_total = 0

    for i in range(50):
        reward_total = 0

        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        is_done = False
        
        step = 0
        while not is_done:
            step += 1
            action = dqn_solver.act_free(state)
            state_next, reward, terminal, info = env.step(action)
            reward_total += reward
            is_done = terminal
            state = np.reshape(state_next, [1, observation_space])
            if step == 100:
                is_done = True
        print(reward_total)
    return reward_total

if __name__ == "__main__":
    if LEARN:
        learn_snake()
    else:
        play_snake()
        # top_rew = -10.0
        # top_ep = 0

        # for i in range(1500, 21100, 100):
        #     print('# run ', i)
        #     rew = play_snake(load_name='5*5-bad-when-no-apple-PART={}'.format(i))
        #     if rew > top_rew:
        #         top_rew = rew
        #         top_ep = i
        #     print('top rew: ', top_rew, ' #', top_ep)

        # print('\n****\n\nFinally: top reward, ', top_rew)
        # print('top episode, ', top_ep)

    print('Jobe Done!')
