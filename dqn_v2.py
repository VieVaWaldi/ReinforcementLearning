# READ ME PEASE

# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288
# LOSSS FUNCTIONS: https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
# BEST OPTIMIZER: https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

import random
import numpy as np
from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation

from flappyb.environment import Environment

from tensorboardX import SummaryWriter
    
GAMMA = 0.9             # try .99
LEARNING_RATE = 0.001   # deafult was 0.001 
LEARNING_WITH_DECAY = 0.01    

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999995

#####################################################################################################
NAME = 'dqn-expdecay=0.999995-gamma=.9-batchsize=20-nn=512-lr=0.001-normalization-HARDCORE'
WRITE = True
DRAW = False
SAVE_MODEL = True
# XXX RIP MY CHILD XXX difference 200 is okay -> 250 # 600 is okay # 900 is funny # 1550 is op # no this is op 3550 # ! 3900 !
# LOAD_NAME = 'dqn-expdecay=0.99995-gamma=.9-batchsize=20-nn=512-lr=0.001-normalization-PART=6650'   # KING # 950 is oke # 1050 is oke # 6650 My baby is back <3
LOAD_NAME = 'dqn-expdecay=0.999995-gamma=.9-batchsize=20-nn=512-lr=0.001-normalization-HARDCORE-PART=6300'  # 2600 is pretty good # 6300 is god

#####################################################################################################



class DQNSolver:

    def __init__(self, observation_space, action_space, model = None):
        self.exploration_rate = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model is None:
            print('new model')
            self.model = Sequential()
            self.model.add(Dense(512, input_shape=(observation_space,), activation="relu")) # andere aktivierungs funktion
            self.model.add(Dense(512, activation="relu"))
            # self.model.add(Dropout(0.85))
            # self.model.add(Dense(512, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))       # Linear sucks? maybe try softmax
            self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))    # Try learning rate deacy
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
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def learn_flappyb():
    env = Environment(DRAW, 1, False)
    writer = None
    if WRITE:
        writer = SummaryWriter(comment=NAME)
    observation_space = env.get_observation_size_buffer()
    action_space = env.get_action_size()
    
    #model = load_model('models/dqn/newenv/{}.h5'.format(LOAD_NAME))
    dqn_solver = DQNSolver(observation_space, action_space) #, model)
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
            state_next, reward, terminal, info = env.step_buffer(action)
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
        if (run % 100 == 0) and SAVE_MODEL:
            name = '{}-PART={}'.format(NAME, run)
            dqn_solver.model.save('models/dqn/{}.h5'.format(name))
    if WRITE:
    	writer.close()



def play_flappyb():
    env = Environment(True, 1, False)

    observation_space = env.get_observation_size_buffer()
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
            state_next, reward, terminal, info = env.step_buffer(action)
            is_done = terminal
            state = np.reshape(state_next, [1, observation_space])



if __name__ == "__main__":
    # learn_flappyb()
    play_flappyb()
    
    print('Jobe Done!')
