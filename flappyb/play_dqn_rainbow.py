#!/usr/bin/env python3
from environment.environment import Environment
import time
import numpy as np

import torch
import torch.nn as nn

import collections

import ptan
from lib import dqn_model


MODEL_NAME = "flappyb-test-the-rainbow350"
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)
NUMBER_NEURONS = 512


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(input_shape, NUMBER_NEURONS),
            nn.ReLU(),
            dqn_model.NoisyLinear(NUMBER_NEURONS, N_ATOMS)
        )

        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(input_shape, NUMBER_NEURONS),
            nn.ReLU(),
            dqn_model.NoisyLinear(NUMBER_NEURONS, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / NUMBER_NEURONS
        val_out = self.fc_val(fx).view(batch_size, 1, N_ATOMS)
        adv_out = self.fc_adv(fx).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + (adv_out - adv_mean)

    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]

    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())


if __name__ == "__main__":

    env = Environment(draw=True, fps=1, debug=True,
                      dist_to_pipe=50, dist_between_pipes=180, obs_this_pipe=True)

    net = RainbowDQN(env.observation_space.n, env.action_space.n)
    net.load_state_dict(torch.load("models/" + MODEL_NAME, map_location=lambda storage, loc: storage))    

    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), ptan.actions.ArgmaxActionSelector())

    for i in range(10):
        state = env.reset()
        total_reward = 0.0
        c = collections.Counter()

        while True:
            start_ts = time.time()
            state_v = torch.tensor(np.array([state], copy=False))
            q_vals = agent(state_v) #.data.numpy()[0]
            action = q_vals[0][0]
            print(action)
            
            c[action] += 1
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
