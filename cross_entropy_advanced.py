import random
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from flappyb.environment import Environment

HIDDEN_SIZE_1 = 128
HIDDEN_SIZE_2 = 64
BATCH_SIZE = 100
PERCENTILE = 30
LEARNING_RATE = 0.01
GAMMA = 1

NAME = 'cross-entropy-batchsize:16-hiddensize1:128'
WRITE = False
DRAW = True

NAME = 'cross-entropy-advanced-128x64-gamma:1-learningrate:0.01'
WRITE = False
DRAW = True
SAVE_MODEL = False


class Net(nn.Module):
    def __init__(self, obs_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE_2, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        # action = env.get_action_random()

        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    # disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    disc_rewards = list(map(lambda s: s.reward * (len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == "__main__":
    random.seed(12345)
    env = Environment(DRAW)         # activate save

    obs_size = env.get_observation_size()
    n_actions = env.get_action_size()

    net = Net(obs_size, n_actions)
    # net.load_state_dict(torch.load('models/cross_entropy/{}.pt'.format(NAME)))
    # net.eval()

    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=LEARNING_RATE)

    writer = None
    if WRITE:
        writer = SummaryWriter(comment=NAME)

    full_batch = []
    for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
        full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        acts_v = torch.LongTensor(acts)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        action_scores_v = net(obs_v)
        loss_v = objective(action_scores_v, acts_v)
        loss_v.backward()
        optimizer.step()
        print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
            iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
        if WRITE:
            writer.add_scalar("reward_mean", reward_mean, iter_no)
        if (iter_no % 30 == 0) and SAVE_MODEL :
            NAME = 'NAME-PART={}'.format(iter_no)
            torch.save(net.state_dict(), 'models/cross_entropy/{}.pt'.format(NAME))
            pass
        if iter_no > 10000:
            print("That should be enough!")
            break

    if WRITE:
        writer.close()
