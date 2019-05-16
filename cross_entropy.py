#!/usr/bin/env python3
import gym
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

from flappyb.environment import Environment

HIDDEN_SIZE = 128									# num of neurons in hidden layer
BATCH_SIZE = 16										# number of episodes in a batch
PERCENTILE_THROW_AWAY = 70							# percentage of episodes in batch to not learn from


class Net(nn.Module):
	def __init__(self, obs_size, hidden_size, n_actions):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(obs_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, n_actions)
		)
													# output is a probability distribution 
	def forward(self, x):							# ... over the actions
		return self.net(x)


# helpers to represent single steps and episodes from the actor 
Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):

	batch = []
	episode_reward = 0.0
	episode_steps = []
	obs = env.reset()
	sm = nn.Softmax(dim=1) 							# converts net output (raw action score)
													# ... to probability distribution
	while True:
		obs_v = torch.FloatTensor([obs])			# converts observation to tensor
		act_probs_v = sm(net(obs_v))				# then generate action probability policy
		act_probs = act_probs_v.data.numpy()[0]		# convert tensor back to array
		
		# choose an action according to the available probability
		action = np.random.choice(len(act_probs), p=act_probs)
		next_obs, reward, is_done, _ = env.step(action)

		# use obs that we started whith in this episode 
		episode_reward += reward
		episode_steps.append(EpisodeStep(observation=obs, action=action))

		# when episode (one single game) ends
		if is_done:
			# remember episode steps and clear environment
			batch.append(Episode(reward=episode_reward, steps=episode_steps))
			episode_reward = 0.0
			episode_steps = []
			next_obs = env.reset()

			# when batch is complete (multiple episodes) pass it to the learning loop
			if len(batch) == batch_size:
				yield batch
				batch = [] 

		obs = next_obs


def filter_batch(batch, percentile):
	rewards = list(map(lambda s: s.reward, batch))
	reward_bound = np.percentile(rewards, percentile)
	reward_mean = float(np.mean(rewards))

	train_obs = []
	train_act = []
	
	for example in batch:
		if example.reward < reward_bound:
			continue							# filters episodes
		train_obs.extend(map(lambda step: step.observation, example.steps))
		train_act.extend(map(lambda step: step.action, example.steps))

	train_obs_v = torch.FloatTensor(train_obs)
	train_act_v = torch.LongTensor(train_act)

	# return elite episodes as tensors 
	return train_obs_v, train_act_v, reward_bound, reward_mean


if __name__ == "__main__":

	env = Environment()

	# obs_size = env.observation_space.shape[0]
	# n_actions = env.action_space.n
	obs_size = env.get_observation_size()
	n_actions = env.get_action_size()

	net = Net(obs_size, HIDDEN_SIZE, n_actions)	# create neural net

	objective = nn.CrossEntropyLoss()			# main function to teach net
	optimizer = optim.Adam(params=net.parameters(), lr=0.01)
	writer = SummaryWriter(comment="-cross-entropy")

	# actual training loop
	for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
		# filter bad episodes so only the best episodes of a batch remain
		obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE_THROW_AWAY)
		optimizer.zero_grad()

		action_scores_v = net(obs_v) 			# pass obs to network again and retreive score
												# calculate cross entropy between net output and actions
												# ... the agent took inorder to learn the good actions
		loss_v = objective(action_scores_v, acts_v)			# calculate loss function
		loss_v.backward()						# apply gradient descent (not sure if this statement is correct)
		optimizer.step()						# optimize network

		print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (iter_no, loss_v.item(), reward_m, reward_b))
		writer.add_scalar("loss", loss_v.item(), iter_no)
		writer.add_scalar("reward_bound", reward_b, iter_no)
		writer.add_scalar("reward_mean", reward_m, iter_no)
		if iter_no > 500:
			print("500 steps should be sufficient")
			break
	writer.close()
