# Reinforcement Learning on games

The algorithms were implemented using the book: "Deep Reinforcement Learning Hands-On" written by Maxim Lapan.
He provides a github repo with multiple implementations, that can be found in here:
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

## Project structure

The root folder consits of:
1. Different games, every game has a RL algorithm, models and graphs
2. requirements.txt (probably not up to date)
3. runTensorBoard [dir], runs tensorboard on a choosen directory
4. old_agents, implementations of weaker RL algorithms

When you want to try out trained model, you have to set the LEARN flag in the agent file to false.
Different models are trained on different observations, so not every combination will work. 
But the models name indicates the settings for the parameters.

## Current Algorithms

* DQN, a simple dqn implementation that offers experience replay. This is currently the best algorithm in this repository.
* PPO, what everyone currently uses.
* Value iteration, a good starting point.
* cross_entropy, another starting point.
* others ...
