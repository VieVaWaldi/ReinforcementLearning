# Reinforcment Learning

This is mostly for myself, thus there wont be too much documentation. 

The algorithms were implemented using the book: "Deep Reinforcement Learning Hands-On" written by Maxim Lapan.
He provided a github repo with multiple implementations, that can be found in here:
https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On

## Project structure

The root folder consits of:
1. Different RL implementations
2. The environment
3. Trained models
4. Runs

## Explanations
* main.py is to play the game yourself
* The environment is a flappy bird clone, i tried to follow Open-Ai's approach when implementing the interface.
* /Runs is a log that plots the learning process via Tensorboard. 
* runTensorboard is a simple script that starts Tensoarboard if you provide a directory.

* There are multiple global parameters in every RL file and the environment that can be tweaked, like game speed, saving, NN size, drawing etc.

## ToDo
* The main difficulty right now is to improve the environment. The RL algoritms were tested and should be sufficient.
* Understand and implement Q-Learning.


