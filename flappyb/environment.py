# Game was made with help of https://www.youtube.com/watch?v=cXgA1d_E-jY

# distance breite plotten
# 2 naechsten pipes als input geben

import time
import pygame
import random

from flappyb.bird import Bird
from flappyb.pipe import Pipe

import numpy as np
import torch

# GAME SPEED
FPS = 1


# AI PARAMETERS
BUFFER_SIZE = 4
ACTIONS = [0, 1]        
ACTION_SIZE = 2
OBSERVATION_SIZE = 5 * BUFFER_SIZE
ROUND_TO_DECIMALS = 2

# GAME PARAMETERS
SCREEN_SIZE = WIDTH, HEIGHT = (640, 880)
BACKGROUND = (0, 0, 0)
BIRD_COLOR = (255, 0, 0)
PIPE_COLOR = (0, 255, 0)
NEXT_PIPE = 140

""" 
Interace:
reset():                resets the whole environment
step(action):           performs one action onto the environment
step_buffer(action):    performs one action onto the environment, returns 4 states for experience replay
get_action_random():    obtain an imporoved random action
get_observation_size(): obtain size of observation
get_action_size():      obtain size of action
"""
class Environment:

    def __init__(self, draw):
        if draw:
            pygame.init()
            pygame.display.set_caption('NN FlappyB')

            self.font = pygame.font.SysFont("comicsansms", 72) # why though

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        self.pipes = [Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)]

        self.actions = ACTIONS
        self.reward = 0

        self.is_done = False
        self.draw = draw

        self.normalizer = Normalizer(OBSERVATION_SIZE)

    # ML INTERFACE ###############################################
    def reset(self):

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        self.pipes = [Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)]

        self.actions = ACTIONS
        self.reward = 0

        self.is_done = False

        obs, reward, is_done, _ = self.step_buffer(0)      # lol no premium action, why did i write that ?

        return obs

    def step(self, action):

        while not self.time_elapsed_since_last_action > FPS:
            dt = self.clock.tick()
            self.time_elapsed_since_last_action += dt

        self.global_time += 1

        return self.run_ai_game_step(action)

    def step_buffer(self, action):

        obs = []
        rew = 0
        don = False

        for i in range(BUFFER_SIZE):
            while not self.time_elapsed_since_last_action > FPS:
                dt = self.clock.tick()
                self.time_elapsed_since_last_action += dt

            self.global_time += 1
            o, r, d, _ = self.run_ai_game_step(action)
            rew += r

            for j in range(len(o)):
                obs.append(o[j])
            if d:
                don = True

        tensor = torch.FloatTensor(np.array(obs))    # Observation equals state        
        self.normalizer.observe(tensor)
        new_obs = self.normalizer.normalize(tensor)

        for i in range (len(new_obs)):
            obs[i] = round(float(new_obs[i]), ROUND_TO_DECIMALS)

        return obs, rew, d, _

    def get_observation_space(self):

        my_pipe = Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)
        my_pipe.x = 9999

        for pipe in self.pipes:
            if (pipe.x < my_pipe.x) and pipe.x >= (self.bird.x - pipe.width):
                my_pipe = pipe

        e1 = self.bird.y                    # bird pos
        e2 = self.bird.vel                  # bird vel
        e3 = my_pipe.x - self.bird.x        # dist to Pipe  
        e4 = my_pipe.top                    # pipe bot
        e5 = my_pipe.bot                    # pipe top

        # TURNING ON ROUNDING HELPS WITH Q-/V-ITERATION

        # obs = torch.FloatTensor(np.array([e1, e2, e3, e4, e5]))    # Observation equals state        
        # self.normalizer.observe(obs)
        # new_obs = self.normalizer.normalize(obs)

        # # nor_obs = np.interp(obs, (obs.min(), obs.max()), (0, 1))
        # wow1 = round(float(new_obs[0]), ROUND_TO_DECIMALS)
        # wow2 = round(float(new_obs[1]), ROUND_TO_DECIMALS)
        # wow3 = round(float(new_obs[2]), ROUND_TO_DECIMALS)
        # wow4 = round(float(new_obs[3]), ROUND_TO_DECIMALS)
        # wow5 = round(float(new_obs[4]), ROUND_TO_DECIMALS)

        obs = (e1, e2, e3, e4, e5)
        return obs

    def get_action_random(self):
        rand = random.uniform(0.2, 0.8)   # more or less and he does nothing
        action = np.random.choice((0, 1), 1, p=(rand, 1 - rand))
        return action.item(0)

    def get_observation_size(self):
        return OBSERVATION_SIZE

    def get_actions(self):
        return self.actions

    def get_actions_q_learning(self):
        return 1

    def get_action_size(self):
        return ACTION_SIZE

    # The actual game step ###

    def run_ai_game_step(self, action):  # call this in loop, action = bool

        current_reward = 0

        if self.global_time % NEXT_PIPE == 0:
            self.pipes.append(Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR))

        for pipe in self.pipes:
            pipe.update()
            if pipe.off_screen():
                self.pipes.remove(pipe)
            if pipe.hits(self.bird):
                if self.draw:
                    text = self.font.render("GAME OVER! Score = {}".format(self.reward), True, (186, 0, 0))
                    self.screen.blit(text,  (320 - text.get_width() // 2, 240 - text.get_height() // 2))
                    pygame.display.flip()
                    time.sleep(0.3)
                print('Score: {}'.format(self.reward))

                current_reward = -1
                self.is_done = True
            if pipe.behind(self.bird):
                
                self.reward += 1
                current_reward = 6

        self.bird.handle_events_ai(action)
        self.bird.update()

        if self.draw:
            self.screen.fill(BACKGROUND)
            for pipe in self.pipes:
                pipe.draw()
            self.bird.draw()
            pygame.display.update()

        self.time_elapsed_since_last_action = 0

        return self.get_observation_space(), current_reward, self.is_done, None

    # HUMAN STUFF ################################################

    def run_human_game(self):

        TEST_VAL_FOUR = 0

        while not self.is_done:

            while not self.time_elapsed_since_last_action > FPS:
                dt = self.clock.tick()
                self.time_elapsed_since_last_action += dt

            self.global_time += 1

            self.screen.fill(BACKGROUND)
            self.handle_events_human()

            if self.global_time % NEXT_PIPE == 0:
                self.pipes.append(Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR))

            for pipe in self.pipes:
                pipe.update()
                pipe.draw()
                if pipe.off_screen():
                    self.pipes.remove(pipe)
                if pipe.hits(self.bird):
                    print('Score: {}'.format(self.reward))
                    self.is_done = True
                    pygame.quit()
                if pipe.behind(self.bird):
                    self.reward += 1

            self.bird.handle_events_human()
            self.bird.update()
            self.bird.draw()

            pygame.display.update()

            self.time_elapsed_since_last_action = 0

            # TEST_VAL_FOUR += 1

            # if TEST_VAL_FOUR % 4 == 0:
            #     time.sleep(0.5)

    def handle_events_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_done = False
                pygame.quit()

# https://discuss.pytorch.org/t/normalization-of-input-data-to-qnetwork/14800 ### Please just work
class Normalizer():
    def __init__(self, num_inputs):
        self.n = torch.zeros(num_inputs)
        self.mean = torch.zeros(num_inputs)
        self.mean_diff = torch.zeros(num_inputs)
        self.var = torch.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.clone()
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = torch.clamp(self.mean_diff/self.n, min=1e-2)

    def normalize(self, inputs):
        obs_std = torch.sqrt(self.var)
        return (inputs - self.mean)/obs_std