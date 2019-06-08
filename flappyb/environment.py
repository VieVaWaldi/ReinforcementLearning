# Game was made with the help of https://www.youtube.com/watch?v=cXgA1d_E-jY
import time
import pygame
import random

import numpy as np
import torch

from flappyb.bird import Bird
from flappyb.pipe import Pipe

# AI PARAMETERS #####################################################################################
BUFFER_SIZE = 4
OBSERVATION_SIZE = 5 
ACTIONS = [0, 1]        
ACTION_SIZE = 2
# ROUND_TO_DECIMALS = 2

# GAME PARAMETERS ###################################################################################
SCREEN_SIZE = WIDTH, HEIGHT = (640, 880)
BACKGROUND = (146, 183, 254)
BIRD_COLOR = (241, 213, 19)
PIPE_COLOR = (44, 176, 26)
NEXT_PIPE = 80  # default 150, 80 looks good



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

    def __init__(self, draw=True, fps=10, debug=False):
        if draw:
            pygame.init()
            pygame.display.set_caption('NN FlappyB')

            self.font = pygame.font.SysFont("comicsansms", 72) # :)
            self.bg = pygame.image.load("flappyb/assets/bg.png")

        self.fps = fps
        self.debug = debug
        self.draw = draw

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        self.pipes = [Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)]

        self.reward = 0
        self.is_done = False
        self.printed_score = False

    # ML INTERFACE ##################################################################################
    def reset(self):

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        self.pipes = [Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)]

        self.reward = 0
        self.is_done = False
        self.printed_score = False

        obs, reward, is_done, _ = self.step_buffer(0)      # lol no premium action, why did i write that ?

        return obs

    def step(self, action):

        while not self.time_elapsed_since_last_action > self.fps:
            dt = self.clock.tick()
            self.time_elapsed_since_last_action += dt

        self.global_time += 1

        return self.run_ai_game_step(action)

    def step_buffer(self, action):

        obs = []
        rew = 0

        for i in range(BUFFER_SIZE):
            while not self.time_elapsed_since_last_action > self.fps:
                dt = self.clock.tick()
                self.time_elapsed_since_last_action += dt

            self.global_time += 1
            o, r, d, _ = self.run_ai_game_step(action)
            rew += r

            for j in range(len(o)):
                obs.append(o[j])

        # Rounding to if wanted #########################
        # for i in range (len(new_obs)):
            # obs[i] = round(float(new_obs[i]), 6)
            # obs[i] = new_obs[i]

        if rew > 1:
            rew = 1
        elif rew < -1:
            rew = -1
        else:
            rew = 0.1

        return obs, rew, d, _

    # The actual game step ##########################################################################
    def run_ai_game_step(self, action): 

        current_reward = 0.1 

        if self.global_time % NEXT_PIPE == 0:
            self.pipes.append(Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR))

        for pipe in self.pipes:
            pipe.update()
            
            if pipe.off_screen():
                self.pipes.remove(pipe)

            if pipe.hits(self.bird):
                self.game_over()
                current_reward = -1
                hit_pipe = True

            if pipe.behind(self.bird):    
                self.reward += 1
                current_reward = 1

        self.bird.handle_events_ai(action)
        if self.bird.update():
            self.game_over()
            current_reward = -1

        if self.draw:
            # self.screen.fill(BACKGROUND)
            self.screen.blit(self.bg, (0, 0))
            for pipe in self.pipes:
                pipe.draw()
            self.bird.draw()
            text = pygame.font.SysFont("comicsansms", 28).render("SCORE {}".format(self.reward), True, (0,0,0))
            self.screen.blit(text,  (565 - text.get_width() // 2, 30 - text.get_height() // 2))
            pygame.display.flip()

        obs = self.get_observation_space()
        
        if self.draw:
            pygame.display.update()

        self.time_elapsed_since_last_action = 0

        return obs, current_reward, self.is_done, None
    #################################################################################################

    def get_observation_space(self):

        my_pipe = Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)
        my_pipe.x = 9999

        for pipe in self.pipes:
            if (pipe.x < my_pipe.x) and pipe.x >= (self.bird.x - pipe.width):
                my_pipe = pipe

        e1 = self.bird.y                    # bird pos
        e2 = self.bird.vel                  # bird vel
        e3 = my_pipe.x - self.bird.x        # dist to Pipe  
        e4 = my_pipe.top                    # pipe top
        e5 = my_pipe.bot                    # pipe bot

        if self.draw and self.debug:
            e_d1 = pygame.rect.Rect(self.bird.x, e1, 2, HEIGHT-e1)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d1)

            e_d2 = pygame.rect.Rect(self.bird.x-self.bird.radius, e2*2+HEIGHT/2, self.bird.x+self.bird.radius, 5)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d2)

            e_d3 = pygame.rect.Rect(self.bird.x, self.bird.y, e3, 2)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d3)

            e_d4 = pygame.rect.Rect(my_pipe.x-5, e4, my_pipe.width+10, 5)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d4)

            e_d5 = pygame.rect.Rect(my_pipe.x-5, e5, my_pipe.width+10, 5)
            pygame.draw.rect(self.screen, (255, 0, 0), e_d5)

        # Normalization ###
        e1 = e1 / HEIGHT
        e2 = e2 / self.bird.vel_cap             
        e3 = e3 / (WIDTH - 50)
        e4 = e4 / HEIGHT
        e5 = e5 / HEIGHT

        obs = (e1, e2, e3, e4, e5)
        # print(obs)
        return obs

    def get_action_random(self):
        action = np.random.choice((0, 1), 1, p=(0.45, 0.55))
        return action.item(0)

    def get_observation_size(self):
        return OBSERVATION_SIZE

    def get_observation_size_buffer(self):
        return OBSERVATION_SIZE * BUFFER_SIZE

    def get_actions(self):
        return ACTIONS

    def get_action_size(self):
        return ACTION_SIZE

    def game_over(self):
        if not self.printed_score:
            print('Score: {}'.format(self.reward))
            self.printed_score = True

        if self.draw:
            text = self.font.render("Game Over!".format(self.reward), True, (0,0,0))
            self.screen.blit(text,  (320 - text.get_width() // 2, 240 - text.get_height() // 2))
            pygame.display.flip()
            time.sleep(0.4)
        self.is_done = True

    # HUMAN STUFF ################################################

    def run_human_game(self):

        while not self.is_done:

            while not self.time_elapsed_since_last_action > self.fps:
                dt = self.clock.tick()
                self.time_elapsed_since_last_action += dt

            self.global_time += 1

            self.screen.fill(BACKGROUND)
            self.handle_events_human()

            current_reward = 0.1 

            if self.global_time % NEXT_PIPE == 0:
                self.pipes.append(Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR))

            for pipe in self.pipes:
                pipe.update()
                
                if pipe.off_screen():
                    self.pipes.remove(pipe)

                if pipe.hits(self.bird):
                    self.game_over()
                    current_reward = -1

                if pipe.behind(self.bird):    
                    self.reward += 1
                    current_reward = 1

            self.bird.handle_events_human()
            if self.bird.update():
                self.game_over()
                current_reward = -1

            if self.draw:
                self.screen.fill(BACKGROUND)
                for pipe in self.pipes:
                    pipe.draw()
                self.bird.draw()

            obs = self.get_observation_space()
            
            if self.draw:
                pygame.display.update()

            self.time_elapsed_since_last_action = 0
            print(current_reward)

    def handle_events_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_done = False
                pygame.quit()
