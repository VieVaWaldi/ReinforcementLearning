# Game was made with help of https://www.youtube.com/watch?v=cXgA1d_E-jY

import time
import pygame

from flappyb.bird import Bird
from flappyb.pipe import Pipe

import numpy as np

SCREEN_SIZE = WIDTH, HEIGHT = (640, 880)
BACKGROUND = (0, 0, 0)
BIRD_COLOR = (255, 0, 0)
PIPE_COLOR = (0, 255, 0)
NEXT_PIPE = 100
FPS = 1


# AI Parameters
ACTIONS = [0, 1]         # space bar press = True, not press = False
ACTION_SIZE = 2
OBSERVATION_SIZE = 6
ROUND_TO_DECIMALS = 2


class Environment:

    def __init__(self, draw):
        if draw:
            pygame.init()
            pygame.display.set_caption('NN FlappyB')

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

        obs, reward, is_done, _ = self.step(False)      # lol no premium action, why did i write that ?

        return obs

    def step(self, action):

        while not self.time_elapsed_since_last_action > FPS:
            dt = self.clock.tick()
            self.time_elapsed_since_last_action += dt

        self.global_time += 1

        return self.run_ai_game_step(action)

    def get_observation_space(self):

        my_pipe = Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR)
        my_pipe.x = 9999

        for pipe in self.pipes:
            if (pipe.x < my_pipe.x) and pipe.x >= (self.bird.x - pipe.width):
                my_pipe = pipe

        e1 = self.bird.y                    # bird pos
        e2 = self.bird.vel                  # bird vel
        e3 = self.bird.gravity              # bird grav
        e4 = my_pipe.x - self.bird.x        # dist to pipe
        e5 = my_pipe.top                    # pipe bot
        e6 = my_pipe.bot                    # pipe top

        obs = np.array([e1, e2, e3, e4, e5, e6])
        nor_obs = np.interp(obs, (obs.min(), obs.max()), (0, 1))

        # print(obs)
        # print('--------------')

        wow1 = round(float(nor_obs[0]), ROUND_TO_DECIMALS)
        wow2 = round(float(nor_obs[1]), ROUND_TO_DECIMALS)
        wow3 = round(float(nor_obs[2]), ROUND_TO_DECIMALS)
        wow4 = round(float(nor_obs[3]), ROUND_TO_DECIMALS)
        wow5 = round(float(nor_obs[4]), ROUND_TO_DECIMALS)
        wow6 = round(float(nor_obs[5]), ROUND_TO_DECIMALS)

        obs = (wow1, wow2, wow3, wow4, wow5, wow6)

        return obs

    def get_action_random(self):
        return np.random.choice(self.get_actions())

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
                print('Score: {}'.format(self.reward))
                self.reward -= 1
                current_reward = -1
                self.is_done = True
            if pipe.behind(self.bird):
                self.reward += 1
                current_reward = 1

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

    def handle_events_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_done = False
                pygame.quit()
