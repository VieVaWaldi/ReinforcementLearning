# Game was made with the help of https://www.youtube.com/watch?v=cXgA1d_E-jY
import time
import pygame
import random
import enum

import numpy as np
import gym

from environment.snake import Snake
from environment.apple import Apple

# AI PARAMETERS ###############################################################
BUFFER_SIZE = 1
OBSERVATION_SIZE = 5 * 5
ACTIONS = [0, 1, 2, 3]
ACTION_SIZE = 4

# GAME PARAMETERS #############################################################
SCALE = 60
SCREEN_SIZE = WIDTH, HEIGHT = (300, 300)    # for 5*5 go 300*300, 60
                                            # for 10*10 go 600*600, 60

BACKGROUND = (72, 72, 72)
SNAKE_COLOR = (57, 255, 20)
APPPLE_COLOR = (255, 8, 0)
FONT = 'dyuthi'

""" Rewards
    1. first apple +1
    2. every next apple n+1
    3. hit wall -1
    4. ate self -2
    5. does nothing 0.1
"""
""" Observations
    1. apple +1
    3. snake head = 0.5
    4. every snake body -0.01
    5. emtpy cell = -1
"""
"""
Interace:
reset():                resets the whole environment
step(action):           performs one action onto the environment
step_buffer(action):    performs one action onto the environment,
                        returns 4 states for experience replay
get_action_random():    obtain an imporoved random action
get_observation_size(): obtain size of observation
get_action_size():      obtain size of action
"""


class Actions(enum.Enum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3


class SnakeEnvironment(gym.Env):

    def __init__(self, draw=True, fps=10, debug=False):

        super(SnakeEnvironment, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=OBSERVATION_SIZE * BUFFER_SIZE)
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        if draw:
            pygame.init()
            pygame.display.set_caption('NN Snake')
            self.font_game_over = pygame.font.SysFont("ani", 72)

        self.fps = fps
        self.debug = debug
        self.draw = draw

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.screen = pygame.display.set_mode(SCREEN_SIZE)

        self.snake = Snake(self.screen, WIDTH, HEIGHT, SNAKE_COLOR,
                           BACKGROUND, SCALE)
        self.apple = Apple(self.screen, WIDTH, HEIGHT, APPPLE_COLOR, SCALE)

        self.reward = 0
        self.score = 0
        self.is_done = False
        self.steps_without_apple = 0

        # self.current_observation = None
        # self.last_observation = None

    # ML INTERFACE ############################################################
    def reset(self):
        """ Resets the whole environment. Must be called in the beginning. """

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        self.reward = 0
        self.is_done = False
        self.score = 0
        self.steps_without_apple = 0

        self.snake = Snake(self.screen, WIDTH, HEIGHT, SNAKE_COLOR,
                           BACKGROUND, SCALE)
        self.apple = Apple(self.screen, WIDTH, HEIGHT, APPPLE_COLOR, SCALE)

        obs, reward, is_done, _ = self.step(0)

        # if self.draw:
        #     self.countdown()

        return obs

    def step(self, action):
        """ Performs one step on the env and returns one observation state. """

        while not self.time_elapsed_since_last_action > self.fps:
            dt = self.clock.tick()
            self.time_elapsed_since_last_action += dt

        self.global_time += 1

        return self.run_ai_game_step(action)

    # The actual game step ####################################################
    def run_ai_game_step(self, action):

        current_reward = 0.1

        self.snake.handle_events_ai(action)

        if self.apple.eat(self.snake.x, self.snake.y, self.snake.tail):
            self.snake.update(True)
            self.steps_without_apple = 0
            self.score += 1
            current_reward = self.score
        else:
            self.snake.update(False)
            self.steps_without_apple += 1
            if self.steps_without_apple > 15:
                current_reward = -1
            else:
                current_reward = 0.1

        if self.draw:
            self.screen.fill(BACKGROUND)
            self.snake.draw()
            self.apple.draw()
            pygame.display.update()

        if self.snake.check_if_hit_wall():
            current_reward = -1
            self.game_over()

        if self.snake.check_if_ate_self():
            current_reward = -1
            self.game_over()

        self.time_elapsed_since_last_action = 0

        obs = self.get_observation_space()

        return obs, current_reward, self.is_done, None

    def get_observation_space(self):

        new_obs = []

        # create 2d matrix
        for i in range(int(WIDTH / SCALE)):
            new_obs.append([])
            for j in range(int(WIDTH / SCALE)):
                new_obs[i].append(-1)

        # add apple
        x_apple = int(self.apple.x / SCALE)
        y_apple = int(self.apple.y / SCALE)
        new_obs[y_apple][x_apple] = 1

        # add snake
        x_snake = int(self.snake.x / SCALE)
        y_snake = int(self.snake.y / SCALE)
        new_obs[y_snake][x_snake] = 0.5

        for i in self.snake.tail:
            x_snake = int(i.x / SCALE)
            y_snake = int(i.y / SCALE)
            new_obs[y_snake][x_snake] = 0.3

        if self.draw and self.debug:
            for i in new_obs:
                print(i, '\n')
            print('\n')

        current_obs = []
        for i in new_obs:
            for j in i:
                current_obs.append(j)

        # if self.last_observation == None:
        #     self.current_observation = current_obs

        # self.last_observation = self.current_observation
        # self.current_observation = current_obs

        # return_obs = []

        # print(self.last_observation)
        # print('\n')
        # print(self.current_observation)

        # for i in self.last_observation:
        #     return_obs.append(i)

        # for i in self.current_observation:
        #     return_obs.append(i)

        current_obs = np.array(current_obs)

        return current_obs

    def get_action_random(self):
        return random.randint(0, 3)

    # HUMAN STUFF ############################################################

    def run_human_game(self):

        if self.draw:
            self.countdown()

        while not self.is_done:

            while not self.time_elapsed_since_last_action > self.fps:
                dt = self.clock.tick()
                self.time_elapsed_since_last_action += dt

            self.global_time += 1

            self.handle_events_human()
            self.snake.handle_events_human()

            if self.apple.eat(self.snake.x, self.snake.y, self.snake.tail):
                self.snake.update(True)
                self.score += 1
            else:
                self.snake.update(False)

            if self.draw:
                self.screen.fill(BACKGROUND)
                self.snake.draw()
                self.apple.draw()
                pygame.display.update()

            if self.snake.check_if_hit_wall():
                self.game_over()

            if self.snake.check_if_ate_self():
                self.game_over()

            self.time_elapsed_since_last_action = 0

            self.get_observation_space()

    def handle_events_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_done = False
                pygame.quit()

    def countdown(self):
        for _ in range(3, 0, -1):
            self.screen.fill(BACKGROUND)
            self.snake.draw()
            text_start = pygame.font.SysFont(FONT, 80). \
                render("Start in  {}".format(_), True, (0, 0, 0))
            self.screen.blit(text_start,
                             (text_start.get_width() //
                              2, text_start.get_height() // 2))
            pygame.display.flip()
            # time.sleep(0.3)

    def game_over(self):
        if self.draw:
            text = pygame.font.SysFont(FONT, 28).render(
                "Game Over!".format(self.reward), True, (0, 0, 0))
            self.screen.blit(text, (320 - text.get_width() //
                                    2, 240 - text.get_height() // 2))
            pygame.display.flip()
            # time.sleep(0.4)
        self.is_done = True
