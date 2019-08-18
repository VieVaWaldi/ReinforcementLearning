# Game was made with the help of https://www.youtube.com/watch?v=cXgA1d_E-jY
import time
import pygame

import numpy as np

from snake import Snake
from apple import Apple

# AI PARAMETERS #####################################################################################
BUFFER_SIZE = 2
OBSERVATION_SIZE = 10*10 
ACTIONS = [0, 1, 2, 3]
ACTION_SIZE = 4

# GAME PARAMETERS ###################################################################################
SCALE = 60
SCREEN_SIZE = WIDTH, HEIGHT = (600, 600)

BACKGROUND = (72, 72, 72)
SNAKE_COLOR = (57, 255, 20)
APPPLE_COLOR = (255, 8, 0)
FONT = 'dyuthi'

""" Rewards
    1. first apple +1
    2. every next apple n+1
    3. hit wall -1
    4. ate self -2
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
step_buffer(action):    performs one action onto the environment, returns 4 states for experience replay
get_action_random():    obtain an imporoved random action
get_observation_size(): obtain size of observation
get_action_size():      obtain size of action
"""


class SnakeEnvironment:

    def __init__(self, draw=True, fps=10, debug=False):
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

        self.snake = Snake(self.screen, WIDTH, HEIGHT, SNAKE_COLOR, BACKGROUND, SCALE)
        self.apple = Apple(self.screen, WIDTH, HEIGHT, APPPLE_COLOR, SCALE)

        self.reward = 0
        self.is_done = False
        self.printed_score = False

    # ML INTERFACE ##################################################################################
    def reset(self):

        self.clock = pygame.time.Clock()
        self.time_elapsed_since_last_action = 0
        self.global_time = 0

        # self.bird = Bird(self.screen, WIDTH, HEIGHT, BIRD_COLOR)
        # self.pipes = [Pipe(self.screen, WIDTH, HEIGHT, PIPE_COLOR, self.pipe_image, self.pipe_long_image)]

        self.reward = 0
        self.is_done = False
        self.printed_score = False

        # lol no premium action, why did i write that ?
        obs, reward, is_done, _ = self.step_buffer(0)

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

        # snake logic
        self.snake.handle_events_human()
        self.snake.update()

        if self.draw:
            self.screen.fill(BACKGROUND)

            self.snake.draw()

            text = pygame.font.SysFont(FONT, 28).render(
                "SCORE {}".format(self.reward), True, (0, 0, 0))
            self.screen.blit(text, (565 - text.get_width() //
                                    2, 30 - text.get_height() // 2))
            pygame.display.flip()

        obs = self.get_observation_space()

        if self.draw:
            pygame.display.update()

        self.time_elapsed_since_last_action = 0

        return obs, current_reward, self.is_done, None
    #################################################################################################

    def get_observation_space(self):

        obs = None

        return obs

    def get_action_random(self):
        action = np.random.choice((0, 1, 2, 3), 1, p=(0.25, 0.25, 0.25, 0.25))
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
        # if not self.printed_score:
        #     print('Score: {}'.format(self.reward))
        #     self.printed_score = True

        if self.draw:
            text = pygame.font.SysFont(FONT, 28).render(
                "Game Over!".format(self.reward), True, (0, 0, 0))
            self.screen.blit(text, (320 - text.get_width() //
                                    2, 240 - text.get_height() // 2))
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

            self.handle_events_human()
            self.snake.handle_events_human()

            if self.apple.eat(self.snake.x, self.snake.y):
                self.snake.update(True)
            else:
                self.snake.update(False)

            if self.draw:
                self.screen.fill(BACKGROUND)
                self.snake.draw()
                self.apple.draw()
                pygame.display.update()

            if self.snake.check_if_dead():
                self.game_over()

            self.time_elapsed_since_last_action = 0

    def handle_events_human(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_done = False
                pygame.quit()
