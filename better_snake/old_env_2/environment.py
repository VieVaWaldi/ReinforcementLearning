from environment.cube import Cube
from environment.snake import Snake

import gym
import pygame

import numpy as np
import random
import enum
import time

#import tkinker as tk
#from tkinter import messagebox

# snake obs
# body = head 0.9, b[0] = 0.8, b[1] = 0.79 ...


W = 500
H = 500
BUFFER_SIZE = 1


class Actions(enum.Enum):
    Up = 0
    Right = 1
    Down = 2
    Left = 3


class SnakeEnvironment(gym.Env):

    def __init__(self, draw=True, speed=10000, rows=20, animation=True):
        super(SnakeEnvironment, self).__init__()


        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(rows, rows), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(n=len(Actions))

        self.draw = draw
        self.speed = speed
        self.rows = rows
        self.animation = animation

        self.snake = Snake((255, 0, 0), (2, 2), self.rows, W)
        self.snack = Cube(self.random_snack(), self.rows, W, color=(0, 255, 0))

        self.is_done = False
        self.reward = 0
        self.step_without_apple = 0

        self.surf = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        if draw:
            pygame.init()
            self.font_game_over = pygame.font.SysFont("ani", 72)

    """ Must alwasy be calles in the beginning. """
    def reset(self):
        self.countdown()

        self.snake.reset((2, 2))
        self.snack = Cube(self.random_snack(), self.rows, W, color=(0, 255, 0))
        self.is_done = False
        self.reward = 0
        self.step_without_apple = 0

        self.surf = pygame.display.set_mode((W, H))
        self.clock = pygame.time.Clock()

        obs, reward, is_done, _ = self.step(1)

        return obs

    def step(self, action):
        pygame.time.delay(50)                      # lower is faster
        self.clock.tick(self.speed)                # lower is slower

        current_reward = 0

        self.snake.move_ai(action)
        # self.snake.move_human()

        if self.snake.ate_itself():
            current_reward = -1
            self.game_over()

        self.step_without_apple += 1
        if self.step_without_apple == 250:
            self.game_over()

        if self.snake.body[0].pos == self.snack.pos:
            self.snake.add_cube()
            self.snack = Cube(self.random_snack(), self.rows, W, color=(0, 255, 0))
            self.reward += 1
            current_reward = 1
            self.step_without_apple = 0

        self.redraw_window()

        obs = self.get_observation_space()

        return obs, current_reward, self.is_done, {}

    def get_observation_space(self):

        new_obs = []

        # create 2d matrix
        for i in range(self.rows):
            new_obs.append([])
            for j in range(self.rows):
                new_obs[i].append(-1)

        # add apple
        x_apple = self.snack.pos[0]
        y_apple = self.snack.pos[1]
        new_obs[y_apple][x_apple] = 1

        # tail
        for i, c in enumerate(self.snake.body):
            x_snake = c.pos[0]
            y_snake = c.pos[1]

            if x_snake == -1 or x_snake == self.rows:
                print('Wtf, this error occured!')
                self.game_over()
                return
            if y_snake == -1 or y_snake == self.rows:
                print('Wtf, this error occured!')
                self.game_over()
                return

            new_obs[y_snake][x_snake] = 0.5

        # add snake head
        x_snake = self.snake.head.pos[0]
        y_snake = self.snake.head.pos[1]
        if x_snake == -1 or x_snake == self.rows:
            print('Wtf, this error occured!')
            self.game_over()
            return
        if y_snake == -1 or y_snake == self.rows:
            print('Wtf, this error occured!')
            self.game_over()
            return
        new_obs[y_snake][x_snake] = 0.8

        # current_obs = []
        # for i in new_obs:
        #     for j in i:
        #         current_obs.append(j)

        # cnt = 0
        # for i in current_obs:
        #     cnt += 1
        #     print(' ', i, ' ', end='')
        #     if cnt % self.rows == 0:
        #         print('')
        # print('')

        # return_obs = np.array(current_obs)

        # print(new_obs)

        # time.sleep(10)

        return new_obs

    def draw_grid(self):
        size_btwn = W // self.rows

        x = 0
        y = 0

        for i in range(self.rows):
            x = x + size_btwn
            y = y + size_btwn

            pygame.draw.line(self.surf, (255, 255, 255), (x, 0), (x, W))
            pygame.draw.line(self.surf, (255, 255, 255), (0, y), (W, y))

    def redraw_window(self):
        if not self.draw:
            return

        self.surf.fill((0, 0, 0))
        self.draw_grid()
        self.snake.draw(self.surf)
        self.snack.draw(self.surf)

        pygame.display.update()

    def random_snack(self):
        positions = self.snake.body
     
        while True:
            x = random.randrange(self.rows)
            y = random.randrange(self.rows)
            if len(list(filter(lambda z:z.pos == (x,y), positions))) > 0:
                continue
            else:
                break
        return (x,y)

    def countdown(self):
        if not self.draw or not self.animation:
            return
        for _ in range(3, 0, -1):
            self.write_text("Start in {}".format(_))
            time.sleep(0.3)

    def game_over(self):
        self.is_done = True
        if not self.draw or not self.animation:
            return
        self.write_text("Score {}".format(self.reward))
        time.sleep(1.5)

    def write_text(self, text):
        self.redraw_window()
        text_start = pygame.font.SysFont('dyuthi', 80). \
            render(text, True, (255, 255, 255))
        self.surf.blit(text_start,
                         (text_start.get_width() //
                          2, text_start.get_height() // 2))
        pygame.display.flip()

    def play_human(self):
        self.countdown()

        while(not self.is_done):
            pygame.time.delay(50)                      # lower is faster
            self.clock.tick(self.speed)                # lower is slower

            self.snake.move_human()

            if self.snake.ate_itself():
                self.game_over()

            if self.snake.body[0].pos == self.snack.pos:
                self.snake.add_cube()
                self.snack = Cube(self.random_snack(), self.rows, W, color=(0, 255, 0))
                self.reward += 1

            self.redraw_window()
            self.get_observation_space()


if __name__ == "__main__":
    env = SnakeEnvironment(draw=True, speed=100, rows=5)
    env.play_human()




        #######
        # if self.last_observation == None:
        #     self.last_observation = current_obs

        # return_obs = []

        # for i in self.last_observation:
        #     return_obs.append(i)
        # for i in current_obs:
        #     return_obs.append(i)

        # return_obs = np.array(return_obs)

        # cnt = 0
        # for i in return_obs:
        #     cnt += 1
        #     print(' ', i, ' ', end='')
        #     if cnt % 10 == 0: 
        #         print('')
        #     if cnt % 100 == 0:
        #         print('')
        #         print('')
        # print('')

        # self.last_observation = current_obs
        #######
