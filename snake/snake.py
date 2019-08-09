# https://www.youtube.com/watch?v=AaGK-fj-BAM&t=630s
import pygame
import random


class Snake:

    def __init__(self, screen, s_width, s_height, color, scale):

        self.screen = screen
        self.s_width = s_width
        self.s_height = s_height
        self.color = color
        self.scale = scale

        self.x = 5 * scale
        self.y = 5 * scale

        self.x_speed = 1
        self.y_speed = 0

        self.total = 1
        self.tail = []

    def handle_events_human(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            self.move(0, -1)
        if keys[pygame.K_RIGHT]:
            self.move(1, 0)
        if keys[pygame.K_DOWN]:
            self.move(0, 1)
        if keys[pygame.K_LEFT]:
            self.move(-1, 0)

    def handle_events_ai(self, action):
        if action == 1:
            pass
        else:
            pass

    def draw(self):
        for i in range(self.total):
            rect = pygame.rect.Rect(
                self.tail[i].x, self.tail[i].y, self.scale, self.scale)
            pygame.draw.rect(self.screen, self.color, rect)

    def update(self):
        for i in range(self.total):
            self.tail[i] = self.tail[i + 1]
        self.tail[self.total - 1] = Vector(self.x, self.y)

        self.x = self.x + self.x_speed * self.scale
        self.y = self.y + self.y_speed * self.scale

        if self.x < 0:
            self.x = 0
        if self.x > self.s_width - self.scale:
            self.x = self.s_width - self.scale
        if self.y < 0:
            self.y = 0
        if self.y > self.s_height - self.scale:
            self.y = self.s_height - self.scale

    def move(self, x, y):
        self.x_speed = x
        self.y_speed = y


class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y
