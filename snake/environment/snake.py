# https://www.youtube.com/watch?v=AaGK-fj-BAM&t=630s
import pygame


class Snake:

    def __init__(self, screen, s_width, s_height, color, body_color, scale):

        self.screen = screen
        self.s_width = s_width
        self.s_height = s_height
        self.color = color
        self.body_color = body_color
        self.scale = scale

        self.scale = scale

        self.x = 2 * scale
        self.y = 2 * scale

        self.x_speed = 1
        self.y_speed = 0

        self.tail = [Vector(self.x, self.y)]

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
        if action == 0:
            self.move(0, -1)
        if action == 1:
            self.move(1, 0)
        if action == 2:
            self.move(0, 1)
        if action == 3:
            self.move(-1, 0)

    def draw(self):

        for i in self.tail:
            rect = pygame.rect.Rect(
                i.x + 1, i.y + 1, self.scale - 2, self.scale - 2)
            pygame.draw.rect(self.screen, self.color, rect)
            rect = pygame.rect.Rect(
                i.x + 16, i.y + 16, self.scale - 32, self.scale - 32)
            pygame.draw.rect(self.screen, self.body_color, rect)

        rect = pygame.rect.Rect(
            self.x, self.y, self.scale, self.scale)
        pygame.draw.rect(self.screen, self.color, rect)

    def update(self, ate_apple):

        length = len(self.tail)

        if ate_apple:
            self.tail.append(Vector(self.x, self.y))
        else:
            for i in range(length - 1):
                self.tail[i] = self.tail[i + 1]
            self.tail[length - 1] = Vector(self.x, self.y)

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

    def check_if_hit_wall(self):
        if self.x == 0:
            return True
        if self.x == self.s_width - self.scale:
            return True
        if self.y == 0:
            return True
        if self.y == self.s_height - self.scale:
            return True

    def check_if_ate_self(self):
        for i in self.tail:
            if (self.x == i.x) and (self.y == i.y):
                return True


class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y
