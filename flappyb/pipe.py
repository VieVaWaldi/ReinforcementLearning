import random
import pygame


class Pipe:

    def __init__(self, screen, s_width, s_height, color):

        self.screen = screen
        self.s_width = s_width
        self.s_height = s_height
        self.color = color

        self.top = random.randrange(0, s_height/2)
        self.bot = random.randrange(s_height/2, s_height)
        self.width = 40
        self.x = s_width
        self.speed = 3

        self.within_pipe = False

    def draw(self):
        rect_top = pygame.rect.Rect(self.x, 0, self.width, self.top)
        rect_bot = pygame.rect.Rect(self.x, self.bot, self.width, self.s_height)
        pygame.draw.rect(self.screen, self.color, rect_top)
        pygame.draw.rect(self.screen, self.color, rect_bot)

    def update(self):
        self.x -= self.speed

    def hits(self, bird):
        if bird.y < self.top or bird.y > self.s_height - self.bot:
            if self.x < bird.x < self.x + self.width:
                return True

    def behind(self, bird):
        if bird.x > self.x + self.width and not self.within_pipe:
            self.within_pipe = True
            return True
        if not (bird.x > self.x + self.width):
            self.within_pipe = False

    def off_screen(self):
        return self.x + self.width + 5 < 0
