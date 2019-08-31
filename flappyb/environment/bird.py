import pygame


class Bird():

    def __init__(self, screen, s_width, s_height, color):

        self.bird_image = pygame.image.load("environment/assets/bird.png")
        self.rotate = 1

        self.screen = screen
        self.s_width = s_width
        self.s_height = s_height
        self.color = color

        self.radius = 20
        # self.radius = 10

        self.x = 50
        self.y = int(s_height / 2)

        self.vel = 0
        self.gravity = 2  # default is 1

        self.bottom = s_height - 20
        self.vel_cap = 20

        self.salto = False
        self.rotation = 0
        self.last_rotation = 0
        self.last_reward = 0

    def handle_events_human(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            self._fly()

    def handle_events_ai(self, action):
        if action == 1:
            self._fly()
        else:
            pass

    def draw(self, reward):

        surf = None

        if reward % 10 == 0 and reward is not self.last_reward:
            self.last_reward = reward
            self.salto = True

        if self.salto:
            if self.last_rotation >= 0:
                self.rotation += 15
                surf = pygame.transform.rotate(self.bird_image, self.rotation)
                if self.rotation == 400:
                    self.salto = False
            else:
                self.rotation -= 15
                surf = pygame.transform.rotate(self.bird_image, self.rotation)
                if self.rotation == -400:
                    self.salto = False

        elif self.vel > 0:
            self.rotation = -40
            self.last_rotation = self.rotation
            surf = pygame.transform.rotate(self.bird_image, self.rotation)

        else:
            self.rotation = 40
            self.last_rotation = self.rotation
            surf = pygame.transform.rotate(self.bird_image, self.rotation)

        self.screen.blit(surf, (self.x - 25, self.y - 20))

    def update(self):
        self.vel += self.gravity
        self.y += self.vel

        if self.y > self.s_height:
            return True

        if self.y < 0:
            self.y = 0

        if self.vel > 20:
            self.vel = 20

        return False

    def _fly(self):
        self.vel += -self.gravity * 2

        if self.vel > 20:
            self.vel = 20
