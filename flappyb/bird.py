import pygame
import random

class Bird():

	def __init__(self, screen, s_width, s_height, color):

		self.bird_image = pygame.image.load("flappyb/assets/bird.png") # 50x35
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
		self.gravity = 1

		self.bottom = s_height-20
		self.vel_cap = 20

	def handle_events_human(self):
		keys = pygame.key.get_pressed()
		if keys[pygame.K_SPACE]:
			self._fly()

	def handle_events_ai(self, action):
		if action == 1:
			self._fly()
		else:
			pass

	def draw(self):
		# pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius)

		# bottom = pygame.rect.Rect(0, self.bottom, self.s_width, 20)
		# pygame.draw.rect(self.screen, (120, 72, 0), bottom)

		surf = None

		if self.vel > 0:
			surf = pygame.transform.rotate(self.bird_image, -40)
		else: 
			surf = pygame.transform.rotate(self.bird_image, 40)

		# self.rotate += 10
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
		self.vel += -self.gravity*2
		
		if self.vel > 20:
			self.vel = 20


