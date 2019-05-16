import pygame


class Bird:

	def __init__(self, screen, s_width, s_height, color):

		self.screen = screen
		self.s_width = s_width
		self.s_height = s_height
		self.color = color

		self.radius = 20
		self.x = 50
		self.y = int(s_height / 2)

		self.vel = 0
		self.gravity = 1

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
		pygame.draw.circle(self.screen, self.color, (self.x, self.y), self.radius)

	def update(self):
		self.vel += self.gravity
		self.y += self.vel

		if self.y > self.s_height:
			self.y = self.s_height
			self.vel = 0

		if self.y < 0:
			self.y = 0
			self.vel = 0

	def _fly(self):
		self.vel += -self.gravity*4

