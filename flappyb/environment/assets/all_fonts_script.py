import pygame
import time

pygame.init()
pygame.display.set_caption('display all fonts')
screen = pygame.display.set_mode((1800, 1200))

y = 150
x = 0
mod = 0

for font in pygame.font.get_fonts():
	if font == 'notocoloremoji':
		continue
	if font == 'kacstoffice':
		break

	if mod == 0:
		x = 200
	if mod == 1:
		x = 600
	if mod == 2:
		x = 1100
	if mod == 3:
		x = 1600
		y+=25
		mod = -1
	mod += 1

	text = pygame.font.SysFont(font, 17).render("{}: GAME 123 !".format(font), True, (255,255,255))
	screen.blit(text,  (x - text.get_width() // 2, y - text.get_height() // 2))




pygame.display.flip()
time.sleep(200)
