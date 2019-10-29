from environment.environment import SnakeEnvironment
import random

env = SnakeEnvironment(draw=True, speed=100, rows=5, animation=False)

# env.play_human()

while True:
	env.reset()
	terminal = False
	while not terminal:
	    action = random.randint(0, 4)
	    next_state, reward, is_done, _ = env.step(action)
	    terminal = is_done
