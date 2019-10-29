# from environment.environment import SnakeEnvironment

env = SnakeEnvironment(draw=True, speed=100000, rows=5)

env.reset()
terminal = False

while not terminal:
    action = random.randint(0, 4)
    next_state, reward, is_done, _ = env.step(action)
    terminal = is_done
