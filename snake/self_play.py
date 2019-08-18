from environment.environment import SnakeEnvironment
import random

env = SnakeEnvironment(draw=True, fps=100, debug=True)

# env.run_human_game()

env.reset()
is_done = False
while not is_done:
    action = random.randint(0, 4)
    next_state, reward, terminal, _ = env.step(action)
    print(reward)
    is_done = terminal
