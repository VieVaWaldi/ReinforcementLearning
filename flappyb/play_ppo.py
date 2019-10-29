import numpy as np
import torch
import ptan

from lib import ppo_model as model
from environment.environment import Environment

# MODEL_NAME = "flappyb-test-the-rainbow350"
MODEL_NAME = "best_+131.310_576000.dat"

env = env = Environment(draw=True, fps=10, debug=False,
                      dist_to_pipe=50, dist_between_pipes=180, obs_this_pipe=True)

net_act = model.ModelActor(env.observation_space.n,
                           env.action_space.n).to("cpu")
net_act.load_state_dict(torch.load("saves/ppo-test-flappyb/" + MODEL_NAME, map_location=lambda storage, loc: storage))    

rewards = 0.0
steps = 0
for _ in range(5):
    obs = env.reset()
    while True:
        obs_v = ptan.agent.float32_preprocessor([obs]).to("cpu")
        mu_v = net_act(obs_v)[0]
        action = mu_v.squeeze(dim=0).data.cpu().numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        rewards += reward
        steps += 1
        if done:
            break
