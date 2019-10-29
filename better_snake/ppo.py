import gym
import gym_snake

from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1


if __name__ == "__main__":
	# env = gym.make('snake-v0')
	env = gym.make('snake-v0')
	# env = SubprocVecEnv([lambda: env])
	env = DummyVecEnv([lambda: env])

	model = PPO1(MlpPolicy, env, verbose=1)

	model.learn(total_timesteps=500000)
	model.save('models/snake-bastard')

	###############################################################################

	# env = gym.make('snake-v0')
	# # env = DummyVecEnv([lambda: env])

	# # model = PPO2(MlpPolicy, env, verbose=1)
	# # model.load('models/snake-basterd')

	# obs = env.reset()
	# is_done = False

	# while not is_done:
	#     action, _states = model.predict(obs)
	#     obs, rewards, terminal, info = env.step(action)
	#     is_done = terminal
	#     env.render()
