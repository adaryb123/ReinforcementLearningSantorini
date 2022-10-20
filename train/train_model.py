from stable_baselines3 import PPO, DQN
import os
from myenv import MyEnv
import time

# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"

models_dir = "models/PPO-2"
logdir = "logs/PPO-2"



if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = MyEnv()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# model = PPO.load(models_dir + ".zip", env=env)


TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	print("---------------------------------------------------------------" + str(iters))
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=100, tb_log_name=f"DQN")
	# model.save(f"{models_dir}/{TIMESTEPS*iters}")
	model.save(models_dir)
