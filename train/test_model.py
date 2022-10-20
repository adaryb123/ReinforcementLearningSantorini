from stable_baselines3 import PPO, DQN
import os
from myenv import MyEnv
import time
from train_model import models_dir

env = MyEnv()
env.reset()

model = PPO.load(models_dir + ".zip", env=env)
# model = DQN.load(models_dir + ".zip", env=env)


obs = env.reset()
i = 0
invalid_count = 0

env.render()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    if reward == -10:
        invalid_count += 1
        print("moron played an invalid move")
    elif done == 1:
        print("victory in " + str(i) + " turns")
        print("and he did " + str(invalid_count) + " invalid moves")
        break

    env.render()

env.render()