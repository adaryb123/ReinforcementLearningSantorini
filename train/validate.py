"""
Script for valid Double deep Q learning agent in 100 games against all basic bots
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

from dqn_agent import DQNAgent
from environment import Environment
from configs import invalid_vs_RL_softmax as conf
import random
import os

C = conf.config
n_episodes = C.get('n_episodes')
epsilon = C.get('epsilon')
eps_min = C.get('eps_min')
checkpoint_every = C.get('checkpoint_every')
learn_frequency = C.get('learn_frequency')
learn_amount = C.get('learn_amount')
mode = C.get('mode')
gamma = C.get('gamma')
lr = C.get('learning_rate')
mem_size = C.get('memory_size')
batch_size = C.get('batch_size')
replace = C.get('replace_network_frequency')
eps_dec = C.get('eps_dec')
invalid_moves_enabled = C.get('invalid_moves_enabled')
# opponent = C.get('opponent')
network = C.get('network')
canals = C.get('canals')
epsilon_softmax = C.get('epsilon_softmax')
seed = C.get('model_name')

def log_move(info, logfile, env):
    action_log = "------------player: " + info.get("player") + " move: " + info.get(
        "move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
    logfile.write(action_log)
    logfile.write(env.render())

def main():

        env = Environment(mode, seed, 'NONE', checkpoint_every, canals)

        agent = DQNAgent(gamma=gamma, epsilon=0, lr=lr,
                         input_dims=env.observation_space.shape,
                         n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                         batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                         learn_amount=learn_amount, seed=seed, checkpoint_dir='models/',
                         invalid_moves_enabled=False, network=network,
                         epsilon_softmax=epsilon_softmax)

        agent.load_models(seed)

        # opponents = ['NONE','RANDOM','HEURISTIC','HEURISTIC-COMPETITIVE','MINMAX','RL']
        opponents = ['RL']

        log_dir = "logs/" + str(seed)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        for i in range(len(opponents)):

            opponent = opponents[i]
            env.set_secondary_player(opponent)

            reward_for_win = env.reward_for_win
            wins = 0

            logfile_name = log_dir + "/" + opponent
            with open(logfile_name, 'w') as logfile:
                for j in range(100):

                    done = False
                    observation = env.reset()
                    if random.randrange(100) < 50:      #black starts with 50% chance
                        observation_, reward, done, info = env.secondary_player_step()
                        log_move(info, logfile, env)
                        observation = observation_

                    while not done:

                        action = agent.choose_action(observation, env)
                        observation_, reward, done, info = env.primary_player_step(action)
                        log_move(info, logfile, env)

                        if reward == reward_for_win:
                            wins += 1
                        if not done:
                            observation_, reward, done, info = env.secondary_player_step()
                            log_move(info, logfile, env)

                print("opponent: " + str(opponent))
                print("winrate: " + str(wins) + "/100")

main()