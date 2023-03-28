import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn_agent import DuelingDQNAgent
from myenv import MyEnv
import os
import random
from datetime import datetime
from line_profiler_pycharm import profile
import pickle
from utils import *

from configs import invalid_vs_rl as conf

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
seed = C.get('model_name')
load = C.get('load')
opponent = C.get('opponent')
old_seed = ""
if load == True:
    old_seed = C.get('model_to_load')

# seed = random.randint(10000,99999)


def update_invalid_move_types(message, types):
    if message == "moved more than 1 level higher":
        types[0] += 1
    elif message == "build on dome":
        types[1] += 1
    elif message == "moved to dome":
        types[2] += 1
    elif message == "build on occupied tile":
        types[3] += 1
    elif message == "moved to occupied tile":
        types[4] += 1
    elif message == "build outside board":
        types[5] += 1
    elif message == "moved outside board":
        types[6] += 1
    return types


def update_invalid_moves_over_time(total, recent):
    for i in range(len(recent)):
        total[i].append(recent[i])
    return total

def log_move(info, logfile, env):
    action_log = "------------player: " + info.get("player") + " move: " + info.get(
        "move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
    logfile.write(action_log)
    logfile.write(env.render())

# @profile
def main():
    setup_output_files_directories(seed)
    logfile_name = "logs/" + str(seed) + "_train"
    with open(logfile_name, 'w') as logfile:
        env = MyEnv(mode, seed, opponent, checkpoint_every)
        env.reset()
        best_score = -np.inf

        agent = DuelingDQNAgent(gamma=gamma, epsilon=epsilon, lr=lr,
                                input_dims=env.observation_space.shape,
                                n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                                batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                                learn_amount=learn_amount, seed=seed, checkpoint_dir='models/',
                                invalid_moves_enabled=invalid_moves_enabled)

        figure_file = 'plots/' + str(seed) + "/"

        if load:
            agent.load_models(old_seed)

        checkpoint_steps = 0
        checkpoint_score = 0
        total_wins = 0
        checkpoint_wins = 0
        ps = PlotItemStorage()
        previous_train_offset = 0
        if load:
            ps = ps_load(old_seed)
            previous_train_offset = ps.episodes_num_array[-1]

        invalid_move_types = [0, 0, 0, 0, 0, 0, 0]
        last_message = ""

        start_time = datetime.now()
        last_timestamp = datetime.now()
        start_log = "start: " + " seed: " + str(seed) + " timestamp: " + str(start_time) + " mode: " + str(mode) + "\n"
        logfile.write(start_log)
        print(start_log)

        for i in range(1, n_episodes + 1):
            done = False
            observation = env.reset()

            if i % checkpoint_every == 0:
                logfile.write("\nstart episode " + str(i) + " of " + str(n_episodes) + "\n")
                logfile.write(env.render())

            episode_steps = 0
            episode_score = -np.inf

            if random.randrange(100) < 50:      #black starts with 50% chance
                observation_, reward, done, info = env.secondary_player_step()
                if i % checkpoint_every == 0:
                    log_move(info, logfile, env)
                observation = observation_


            while not done:
                action = agent.choose_action(observation, env)
                observation_, reward, done, info = env.primary_player_step(action)
                if i % checkpoint_every == 0:
                    log_move(info,logfile,env)
                if done:
                    agent.store_transition(observation, action, reward, observation_, done)

                else:
                    observation_, reward, done, info = env.secondary_player_step()
                    if i % checkpoint_every == 0:
                        log_move(info, logfile, env)
                    if done:
                        if env.bot_name != "NONE":
                            agent.store_transition(observation, action, reward, observation_, done)

                    else:
                        reward = env.calculate_reward()
                        agent.store_transition(observation, action, reward, observation_, done)

                    if i % checkpoint_every == 0:
                        logfile.write("primary player reward: " + str(reward) + "\n")

                last_message = info.get('message')
                # episode_score += reward

                if i % learn_frequency == 0:
                    agent.learn()
                observation = observation_
                episode_steps += 1

                # if i % checkpoint_every == 0:
                    # action_log = "------------player: " + info.get("player") + " move: " + info.get(
                    #     "move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
                    # logfile.write(action_log)
                    # logfile.write(env.render())
                    # if info_:
                    #     action_log = "------------player: " + info_.get("player") + " move: " + info_.get(
                    #         "move") + " which is: " + info_.get("valid") + ": " + info_.get("message") + "\n"
                    #     logfile.write(action_log)
                    #     logfile.write(env.render())
                    # logfile.write("primary player reward: " + str(reward) + "\n")

            # if episode_score > best_score:
                # best_score = episode_score
            if reward > episode_score:
                episode_score = reward

            if episode_score > best_score:
                best_score = episode_score

            if reward == env.reward_for_win:
                checkpoint_wins += 1
                total_wins += 1

            checkpoint_steps += episode_steps
            checkpoint_score += episode_score

            invalid_move_types = update_invalid_move_types(last_message, invalid_move_types)

            if i % checkpoint_every == 0:
                agent.save_models()

                average_score = checkpoint_score / checkpoint_every
                average_steps = checkpoint_steps / checkpoint_every
                elapsed_time = datetime.now() - last_timestamp
                last_timestamp = datetime.now()

                logfile.write("\nend episode " + str(i) + " of " + str(n_episodes) + " steps: " + str(
                    episode_steps) + " score: " + str(episode_score) + "\n")
                episode_log = "checkpoint: " + str(i) + "/" + str(n_episodes) + " best score: " + str(
                    best_score) + " average score: " + "{:.2f}".format(average_score) + " epsilon " + "{:.2f}".format(
                    agent.epsilon) + "\n" + " average steps: " + str(average_steps) + " elapsed time: " + str(
                    elapsed_time) + " wins this checkpoint: " + str(checkpoint_wins) + " total wins: " + str(
                    total_wins) + "\n"
                logfile.write(episode_log)
                print(episode_log)

                ps.checkpoint_scores_array.append(average_score)
                ps.checkpoint_steps_array.append(average_steps)
                ps.checkpoint_epsilon_array.append(agent.epsilon)
                ps.episodes_num_array.append(i+previous_train_offset)
                ps.checkpoint_wins_array.append(checkpoint_wins)

                ps.invalid_moves_over_time = update_invalid_moves_over_time(ps.invalid_moves_over_time, invalid_move_types)
                plot_learning_curve(ps.episodes_num_array, ps.checkpoint_scores_array, ps.checkpoint_epsilon_array,
                                    ps.checkpoint_steps_array, ps.invalid_moves_over_time, ps.checkpoint_wins_array, figure_file)
                ps_store(ps, seed)
                checkpoint_steps = 0
                checkpoint_score = 0
                checkpoint_wins = 0
                invalid_move_types = [0, 0, 0, 0, 0, 0, 0]

        end_timestamp = datetime.now()
        end_elapsed_time = datetime.now() - start_time
        end_log = "end: " + " seed: " + str(seed) + " timestamp " + str(end_timestamp) + " , training time: " + str(
            end_elapsed_time)
        logfile.write(end_log)
        print(end_log)


main()
