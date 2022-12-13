import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn_agent import DuelingDQNAgent
from myenv import MyEnv
import os
import random
from datetime import datetime
from line_profiler_pycharm import profile
from configs import coop_then_compet_then_compet as conf

# if training a new model
# seed = random.randint(10000, 99999)
# seed = "200k-compet"
# load = False

# if continuing on an already trained model
# seed = 97620
# load = True


C = conf.config
n_episodes = C.get('n_episodes')
epsilon = C.get('epsilon')
eps_min = C.get('eps_min')
checkpoint_every = C.get('checkpoint_every')
learn_frequency = C.get('learn_frequency')
learn_amount = C.get('learn_amount')
reward_for_win = C.get('reward_for_win')
mode = C.get('mode')
gamma = C.get('gamma')
lr = C.get('learning_rate')
mem_size = C.get('memory_size')
batch_size = C.get('batch_size')
replace = C.get('replace_network_frequency')
eps_dec = C.get('eps_dec')
seed = C.get('model_name')
load = C.get('load')
old_seed = ""
if load == True:
    old_seed = C.get('model_to_load')

def setup_output_files_directories():
    models_dir = "models"
    logs_dir = "logs"
    plots_dir = "plots"

    plots_seed_dir = "plots/" + str(seed)

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    if not os.path.exists(plots_seed_dir):
        os.makedirs(plots_seed_dir)


def plot_learning_curve(x, scores, epsilons, steps, invalid_moves, wins, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    plt.savefig(filename + "epsilon.png")
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, label="2")
    ax2.plot(x, scores, color="C1")
    ax2.set_xlabel("Episode", color="C1")
    ax2.set_ylabel('Average score', color="C1")
    ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")
    plt.savefig(filename + "scores.png")
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, label="3")
    ax3.plot(x, steps, color="C2")
    ax3.set_xlabel("Episode", color="C2")
    ax3.set_ylabel('Average steps', color="C2")
    ax3.tick_params(axis='x', colors="C2")
    ax3.tick_params(axis='y', colors="C2")
    plt.savefig(filename + "steps.png")
    plt.close(fig3)

    labels = ["moved more than 1 level higher", "build on dome", "moved to dome", "build on occupied tile",
              "moved to occupied tile", "build outside board", "moved outside board"]
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111, label="4")
    ax4.plot(x, invalid_moves[0], '--', label=labels[0])
    ax4.plot(x, invalid_moves[1], '--', label=labels[1])
    ax4.plot(x, invalid_moves[2], ':', label=labels[2])
    ax4.plot(x, invalid_moves[3], ':', label=labels[3])
    ax4.plot(x, invalid_moves[4], ':', label=labels[4])
    ax4.plot(x, invalid_moves[5], label=labels[5])
    ax4.plot(x, invalid_moves[6], label=labels[6])
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Count')
    box = ax.get_position()
    ax4.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=45)
    plt.savefig(filename + "invalid_moves.png")
    plt.close(fig4)

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111, label="5")
    ax5.scatter(x, wins, color="C2")
    ax5.set_xlabel("Episode", color="C3")
    ax5.set_ylabel('Wins', color="C3")
    ax5.tick_params(axis='x', colors="C3")
    ax5.tick_params(axis='y', colors="C3")
    plt.savefig(filename + "wins.png")
    plt.close(fig5)

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

# @profile
def main():  # vypisovat cas epizody/ epizod
    logfile_name = "logs/" + str(seed) + "_train"
    with open(logfile_name, 'w') as logfile:
        env = MyEnv()
        env.mode = mode
        env.reset()
        best_score = -np.inf

        agent = DuelingDQNAgent(gamma=gamma, epsilon=epsilon, lr=lr,
                                input_dims=env.observation_space.shape,
                                n_actions=env.action_space.n, mem_size=mem_size, eps_min=eps_min,
                                batch_size=batch_size, replace=replace, eps_dec=eps_dec,
                                learn_amount=learn_amount, seed=seed, checkpoint_dir='models/')

        figure_file = 'plots/' + str(seed) + "/"

        if load:
            agent.load_models(old_seed)

        checkpoint_steps = 0
        checkpoint_score = 0
        total_wins = 0
        checkpoint_wins = 0
        checkpoint_scores_array, checkpoint_epsilon_array, checkpoint_steps_array, episodes_num_array, checkpoint_wins_array = [], [], [], [], []
        invalid_move_types = [0, 0, 0, 0, 0, 0, 0]
        invalid_moves_over_time = [[], [], [], [], [], [], []]
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
            episode_score = 0

            while not done:
                action = agent.choose_action(observation)  # env by mohol poslat agentovi ktore tahy su neplatne
                observation_, reward, done, info = env.step(action)
                last_message = info.get('message')
                episode_score += reward
                agent.store_transition(observation, action, reward, observation_, int(done))
                if i % learn_frequency == 0:
                    agent.learn()
                observation = observation_
                episode_steps += 1

                if i % checkpoint_every == 0:
                    action_log = "------------player: " + info.get("player") + " move: " + info.get(
                        "move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
                    logfile.write(action_log)
                    logfile.write(env.render())

            if episode_score > best_score:
                best_score = episode_score

            if reward == reward_for_win:
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

                logfile.write("\nend episode " + str(i) + " of " + str(n_episodes) + " steps: " + str(episode_steps) + " score: " + str(episode_score) + "\n")
                episode_log = "checkpoint: " + str(i) + "/" + str(n_episodes) + " best score: " + str(best_score) + " average score: " + "{:.2f}".format(average_score) + " epsilon " + "{:.2f}".format(agent.epsilon) + "\n" + " average steps: " + str(average_steps) + " elapsed time: " + str(elapsed_time) + " wins this checkpoint: " + str(checkpoint_wins) + " total wins: " + str(total_wins) + "\n"
                logfile.write(episode_log)
                print(episode_log)

                checkpoint_scores_array.append(average_score)
                checkpoint_steps_array.append(average_steps)
                checkpoint_epsilon_array.append(agent.epsilon)
                episodes_num_array.append(i)
                checkpoint_wins_array.append(checkpoint_wins)

                invalid_moves_over_time = update_invalid_moves_over_time(invalid_moves_over_time, invalid_move_types)
                plot_learning_curve(episodes_num_array, checkpoint_scores_array, checkpoint_epsilon_array, checkpoint_steps_array, invalid_moves_over_time, checkpoint_wins_array, figure_file)

                checkpoint_steps = 0
                checkpoint_score = 0
                checkpoint_wins = 0
                invalid_move_types = [0, 0, 0, 0, 0, 0, 0]

        end_timestamp = datetime.now()
        end_elapsed_time = datetime.now() - start_time
        end_log = "end: " + " seed: " + str(seed) + " timestamp " + str(end_timestamp) + " , training time: " + str(end_elapsed_time)
        logfile.write(end_log)
        print(end_log)

setup_output_files_directories()
main()
