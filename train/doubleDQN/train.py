import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn_agent import DuelingDQNAgent
from myenv import MyEnv
import os
import random
from datetime import datetime
from line_profiler_pycharm import profile

# if training a new model
# seed = random.randint(10000,99999)
# load = False

# if continuing on an already trained model
seed = 10225
load = True

n_episodes = 75000
epsilon = 0.17
eps_min = 0.01
log_every = 1000
# plot_every = 1000
learn_frequency = 100
reward_for_win = 1


def setup_output_files_directories():
    models_dir = "models"
    logs_dir = "logs"
    plots_dir = "plots"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


def plot_learning_curve(x, scores, epsilons, steps, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Episode", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    plt.savefig(filename + '.png')
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, label="2")
    ax2.plot(x, scores, color="C1")
    ax2.set_xlabel("Episode", color="C1")
    ax2.set_ylabel('Average steps', color="C1")
    ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    plt.savefig(filename + "_1" + '.png')
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, label="2")

    ax3.scatter(x, steps, color="C2")
    ax3.set_xlabel("Episode", color="C2")
    ax3.set_ylabel('Average score', color="C2")
    ax3.tick_params(axis='x', colors="C2")
    ax3.tick_params(axis='y', colors="C2")

    plt.savefig(filename + "_2" + '.png')
    plt.close(fig3)

# @profile
def main():  # vypisovat cas epizody/ epizod
    with open('logs/train_log.txt', 'w') as logfile:
        env = MyEnv()
        env.reset()
        best_score = -np.inf

        agent = DuelingDQNAgent(gamma=0.99, epsilon=epsilon, lr=0.0001,
                                input_dims=env.observation_space.shape,
                                n_actions=env.action_space.n, mem_size=50000, eps_min=eps_min,
                                batch_size=32, replace=10000, eps_dec=1e-5,
                                chkpt_dir='models/', algo='DuelingDQNAgent_' + str(seed),
                                env_name='Santorini')

        figure_file = 'plots/' + str(seed)

        if load:
            agent.load_models()

        total_steps = 0
        total_score = 0
        scores, eps_history, steps_array, episodes_num = [], [], [], []
        win_count = 0

        start_time = datetime.now()
        last_timestamp = datetime.now()
        start_log = "start: " + " seed: " + str(seed) + " timestamp: " + str(start_time)
        logfile.write(start_log)
        print(start_log)

        for i in range(1, n_episodes + 1):
            done = False
            observation = env.reset()

            if i % log_every == 0:
                logfile.write("start episode " + str(i) + " of " + str(n_episodes) + "\n")
                logfile.write(env.render())

            episode_steps = 0
            episode_score = 0
            while not done:
                action = agent.choose_action(observation)  # env by mohol poslat agentovi ktore tahy su neplatne
                observation_, reward, done, info = env.step(action)
                episode_score += reward
                agent.store_transition(observation, action, reward, observation_, int(done))
                if i % learn_frequency == 0:
                    agent.learn()
                observation = observation_
                episode_steps += 1

                if i % log_every == 0:
                    action_log = "------------player: " + info.get("player") + " move: " + info.get(
                        "move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
                    logfile.write(action_log)
                    logfile.write(env.render())

            if episode_score > best_score:
                best_score = episode_score

            if reward == reward_for_win:
                win_count += 1

            agent.save_models()

            total_steps += episode_steps
            total_score += episode_score

            if i % log_every == 0:
                average_score = total_score / log_every
                average_steps = total_steps / log_every
                elapsed_time = datetime.now() - last_timestamp
                last_timestamp = datetime.now()

                episode_log = "episode: " + str(i) + "/" + str(n_episodes) + " score: " + str(episode_score) + " best score: " + str(best_score) + " average score: " + "{:.2f}".format(average_score) + " epsilon " + "{:.2f}".format(agent.epsilon) + "\n   steps: " + str(episode_steps) + " average steps: " + str(average_steps) + " elapsed time: " + str(elapsed_time) + " win count: " + str(win_count) + "\n"
                logfile.write(episode_log)
                print(episode_log)

                scores.append(average_score)
                steps_array.append(average_steps)
                eps_history.append(agent.epsilon)
                episodes_num.append(i)
                plot_learning_curve(episodes_num, scores, eps_history, steps_array, figure_file)

                total_steps = 0
                total_score = 0

        end_timestamp = datetime.now()
        end_elapsed_time = datetime.now() - start_time
        end_log = "end: " + " seed: " + str(seed) + " timestamp " + str(end_timestamp) + " , training time: " + str(end_elapsed_time)
        logfile.write(end_log)
        print(end_log)

setup_output_files_directories()
main()
