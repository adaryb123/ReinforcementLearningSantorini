import numpy as np
import matplotlib.pyplot as plt
from dueling_dqn_agent import DuelingDQNAgent
from myenv import MyEnv
import os
import random


# if training a new model
# seed = random.randint(10000,99999)
# load = False

# if continuing on an already trained model
seed = 71846
load = True

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

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    ax2.scatter(x, scores, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def main():
    with open('logs/train_log.txt', 'w') as logfile:
        env = MyEnv()
        env.reset()
        best_score = -np.inf
        n_episodes = 10
        agent = DuelingDQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                                input_dims=(env.observation_space.shape),
                                n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                                batch_size=32, replace=10000, eps_dec=1e-5,
                                chkpt_dir='models/', algo='DuelingDQNAgent_' + str(seed),
                                env_name='Santorini')

        figure_file = 'plots/' + str(seed) + '.png'

        if load:
            agent.load_models()

        total_steps = 0
        scores, eps_history, steps_array = [], [], []

        for i in range(n_episodes+1):
            done = False
            observation = env.reset()
            logfile.write("start episode "+str(i) + "\n")
            logfile.write(env.render())

            episode_steps = 0
            score = 0
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)

                score += reward
                action_log = "------------player: " + info.get("player") + " move: " + info.get("move") + " which is: " + info.get("valid") + ": " + info.get("message") + "\n"
                logfile.write(action_log)
                logfile.write(env.render())

                agent.store_transition(observation, action, reward, observation_, int(done))
                agent.learn()
                observation = observation_
                episode_steps += 1

            episode_log  = 'end episode: ' + str(i) + ' score: ' +str(score)  + str(' best score %.2f ' % best_score) + str(' epsilon %.2f ' % agent.epsilon) + ' steps ' + str(episode_steps) + "\n"
            logfile.write(episode_log)

            if score > best_score:
                best_score = score

            agent.save_models()

            scores.append(score)
            total_steps += episode_steps
            steps_array.append(total_steps)
            eps_history.append(agent.epsilon)

            if i % 100 == 0:
                print("episode " + str(i) + " of " + str(n_episodes))

        print("done")
        x = [i+1 for i in range(len(scores))]
        plot_learning_curve(steps_array, scores, eps_history, figure_file)

setup_output_files_directories()
main()