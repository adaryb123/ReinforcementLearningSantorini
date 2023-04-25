"""
Utility functions used in trian.py script
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

import os
import matplotlib.pyplot as plt
import pickle

class PlotItemStorage:
    def __init__(self):
        self.episodes_num_array = []
        self.checkpoint_scores_array = []
        self.checkpoint_epsilon_array = []
        self.checkpoint_steps_array = []
        self.checkpoint_wins_array = []
        self.invalid_moves_over_time = [[], [], [], [], [], [], []]

    def __str__(self):
        output_string = ""
        output_string += str(self.episodes_num_array) + "\n"
        output_string += str(self.checkpoint_scores_array) + "\n"
        output_string += str(self.checkpoint_epsilon_array) + "\n"
        output_string += str(self.checkpoint_steps_array) + "\n"
        output_string += str(self.invalid_moves_over_time) + "\n"
        return output_string

def ps_store(plot_object, seed):
    filename = "plots/" + str(seed) + "/values.pickle"
    if not os.path.exists(filename):
        with open(filename, 'w') as _:
            pass

    with open(filename, "wb") as pickle_out:
        pickle.dump(plot_object, pickle_out)

def ps_load(seed):
    filename = "plots/" + str(seed) + "/values.pickle"
    if not os.path.exists(filename):
        print("ERROR: no plot values to load")
        exit(0)

    with open(filename, "rb") as pickle_in:
        return pickle.load(pickle_in)


def setup_output_files_directories(seed):
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
    plt.xticks(rotation=45)
    plt.savefig(filename + "epsilon.png")
    plt.close(fig)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, label="2")
    ax2.plot(x, scores, color="C1")
    ax2.set_xlabel("Episode", color="C1")
    ax2.set_ylabel('Average score', color="C1")
    ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")
    plt.xticks(rotation=45)
    plt.savefig(filename + "scores.png")
    plt.close(fig2)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, label="3")
    ax3.plot(x, steps, color="C2")
    ax3.set_xlabel("Episode", color="C2")
    ax3.set_ylabel('Average steps', color="C2")
    ax3.tick_params(axis='x', colors="C2")
    ax3.tick_params(axis='y', colors="C2")
    plt.xticks(rotation=45)
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
    plt.xticks(rotation=45)
    plt.savefig(filename + "wins.png")
    plt.close(fig5)