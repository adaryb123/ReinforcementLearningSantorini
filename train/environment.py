"""
Environment of the game Santorini for Double deep Q learning
Author: Adam Rybansky (xryban00)
FIT VUT 2023
Based on: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DuelingDQN
"""

from gym import Env, spaces
from engine.Board import Board, encode_board
from ai.RLBot import RLBot
from ai.MinMaxBot import MinMaxBot
from ai.RandomBot import RandomBot
from ai.HeuristicBot import HeuristicBot
from ai.HeuristicCompetitiveBot import HeuristicCompetitiveBot
import copy


class Environment(Env):

    def __init__(self, mode, seed, bot_name, checkpoint_frequency, canals, adamw_optimizer):
        super(Environment, self).__init__()
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Box(low=-1, high=5, shape=(canals, 5, 5), dtype=int)

        self.board = Board()
        self.mode = mode
        if self.mode == "single_lookback":
            self.last_board = copy.deepcopy(self.board)

        self.reward_for_invalid_move = -100
        self.reward_for_valid_move = 1
        self.reward_for_lose = -10
        self.reward_for_win = 10

        if self.mode == "normalized":
            self.reward_for_invalid_move = -1
            self.reward_for_win = 1
            self.reward_for_valid_move = 0

        self.primary_player_color = "white"
        self.secondary_player_color = "black"
        self.bot_name = bot_name

        self.secondary_player = "NONE"
        self.seed = seed
        self.checkpoint_frequency = checkpoint_frequency
        self.adamw_optimizer = adamw_optimizer

        if self.bot_name == "RANDOM":
            self.secondary_player = RandomBot(self.secondary_player_color)
        elif self.bot_name == "MINMAX":
            self.secondary_player = MinMaxBot(self.secondary_player_color)
        elif self.bot_name == "RL":
            self.secondary_player = RLBot(self.secondary_player_color, self.seed, self.observation_space.shape,
                                          self.action_space.n, self.checkpoint_frequency, self.adamw_optimizer)
        elif self.bot_name == "HEURISTIC":
            self.secondary_player = HeuristicBot(self.secondary_player_color)

        elif self.bot_name == "HEURISTIC-COMPETITIVE":
            self.secondary_player = HeuristicCompetitiveBot(self.secondary_player_color)

    def set_secondary_player(self,new_bot_name):
        self.bot_name = new_bot_name
        if self.bot_name == "RANDOM":
            self.secondary_player = RandomBot(self.secondary_player_color)
        elif self.bot_name == "MINMAX":
            self.secondary_player = MinMaxBot(self.secondary_player_color)
        elif self.bot_name == "RL":
            self.secondary_player = RLBot(self.secondary_player_color, self.seed, self.observation_space.shape,
                                          self.action_space.n, self.checkpoint_frequency)
        elif self.bot_name == "HEURISTIC":
            self.secondary_player = HeuristicBot(self.secondary_player_color)

        elif self.bot_name == "HEURISTIC-COMPETITIVE":
            self.secondary_player = HeuristicCompetitiveBot(self.secondary_player_color)

    def primary_player_step(self, action):
        """ Process move from Double deep Q learning agent"""
        if self.mode == "single_lookback":
            self.last_board = copy.deepcopy(self.board)

        chosenMove = self.board.create_move_from_number(action, self.primary_player_color)
        valid, msg = self.board.check_move_valid(chosenMove)
        if not valid:
            return encode_board(self.board), self.reward_for_invalid_move, 1, {"move": chosenMove.__str__(),
                                                               "player": self.primary_player_color, "valid": "INVALID",
                                                               "win": False, "message": msg}
        self.board.update_board_after_move(chosenMove)
        end, _ = self.board.check_if_player_won(self.primary_player_color)
        if end:
            return encode_board(self.board), self.reward_for_win, 1, {"move": chosenMove.__str__(),
                                                                   "player": self.primary_player_color, "valid": "WIN",
                                                                   "win": True, "message": ""}
        else:
            return encode_board(self.board), 0, 0, {"move": chosenMove.__str__(),
                                                                      "player": self.primary_player_color, "valid": "VALID",
                                                                       "win": False, "message": ""}


    def secondary_player_step(self):
        """ Process move from enemy bot"""
        chosenMove = ""
        if self.bot_name == "NONE":
            chosenMove = "NONE"
        else:
            chosenMove = self.secondary_player.make_turn(self.board)
            self.board.update_board_after_move(chosenMove)
        end, _ = self.board.check_if_player_won(self.secondary_player_color)
        if end:
            return encode_board(self.board), self.reward_for_lose, 1, {"move": chosenMove.__str__(),
                                                                   "player": self.secondary_player_color, "valid": "LOSE",
                                                                   "win": False, "message": ""}
        else:
            return encode_board(self.board), 0, 0, {"move": chosenMove.__str__(),
                                                                      "player": self.secondary_player_color, "valid": "VALID",
                                                                       "win": False, "message": ""}

    def calculate_reward(self):
        """ Calculate reward based on current state of the board"""
        if self.mode == "cooperative":
            return self.reward_for_valid_move
        elif self.mode == "competitive":
            return self.board.get_player_height_diff(self.primary_player_color)
        elif self.mode == "single":
            return self.board.get_player_height(self.primary_player_color)
        elif self.mode == "single_lookback":
            return self.get_player_height_change(self.board, self.last_board, self.primary_player_color)
        elif self.mode == "normalized":
            return 0

    def reset(self):
        """ Reset board to default states"""
        self.board = Board()
        return encode_board(self.board)

    def render(self):
        return self.board.__str__()

    def get_player_height_change(self, new_board, prev_board, player_color):
        return new_board.get_player_height(player_color) - prev_board.get_player_height(player_color)