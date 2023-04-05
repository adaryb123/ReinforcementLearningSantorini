import gym
from gym import spaces
from engine.Board import Board, encode_board
from engine.Move import Move
from RLBot import RLBot
from ai.MinMaxBot import MinMaxBot
from ai.RandomBot import RandomBot
from ai.HeuristicBot import HeuristicBot
import copy


class MyEnv(gym.Env):

    def __init__(self, mode, seed, bot_name, checkpoint_frequency):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Box(low=-1, high=5, shape=(3, 5, 5), dtype=int)

        self.board = Board()
        self.mode = mode # cooperative or competitive or single or single_lookback

        self.reward_for_invalid_move = -100
        self.reward_for_valid_move = 1
        self.reward_for_lose = -10
        self.reward_for_win = 10

        self.primary_player_color = "white"
        self.secondary_player_color = "black"
        self.bot_name = bot_name
        self.secondary_player = "NONE"
        if self.bot_name == "RANDOM":
            self.secondary_player = RandomBot(self.secondary_player_color)
        elif self.bot_name == "MINMAX":
            self.secondary_player = MinMaxBot(self.secondary_player_color)
        elif self.bot_name == "RL":
            self.secondary_player = RLBot(self.secondary_player_color, self.observation_space.shape,
                                          self.action_space.n, seed, checkpoint_frequency)
        elif self.bot_name == "HEURISTIC":
            self.secondary_player = HeuristicBot(self.secondary_player_color)
        if self.mode == "single_lookback":
            self.last_board = copy.deepcopy(self.board)

    def primary_player_step(self, action):
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
        chosenMove = ""
        if self.bot_name == "RANDOM" or self.bot_name == "MINMAX" or self.bot_name == "HEURISTIC":
            chosenMove = self.secondary_player.make_turn(self.board)
            self.board.update_board_after_move(chosenMove)
        elif self.bot_name == "RL":
            chosenMove = self.secondary_player.make_turn(self)
            self.board.update_board_after_move(chosenMove)
        elif self.bot_name == "NONE":
            chosenMove = "NONE"
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
        if self.mode == "cooperative":
            return self.reward_for_valid_move
        elif self.mode == "competitive":
            return self.get_player_height_diff(self.board, self.primary_player_color)
        elif self.mode == "single":
            return self.get_player_height(self.board, self.primary_player_color)
        elif self.mode == "single_lookback":
            return self.get_player_height_change(self.board, self.last_board, self.primary_player_color)


    def get_player_height_diff(self, board, player_color):
        height_diff = 0
        for i in range(5):
            for j in range(5):
                if board.tiles[i][j].player >= 1:
                    height_diff += board.tiles[i][j].level
                elif board.tiles[i][j].player <= -1:
                    height_diff -= board.tiles[i][j].level
        if player_color == "black":
            height_diff *= -1

        return height_diff

    def get_player_height(self, board, player_color):
        height = 0
        for i in range(5):
            for j in range(5):
                if player_color == "white":
                    if board.tiles[i][j].player >= 1:
                        height += board.tiles[i][j].level
                elif player_color == "black":
                    if board.tiles[i][j].player <= -1:
                        height += board.tiles[i][j].level

        return height

    def get_player_height_change(self, new_board, prev_board, player_color):
        return self.get_player_height(new_board,player_color) - self.get_player_height(prev_board,player_color)


    def reset(self):
        self.board = Board()
        return encode_board(self.board)

    def render(self):
        return self.board.__str__()
