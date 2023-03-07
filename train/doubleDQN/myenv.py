import gym
from gym import spaces
from engine.Board import Board
from engine.Move import Move
from ai.MinMaxBot import MinMaxBot
from ai.RandomBot import RandomBot
from ai.RLBot import RLBot

class MyEnv(gym.Env):

    def __init__(self, mode = "cooperative"):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Box(low=-1, high=5, shape=(2, 5, 5), dtype=int)

        self.board = Board()
        self.mode = mode # cooperative or competitive or single

        self.reward_for_invalid_move = -100
        self.reward_for_valid_move = 1
        self.reward_for_lose = -10
        self.reward_for_win = 10

        self.primary_player_color = "white"
        self.secondary_player_color = "black"
        # self.secondary_player = MinMaxBot(self.secondary_player_color)
        self.secondary_player = RandomBot(self.secondary_player_color)
        # self.secondary_player = RLBot(self.secondary_player_color)
        self.bot_name = "RANDOM" # NONE/RANDOM/MINMAX/RL

    def primary_player_step(self, action):
        chosenMove = self.create_move(action, self.primary_player_color)
        valid, msg = self.check_move_valid(chosenMove, self.board)
        if not valid:
            return self.encode_input(self.board), self.reward_for_invalid_move, 1, {"move": chosenMove.__str__(),
                                                               "player": self.primary_player_color, "valid": "INVALID",
                                                               "win": False, "message": msg}
        self.board.update_board_after_move(chosenMove)
        end, _ = self.board.check_if_game_ended(self.primary_player_color)
        if end:
            return self.encode_input(self.board), self.reward_for_win, 1, {"move": chosenMove.__str__(),
                                                                   "player": self.primary_player_color, "valid": "WIN",
                                                                   "win": True, "message": ""}
        else:
            return self.encode_input(self.board), 0, 0, {"move": chosenMove.__str__(),
                                                                      "player": self.primary_player_color, "valid": "VALID",
                                                                       "win": False, "message": ""}


    def secondary_player_step(self):
        chosenMove = ""
        if self.bot_name == "RANDOM" or self.bot_name == "MINMAX":
            chosenMove = self.secondary_player.make_turn(self.board)
            self.board.update_board_after_move(chosenMove)
        elif self.bot_name == "RL":
            chosenMove = self.secondary_player.make_turn(self)
            self.board.update_board_after_move(chosenMove)
        elif self.bot_name == "NONE":
            chosenMove = "NONE"
        end, _ = self.board.check_if_game_ended(self.secondary_player_color)
        if end:
            return self.encode_input(self.board), self.reward_for_lose, 1, {"move": chosenMove.__str__(),
                                                                   "player": self.secondary_player_color, "valid": "LOSE",
                                                                   "win": False, "message": ""}
        else:
            return self.encode_input(self.board), 0, 0, {"move": chosenMove.__str__(),
                                                                      "player": self.secondary_player_color, "valid": "VALID",
                                                                       "win": False, "message": ""}

    def calculate_reward(self):
        if self.mode == "cooperative":
            return self.reward_for_valid_move
        elif self.mode == "competitive":
            return self.get_player_height_diff(self.board, self.primary_player_color)
        elif self.mode == "single":
            return self.get_player_height(self.board, self.primary_player_color)


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
                if board.tiles[i][j].player >= 1:
                    height += board.tiles[i][j].level

        if player_color == "black":
            height *= -1

        return height

    def reset(self):
        self.board = Board()
        return self.encode_input(self.board)

    def encode_input(self, board):
        inputTensor = []
        board_heights = []
        board_players = []

        for i in board.tiles:
            row_heights = []
            row_players = []

            for j in i:
                row_heights.append(j.level)
                row_players.append(j.player)
                # if self.players_turn == "white":
                #     row_players.append(j.player)
                # else:
                #     row_players.append(-j.player)

            board_heights.append(row_heights)
            board_players.append(row_players)

        inputTensor.append(board_heights)
        inputTensor.append(board_players)
        return inputTensor

    def render(self):
        return self.board.__str__()

    def make_clockwise_list(self, row1, col1):
        coordsList = [(row1 - 1, col1), (row1 - 1, col1 + 1), (row1, col1 + 1), (row1 + 1, col1 + 1),
                      (row1 + 1, col1), (row1 + 1, col1 - 1), (row1, col1 - 1), (row1 - 1, col1 - 1)]
        return coordsList

    def check_move_valid(self, move, board):
        row_from, col_from = move.fromCoords
        row_to, col_to = move.toCoords
        row_build, col_build = move.buildCoords

        if 0 <= row_to < 5 and 0 <= col_to < 5:
            if 0 <= row_build < 5 and 0 <= col_build < 5:
                current_tile = board.tiles[row_from][col_from]
                move_target = board.tiles[row_to][col_to]
                build_target = board.tiles[row_build][col_build]
                if move_target.player == 0:
                    if build_target.player == 0 or build_target == current_tile:
                        if move_target.level != 4:
                            if build_target.level != 4:
                                if move_target.level <= current_tile.level + 1:
                                    return True, "OK"

                                return False, "moved more than 1 level higher"
                            return False, "build on dome"
                        return False, "moved to dome"
                    return False, "build on occupied tile"
                return False, "moved to occupied tile"
            return False, "build outside board"
        return False, "moved outside board"


    def create_move(self,number, player_color):
        if number < 64:
            if player_color == "white":
                from_row, from_col = self.board.white1
            else:
                from_row, from_col = self.board.black1
        else:
            if player_color == "white":
                from_row, from_col = self.board.white2
            else:
                from_row, from_col = self.board.black2
            number -= 64

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        to_row, to_col = toCoordsList[int(number / 8)]
        buildCoordsList = self.make_clockwise_list(to_row, to_col)
        build_row, build_col = buildCoordsList[int(number % 8)]

        return Move((from_row, from_col), (to_row, to_col), (build_row, build_col), player_color)