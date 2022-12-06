import gym
from gym import spaces
import numpy as np
from engine.Board import Board
from engine.Move import Move
import torch as T

class MyEnv(gym.Env):

    def __init__(self):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Box(low=-1, high=5, shape=(2, 5, 5), dtype=int)

        self.board = Board()
        self.players_turn = "white"
        self.prev_actions = []
        self.mode = "cooperative" # cooperative or competitive
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def set_next_player(self):
        if self.players_turn == "white":
            self.players_turn = "black"
        else:
            self.players_turn = "white"

    def get_prev_player(self):
        if self.players_turn == "white":
            return "black"
        else:
            return "white"

    def step(self, action):
        if self.mode == "competitive":
            chosenMove = self.create_move(action)
            valid, log = self.check_move_valid(chosenMove,self.board)
            if not valid:
                return self.encode_input(self.board), -10, 1, {"move": chosenMove.__str__(), "player": self.players_turn, "valid": "INVALID", "win": False, "message": log}
            else:
                self.prev_actions.append(chosenMove)
                self.board.update_board_after_move(chosenMove)
                end, _ = self.board.check_if_game_ended(self.players_turn)
                if end:
                    return self.encode_input(self.board), 100, 1, {"move": chosenMove.__str__(), "player": self.players_turn,  "valid": "WIN", "win": True, "message": ""}
                else:
                    reward = self.get_player_height_diff(self.board)        # iba jeho vysku
                    self.set_next_player()
                    return self.encode_input(self.board), reward, 0, {"move": chosenMove.__str__(), "player": self.get_prev_player(),  "valid": "VALID", "win": False, "message": ""}
        elif self.mode == "cooperative":
            chosenMove = self.create_move(action)
            valid, log = self.check_move_valid(chosenMove, self.board)
            if not valid:
                return self.encode_input(self.board), 0, 1, {"move": chosenMove.__str__(),
                                                               "player": self.players_turn, "valid": "INVALID",
                                                               "win": False, "message": log}
            else:
                self.prev_actions.append(chosenMove)
                self.board.update_board_after_move(chosenMove)
                end, _ = self.board.check_if_game_ended(self.players_turn)
                if end:
                    return self.encode_input(self.board), 100, 1, {"move": chosenMove.__str__(),
                                                                  "player": self.players_turn, "valid": "WIN",
                                                                  "win": True, "message": ""}
                else:
                    self.set_next_player()
                    return self.encode_input(self.board), 1 , 0, {"move": chosenMove.__str__(),
                                                                      "player": self.get_prev_player(),
                                                                      "valid": "VALID", "win": False, "message": ""}
        elif self.mode == "single":
            chosenMove = self.create_move(action)
            valid, log = self.check_move_valid(chosenMove, self.board)
            if not valid:
                return self.encode_input(self.board), -10, 1, {"move": chosenMove.__str__(),
                                                               "player": self.players_turn, "valid": "INVALID",
                                                               "win": False, "message": log}
            else:
                self.prev_actions.append(chosenMove)
                self.board.update_board_after_move(chosenMove)
                end, _ = self.board.check_if_game_ended(self.players_turn)
                if end:
                    return self.encode_input(self.board), 100, 1, {"move": chosenMove.__str__(),
                                                                   "player": self.players_turn, "valid": "WIN",
                                                                   "win": True, "message": ""}
                else:
                    reward = self.get_player_height(self.board)
                    self.set_next_player()
                    return self.encode_input(self.board), reward, 0, {"move": chosenMove.__str__(),
                                                                      "player": self.get_prev_player(),
                                                                      "valid": "VALID", "win": False, "message": ""}



        else:
                print("unknown mode")
                exit(0)

    def get_player_height_diff(self, board):
        height_diff = 0
        for i in range(5):
            for j in range(5):
                if board.tiles[i][j].player == 1:
                    height_diff += board.tiles[i][j].level
                elif board.tiles[i][j].player == -1:
                    height_diff -= board.tiles[i][j].level
        if self.players_turn == "black":
            height_diff *= -1

        return height_diff

    def get_player_height(self, board):
        height = 0
        for i in range(5):
            for j in range(5):
                if board.tiles[i][j].player == self.players_turn:
                    height += board.tiles[i][j].level

        return height


    def reset(self):
        self.board = Board()
        self.prev_actions = []
        self.players_turn = "white"
        # return np.array(self.encode_input(self.board))
        return self.encode_input(self.board)

    def encode_input(self, board):
        inputTensor = T.empty((1, 2, 5, 5), dtype=T.float32, device=self.device)
        for i in range(len(board.tiles)):
            for j in range(5):
                inputTensor[0, 0, i, j] = board.tiles[i][j].level
                if self.players_turn == "white":
                    inputTensor[0, 1, i, j] = board.tiles[i][j].player
                else:
                    inputTensor[0, 1, i, j] = - board.tiles[i][j].player
        # inputTensor = []
        # board_heigths = []
        # for i in range(len(board.tiles)):
            # row = []
            # for j in range(5):
                # row.append(board.tiles[i][j].level)
                # inputTensor[0, i, j] = board.tiles[i][j].level
            # board_heigths.append(row)
        # inputTensor.append(board_heigths)

        # board_players = []
        # for i in range(5):
            # row = []
            # for j in range(5):
            #     if self.players_turn == "white":
                    # row.append(board.tiles[i][j].player)
                    # inputTensor[1, i, j] = board.tiles[i][j].player
                # else:
                    # row.append(- board.tiles[i][j].player)
                    # inputTensor[1, i, j] = - board.tiles[i][j].player
            # board_players.append(row)
        # inputTensor.append(board_players)
        # print(inputTensor)
        return inputTensor

    def render(self):
        return self.board.__str__()

    def make_clockwise_list(self,row1, col1):
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
                    if build_target.player == 0:
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


    def create_move(self,number):
        if number < 64:
            if self.players_turn == "white":
                from_row, from_col = self.board.white1
            else:
                from_row, from_col = self.board.black1
        else:
            if self.players_turn == "white":
                from_row, from_col = self.board.white2
            else:
                from_row, from_col = self.board.black2
            number -= 64

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        to_row, to_col = toCoordsList[int(number / 8)]
        buildCoordsList = self.make_clockwise_list(to_row, to_col)
        build_row, build_col = buildCoordsList[int(number % 8)]

        return Move((from_row, from_col), (to_row, to_col), (build_row, build_col), self.players_turn)