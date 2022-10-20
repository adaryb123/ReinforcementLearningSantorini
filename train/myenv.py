import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque
from engine.Board import Board
from engine.Move import Move

class MyEnv(gym.Env):

    def __init__(self):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Discrete(128)
        self.observation_space = spaces.Box(low=0, high=255, shape=(5, 5, 2))

        # self.observation_space = spaces.Box(low=-500, high=500)
        self.board = Board()
        self.color = "white"
        self.prev_actions = []

    def step(self, action):
        # print(action)
        chosenMove = self.create_move(action)
        # print(chosenMove)
        valid,log = self.check_move_valid(chosenMove,self.board)
        if not valid:
            print("---------------------------------------------------------------invalid move: " + log)
            return self.encode_input(self.board), -10, 1, {}
        else:
            self.prev_actions.append(chosenMove)
            self.board.update_board_after_move(chosenMove)
            end, _ = self.board.check_if_game_ended("white")
            if end:
                print("---------------------------------------------------------------win")
                return self.encode_input(self.board), 1, 1, {}
            else:
                print("---------------------------------------------------------------good move")
                return self.encode_input(self.board), 0, 0, {}


    def reset(self):
        self.board = Board()
        self.prev_actions = []
        return np.array(self.encode_input(self.board))


    def encode_input(self, board):
        inputTensor = []
        for i in board.tiles:
            row = []
            for j in i:
                if self.color == "black":
                    row.append([j.level, - j.player])
                elif self.color == "white":                   # WTF
                    row.append([j.level, j.player])
            inputTensor.append(row)

        return inputTensor

    def render(self):
        print(self.board.__str__())

  # def decode_output(self, output, board):
  #       row1 = 0
  #       col1 = 0
  #       row2 = 0
  #       col2 = 0
  #       moves = []
  #       if self.color == "black":
  #           row1, col1 = board.black1
  #           row2, col2 = board.black2
  #       else:
  #           row1, col1 = board.white1
  #           row2, col2 = board.white2
  #
  #       fromCoords = ()
  #       toCoordsList = []
  #       buildCoordsList = []
  #       offset = 0
  #
  #       for i in range(output):
  #           if i == 0:
  #               offset = 0
  #               fromCoords = (row1, col1)
  #               toCoordsList = self.make_clockwise_list(row1,col1)
  #           elif i == 64:
  #               offset = 64
  #               fromCoords = (row2, col2)
  #               toCoordsList = self.make_clockwise_list(row2, col2)
  #           index = i - offset
  #           toCoords = toCoordsList[int(index / 8)]
  #
  #           if i % 8 == 0:
  #               row_to, col_to = toCoords
  #               buildCoordsList = self.make_clockwise_list(row_to, col_to)
  #           buildCoords = buildCoordsList[int(i % 8)]
  #
  #           # teraz sa zahodia neplatnne tahy, vektor sa znormalizuje a vyberie sa najlepsi (epsilon nieco) tah
  #           potential_move = Move(fromCoords,toCoords,buildCoords,self.color)
  #           if self.check_move_valid(potential_move,board):
  #               moves.append(potential_move)
  #           else:
  #               output[i] = 0



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
            from_row, from_col = self.board.white1
        else:
            from_row, from_col = self.board.white2
            number -= 64

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        to_row, to_col = toCoordsList[int(number / 8)]
        buildCoordsList = self.make_clockwise_list(to_row, to_col)
        build_row, build_col = buildCoordsList[int(number % 8)]

        return Move((from_row, from_col), (to_row, to_col), (build_row, build_col), "white")

