from engine.Tile import Tile
from engine.Move import Move
import random

class Board:
    def __init__(self, tensor="NONE"):
        if tensor != "NONE":
            self.init_from_tensor(tensor)
        else:
            self.heights = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
            self.players = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

            self.black1, self.black2, self.white1, self.white2 = self.setup_players()

    def init_from_tensor(self, tensor):
        self.heights = tensor[0]
        self.players = tensor[1]

        for i in range(len(self.players)):
            for j in range(len(self.players)):
                player = self.players[i][j]
                if player == -1:
                    self.black1 = (i, j)
                elif player == -2:
                    self.black2 = (i, j)
                elif player == 1:
                    self.white1 = (i, j)
                elif player == 2:
                    self.white2 = (i, j)

    def construct_tile_string(self,row,col):
        output = ""
        spaces = 5
        if self.heights[row][col] == 0:
            spaces = 5
        if self.heights[row][col] == 1:
            output += "C"
            spaces = 4
        elif self.heights[row][col] == 2:
            output += "CC"
            spaces = 3
        elif self.heights[row][col] == 3:
            output += "CCC"
            spaces = 2
        elif self.heights[row][col] == 4:
            output += "CCCD"
            spaces = 1

        if self.players[row][col] <= -1:
            output += "_X"
            spaces -= 2
        elif self.players[row][col] >= 1:
            output += "_0"
            spaces -= 2

        for i in range(spaces):
            output += " "

        return output
    def __str__(self):
        output = "                |   A   |   B   |   C   |   D   |   E   |\n"
        output += "            ____|_______|_______|_______|_______|_______|\n"
        for i in range(5):
            output += "            " + str(i+1) + "  "
            for j in range(5):
                output += " | " + self.construct_tile_string(i,j)
            output += " |\n            ____|_______|_______|_______|_______|_______|\n"
        return output

    def encode(self):
        return [self.heights,self.players]

    def setup_players(self):
        finished = 0
        current = -1
        black1 = 0
        black2 = 0
        white1 = 0
        white2 = 0
        while finished < 4:
            row = random.randint(1,3)
            col = random.randint(1,3)
            if self.players[row][col] == 0:
                self.players[row][col] = current
                finished += 1
                if finished == 1:
                    black1 = (row, col)
                    current = -2
                elif finished == 2:
                    black2 = (row, col)
                    current = 1
                elif finished == 3:
                    white1 = (row, col)
                    current = 2
                elif finished == 4:
                    white2 = (row, col)

        return black1, black2, white1, white2

    def make_clockwise_list(self, row1, col1):
        coordsList = [(row1 - 1, col1), (row1 - 1, col1 + 1), (row1, col1 + 1), (row1 + 1, col1 + 1),
                      (row1 + 1, col1), (row1 + 1, col1 - 1), (row1, col1 - 1), (row1 - 1, col1 - 1)]
        return coordsList

    def check_move_valid(self, move):
        row_from, col_from = move.fromCoords
        row_to, col_to = move.toCoords

        if 0 <= row_to < 5 and 0 <= col_to < 5:
                if self.players[row_to][col_to] == 0:
                        if self.heights[row_to][col_to] != 4:
                                if self.heights[row_to][col_to] <= self.heights[row_from][col_from] + 1:
                                    return True, "OK",

                                return False, "moved more than 1 level higher"
                        return False, "moved to dome"
                return False, "moved to occupied tile"
        return False, "moved outside board"

    def check_build_valid(self, move):
        row_from, col_from = move.fromCoords
        row_build, col_build = move.buildCoords

        if 0 <= row_build < 5 and 0 <= col_build < 5:
            if self.players[row_build][col_build] == 0 or (row_build == row_from and col_build == col_from):
                if self.heights[row_build][col_build] != 4:
                    return True, "OK",

                return False, "build on dome"
            return False, "build on occupied tile"
        return False, "build outside board"
    def find_possible_moves(self, player):
        moves = []
        valid_actions = []
        action_index = 0

        if player == "black":
            from_row, from_col = self.black1
        else:
            from_row, from_col = self.white1

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        for to_row, to_col in toCoordsList:
            move = Move((from_row, from_col), (to_row, to_col), (), player)
            move_valid, _ = self.check_move_valid(move)
            if move_valid:
                buildCoordsList = self.make_clockwise_list(to_row, to_col)
                for build_row, build_col in buildCoordsList:
                    move.setBuildCoords(build_row, build_col)
                    build_valid, _ = self.check_build_valid(move)
                    if build_valid:
                        moves.append(move)
                        valid_actions.append(action_index)
                    action_index += 1
            else:
                action_index += 8

        if player == "black":
            from_row, from_col = self.black2
        else:
            from_row, from_col = self.white2

        action_index = 64
        toCoordsList = self.make_clockwise_list(from_row, from_col)
        for to_row, to_col in toCoordsList:
            move = Move((from_row, from_col), (to_row, to_col), (), player)
            move_valid, _ = self.check_move_valid(move)
            if move_valid:
                buildCoordsList = self.make_clockwise_list(to_row, to_col)
                for build_row, build_col in buildCoordsList:
                    move.setBuildCoords(build_row, build_col)
                    build_valid, _ = self.check_build_valid(move)
                    if build_valid:
                        moves.append(move)
                        valid_actions.append(action_index)
                    action_index += 1
            else:
                action_index += 8

        return moves, valid_actions

    def create_move_from_number(self, number, player_color):
        if number < 64:
            if player_color == "white":
                from_row, from_col = self.white1
            else:
                from_row, from_col = self.black1
        else:
            if player_color == "white":
                from_row, from_col = self.white2
            else:
                from_row, from_col = self.black2
            number -= 64

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        to_row, to_col = toCoordsList[int(number / 8)]
        buildCoordsList = self.make_clockwise_list(to_row, to_col)
        build_row, build_col = buildCoordsList[int(number % 8)]

        return Move((from_row, from_col), (to_row, to_col), (build_row, build_col), player_color)

    def update_board_after_move(self, move):
        fromCoords = move.fromCoords
        toCoords = move.toCoords
        buildCoords = move.buildCoords

        prev_row, prew_col = fromCoords
        new_row, new_col = toCoords
        build_row, build_col = buildCoords

        if fromCoords == self.black1:
            self.black1 = toCoords
        elif fromCoords == self.black2:
            self.black2 = toCoords
        elif fromCoords == self.white1:
            self.white1 = toCoords
        elif fromCoords == self.white2:
            self.white2 = toCoords

        self.players[new_row][new_col] = self.players[prev_row][prew_col]
        self.players[prev_row][prew_col] = 0
        self.heights[build_row][build_col] += 1

    def check_if_player_won(self, last_player):     #also returns available moves for next player
        if last_player == "black":
            white_moves, _ = self.find_possible_moves("white")
            if len(white_moves) == 0:
                return True,[]
            black1_row, black1_col = self.black1
            black2_row, black2_col = self.black2
            if self.heights[black1_row][black1_col] == 3:
                return True,[]
            elif self.heights[black2_row][black2_col] == 3:
                return True,[]
            return False, white_moves

        else:
            black_moves, _ = self.find_possible_moves("black")
            if len(black_moves) == 0:
                return True,[]
            white1_row, white1_col = self.white1
            white2_row, white2_col = self.white2
            if self.heights[white1_row][white1_col] == 3:
                return True,[]
            elif self.heights[white2_row][white2_col] == 3:
                return True,[]
            return False, black_moves
