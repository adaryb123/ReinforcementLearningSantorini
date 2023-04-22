"""
The game board for Santorini
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

from engine.Tile import Tile
from engine.Move import Move
import random

class Board:
    def __init__(self):
        self.tiles = []
        for i in range(5):
            self.tiles.append([Tile(), Tile(), Tile(), Tile(), Tile()])

        self.black1, self.black2, self.white1, self.white2 = self.setup_players()

    def __str__(self):
        output = "                |   A   |   B   |   C   |   D   |   E   |\n"
        output += "            ____|_______|_______|_______|_______|_______|\n"
        for i in range(5):
            output += "            " + str(i+1) + "  "
            for j in range(5):
                output += " | " + self.tiles[i][j].__str__()
            output += " |\n            ____|_______|_______|_______|_______|_______|\n"
        return output

    def setup_players(self):
        """ Initial setup of players is random"""
        finished = 0
        current = -1
        black1 = 0
        black2 = 0
        white1 = 0
        white2 = 0
        while finished < 4:
            row = random.randint(1,3)
            col = random.randint(1,3)
            if self.tiles[row][col].player == 0:
                self.tiles[row][col].player = current
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
        """ There are 7 ways a move can be illegal, check them all here"""
        row_from, col_from = move.fromCoords
        row_to, col_to = move.toCoords
        row_build, col_build = move.buildCoords

        if 0 <= row_to < 5 and 0 <= col_to < 5:
            if 0 <= row_build < 5 and 0 <= col_build < 5:
                current_tile = self.tiles[row_from][col_from]
                move_target = self.tiles[row_to][col_to]
                build_target = self.tiles[row_build][col_build]
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
    def find_possible_moves(self, player):
        moves = []

        """Find all legal moves of the first worker"""
        if player == "black":
            from_row, from_col = self.black1
        else:
            from_row, from_col = self.white1

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        for to_row, to_col in toCoordsList:
            buildCoordsList = self.make_clockwise_list(to_row, to_col)
            for build_row, build_col in buildCoordsList:
                move = Move((from_row, from_col), (to_row, to_col), (build_row, build_col), player)
                valid, _ = self.check_move_valid(move)
                if valid:
                    moves.append(move)

        """Find all legal moves of the second worker"""
        if player == "black":
            from_row, from_col = self.black2
        else:
            from_row, from_col = self.white2

        toCoordsList = self.make_clockwise_list(from_row, from_col)
        for to_row, to_col in toCoordsList:
            buildCoordsList = self.make_clockwise_list(to_row, to_col)
            for build_row, build_col in buildCoordsList:
                move = Move((from_row, from_col), (to_row, to_col), (build_row, build_col), player)
                valid, _ = self.check_move_valid(move)
                if valid:
                    moves.append(move)


        return moves

    def create_move_from_number(self, number, player_color):
        """Number is converted to the corresponding move based on the clockwise list"""
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
        to_row, to_col = toCoordsList[number // 8]
        buildCoordsList = self.make_clockwise_list(to_row, to_col)
        build_row, build_col = buildCoordsList[number % 8]

        return Move((from_row, from_col), (to_row, to_col), (build_row, build_col), player_color)

    def update_board_after_move(self, move):
        """Execute the move on the game board"""
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

        self.tiles[new_row][new_col].player = self.tiles[prev_row][prew_col].player
        self.tiles[prev_row][prew_col].player = 0
        self.tiles[build_row][build_col].level += 1

    def check_if_player_won(self, last_player):
        """There are 2 waysa player can win. Check them both.
        Function also returns all possible moves of the next player"""
        if last_player == "black":
            white_moves = self.find_possible_moves("white")
            if len(white_moves) == 0:
                return True,[]
            black1_row, black1_col = self.black1
            black2_row, black2_col = self.black2
            if self.tiles[black1_row][black1_col].level == 3:
                return True,[]
            elif self.tiles[black2_row][black2_col].level == 3:
                return True,[]
            return False, white_moves

        else:
            black_moves = self.find_possible_moves("black")
            if len(black_moves) == 0:
                return True,[]
            white1_row, white1_col = self.white1
            white2_row, white2_col = self.white2
            if self.tiles[white1_row][white1_col].level == 3:
                return True,[]
            elif self.tiles[white2_row][white2_col].level == 3:
                return True,[]
            return False, black_moves

    def get_player_height(self, player_color):
        """Sum total height of the player's workers"""
        height = 0
        for i in range(5):
            for j in range(5):
                if player_color == "white":
                    if self.tiles[i][j].player >= 1:
                        height += self.tiles[i][j].level
                elif player_color == "black":
                    if self.tiles[i][j].player <= -1:
                        height += self.tiles[i][j].level

        return height

    def get_player_height_diff(self, player_color):
        """Sum total height of the player's workers, subtracted by the height of the enemy player's workers"""
        height_diff = 0
        for i in range(5):
            for j in range(5):
                if self.tiles[i][j].player >= 1:
                    height_diff += self.tiles[i][j].level
                elif self.tiles[i][j].player <= -1:
                    height_diff -= self.tiles[i][j].level
        if player_color == "black":
            height_diff *= -1

        return height_diff

def encode_board(board):
    """Create tensor from Board object"""
    inputTensor = []
    board_heights = []
    board_players = []
    valid_tiles = [[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1],[1,1,1,1,1]]

    for i in board.tiles:
        row_heights = []
        row_players = []

        for j in i:
            row_heights.append(j.level)
            row_players.append(j.player)

        board_heights.append(row_heights)
        board_players.append(row_players)

    inputTensor.append(board_heights)
    inputTensor.append(board_players)
    inputTensor.append(valid_tiles)
    return inputTensor

def decode_board(inputTensor):
    """Create Board object from tensor"""
    board = Board()
    for i in range(len(board.tiles)):
        for j in range(len(board.tiles)):
            board.tiles[i][j].level = inputTensor[0, i, j]
            player = inputTensor[1, i, j]
            board.tiles[i][j].player = player
            if player == -1:
                board.black1 = (i,j)
            elif player == -2:
                board.black2 = (i,j)
            elif player == 1:
                board.white1 = (i,j)
            elif player == 2:
                board.white2 = (i,j)
    return board