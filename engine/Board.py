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
        output = "    |   A   |   B   |   C   |   D   |   E   |\n"
        output += "____|_______|_______|_______|_______|_______|\n"
        for i in range(5):
            output += str(i+1) + "  "
            for j in range(5):
                output += " | " + self.tiles[i][j].__str__()
            output += " |\n____|_______|_______|_______|_______|_______|\n"
        return output

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
            if self.tiles[row][col].player == 0:
                self.tiles[row][col].player = current
                finished += 1
                if finished == 1:
                    black1 = (row, col)
                elif finished == 2:
                    current = 1
                    black2 = (row, col)
                elif finished == 3:
                    white1 = (row, col)
                elif finished == 4:
                    white2 = (row, col)

        return black1, black2, white1, white2


    def find_possible_moves(self, player):
        moves = []
        if player == "black":
            row, col = self.black1
        else:
            row, col = self.white1

        currentLevel = self.tiles[row][col].level
        toCoordsList = [(row,col-1), (row,col+1), (row-1,col-1), (row-1,col), (row-1,col+1), (row+1,col-1), (row+1,col), (row+1,col+1)]
        for i, j in toCoordsList:
            if self.check_valid_for_move(i, j, currentLevel):
                moves.extend(self.add_all_builds((row, col), (i, j), player))

        if player == "black":
            row, col = self.black2
        else:
            row, col = self.white2

        currentLevel = self.tiles[row][col].level
        # clockwise
        toCoordsList = [(row-1,col), (row-1,col+1), (row,col+1), (row+1,col+1), (row+1,col), (row+1,col-1), (row,col-1), (row-1,col-1)]
        for i, j in toCoordsList:
            if self.check_valid_for_move(i, j, currentLevel):
                moves.extend(self.add_all_builds((row, col), (i, j), player))

        return moves

    def check_valid_for_move(self, row, col, currentLevel):
        if 0 <= row < 5 and 0 <= col < 5:
            if self.tiles[row][col].level <= currentLevel + 1:
                if self.tiles[row][col].level != 4 and self.tiles[row][col].player == 0:
                    return True
        return False

    def add_all_builds(self, fromCoords, toCoords, player):
        moves = []
        row, col = toCoords
        moves.append(Move(fromCoords, toCoords, fromCoords, player))   # always can build on the tile from were it came
        # clockwise
        buildCoordsList = [(row-1,col), (row-1,col+1), (row,col+1), (row+1,col+1), (row+1,col), (row+1,col-1), (row,col-1), (row-1,col-1)]
        for i, j in buildCoordsList:
            if self.check_valid_for_build(i, j):
                moves.append(Move(fromCoords, toCoords, (i, j), player))

        return moves

    def check_valid_for_build(self, row, col):
        if 0 <= row < 5 and 0 <= col < 5:
            if self.tiles[row][col].level != 4 and self.tiles[row][col].player == 0:
                return True
        return False

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

        self.tiles[new_row][new_col].player = self.tiles[prev_row][prew_col].player
        self.tiles[prev_row][prew_col].player = 0
        self.tiles[build_row][build_col].level += 1

    def check_if_game_ended(self, last_player):     #also returns available moves for next player
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
            return False,white_moves

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
            return False,black_moves
