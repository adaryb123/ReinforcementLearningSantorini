from engine.Board import Board
from engine.Move import Move


class RLBot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board, availableMoves):
        pass

    def encode_input(self, board):
        input = []
        for i in board.tiles:
            for j in i:
                if self.color == "black":
                    input.append([j.level, - j.player])
                elif self.color == "white":
                    input.append([j.level, j.player])

        return input

    def decode_output(self, output, board):
        row1 = 0
        col1 = 0
        row2 = 0
        col2 = 0
        moves = []
        if self.color == "black":
            row1, col1 = board.black1
            row2, col2 = board.black2
        else:
            row1, col1 = board.white1
            row2, col2 = board.white2

        fromCoords = ()
        toCoordsList = []
        buildCoordsList = []
        offset = 0

        for i in range(output):
            if i == 0:
                offset = 0
                fromCoords = (row1, col1)
                toCoordsList = self.make_clockwise_list(row1,col1)
            elif i == 64:
                offset = 64
                fromCoords = (row2, col2)
                toCoordsList = self.make_clockwise_list(row1, col1)
            index = i - offset
            toCoords = toCoordsList[int(index / 8)]

            if i % 8 == 0:
                row_to, col_to = toCoords
                buildCoordsList = self.make_clockwise_list(row_to, col_to)
            buildCoords = buildCoordsList[int(i % 8)]

            # teraz sa zahodia neplatnne tahy, vektor sa znormalizuje a vyberie sa najlepsi (epsilon nieco) tah
            potential_move = Move(fromCoords,toCoords,buildCoords,self.color)
            if self.check_move_valid(potential_move,board):
                moves.append(potential_move)
            else:
                output[i] = 0



    def make_clockwise_list(self,row, col):
        coordsList = [(row1 - 1, col1), (row1 - 1, col1 + 1), (row1, col1 + 1), (row1 + 1, col1 + 1),
                      (row1 + 1, col1), (row1 + 1, col1 - 1), (row1, col1 - 1), (row1 - 1, col1 - 1)]
        return coordsList

    def check_move_valid(self, move, board):
        row_from, col_from = move.fromCoords
        row_to, col_to = move.toCoords
        row_build, col_build = move.buildCoords
        current_tile = board.tiles[row_from][col_from]
        move_target = board.tiles[row_to][col_to]
        build_target = board.tiles[row_build][col_build]

        if 0 <= row_to < 5 and 0 <= col_to < 5 and 0 <= row_build < 5 and 0 <= col_build < 5:
            if move_target.player == 0 and build_target.player == 0:
                if move_target.level != 4 and build_target.level != 4:
                    if move_target.level <= current_tile.level + 1:
                        return True
        return False
