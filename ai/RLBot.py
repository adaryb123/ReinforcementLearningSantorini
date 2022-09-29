from engine.Board import Board
from engine.Move import Move


class RLBot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board, availableMoves):
        return availableMoves[0]

    def encode_input(self):
        return True

    def decode_output(self, output, board):
        row1 = 0
        col1 = 0
        row2 = 0
        col2 = 0
        moves = []
        coordsList = []
        coordsList2 = []
        if self.color == "black":
            row1, col1 = board.black1
            row2, col2 = board.black2
        else:
            row1, col1 = board.white1
            row2, col2 = board.white2

        # # clockwise
        # coordsList = [(row1 - 1, col1), (row1 - 1, col1 + 1), (row1, col1 + 1), (row1 + 1, col1 + 1), (row1 + 1, col1),
        #               (row1 + 1, col1 - 1), (row1, col1 - 1), (row1 - 1, col1 - 1)]
        # coordsList2 = [(row2 - 1, col2), (row2 - 1, col2 + 1), (row2, col2 + 1), (row2 + 1, col2 + 1), (row2 + 1, col2),
        #               (row2 + 1, col2 - 1), (row2, col2 - 1), (row2 - 1, col2 - 1)]
        toCoordsList = []
        buildCoordsList = []
        offset = 0

        for i in range(output):
            fromCoords = ()
            toCoords = ()
            buildCoords = ()

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

            moves.append(Move(fromCoords,toCoords,buildCoords,self.color))
            # teraz sa zahodia neplatnne tahy, vektor sa znormalizuje a vyberie sa najlepsi (epsilon nieco) tah

    def make_clockwise_list(self,row, col):
        coordsList = [(row1 - 1, col1), (row1 - 1, col1 + 1), (row1, col1 + 1), (row1 + 1, col1 + 1),
                      (row1 + 1, col1), (row1 + 1, col1 - 1), (row1, col1 - 1), (row1 - 1, col1 - 1)]
        return coordsList
