from engine.Board import Board
from engine.Move import Move


class Bot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board, availableMoves):
        return availableMoves[0]