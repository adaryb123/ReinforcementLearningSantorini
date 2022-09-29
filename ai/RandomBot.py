from engine.Board import Board
from engine.Move import Move
import random

class RandomBot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board, availableMoves):
        index = random.randint(0, len(availableMoves) - 1)
        return availableMoves[index]