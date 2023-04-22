"""
Random bot
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

import random

class RandomBot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board):
        availableMoves = board.find_possible_moves(self.color)
        index = random.randint(0, len(availableMoves) - 1)
        return availableMoves[index]