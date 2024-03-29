"""
Heuritsic bot - more friendly version
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

import copy

class HeuristicBot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board, availableMoves=None):
        if availableMoves is None:
            availableMoves = board.find_possible_moves(self.color)
        best_move = availableMoves[0]
        best_score = -100
        for i in range(len(availableMoves)):
            boardCopy = copy.deepcopy(board)
            boardCopy.update_board_after_move(availableMoves[i])
            score = boardCopy.get_player_height(self.color)
            if score > best_score:
                best_score = score
                best_move = availableMoves[i]
        return best_move

