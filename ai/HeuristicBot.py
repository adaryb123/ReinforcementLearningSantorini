from engine.Board import Board
import copy

class HeuristicBot:
    def __init__(self, color):
        self.color = color

    def make_turn(self, board, availableMoves=None):
        if availableMoves is None:
            availableMoves, _ = board.find_possible_moves(self.color)
        best_move = availableMoves[0]
        best_score = -100
        for i in range(len(availableMoves)):
            boardCopy = copy.deepcopy(board)
            boardCopy.update_board_after_move(availableMoves[i])
            score = self.get_player_height(boardCopy, self.color)
            if score > best_score:
                best_score = score
                best_move = availableMoves[i]
        return best_move



    def get_player_height(self, board, player_color):
        height = 0
        for i in range(5):
            for j in range(5):
                if player_color == "white":
                    if board.players[i][j] >= 1:
                        height += board.heights[i][j]
                elif player_color == "black":
                    if board.players[i][j] <= -1:
                        height += board.heights[i][j]

        return height