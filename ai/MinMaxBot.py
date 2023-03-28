from engine.Board import Board
import copy


class MinMaxBot:
    def __init__(self, color, depth):
        self.color = color
        self.depth = depth

    def make_turn(self, board, availableMoves=None):
        if availableMoves is None:
            availableMoves, _ = board.find_possible_moves(self.color)
        _,move = self.max(self.depth, -99999, 99999, board, availableMoves, self.color)
        return move

    def get_next_player(self, current_player):
        if current_player == "black":
            return "white"
        else:
            return "black"

    def max(self, depth, alpha, beta, current_board, available_moves, current_player):
        next_player = self.get_next_player(current_player)
        bestMove = available_moves[0]
        for move in available_moves:
            boardCopy = copy.deepcopy(current_board)
            boardCopy.update_board_after_move(move)
            win, next_player_moves = boardCopy.check_if_player_won(current_player)
            if win:
                score = 100
            elif depth == 0:
                score = self.evaluate(boardCopy, current_player)
            else:
                score, _ = self.min(depth - 1, alpha, beta, boardCopy, next_player_moves, next_player)

            if score >= beta:
                return beta,bestMove

            if score > alpha:
                alpha = score
                bestMove = move

        return alpha, bestMove

    def min(self,depth,alpha,beta,current_board,available_moves,current_player):
        next_player = self.get_next_player(current_player)
        bestMove = available_moves[0]
        for move in available_moves:
            boardCopy = copy.deepcopy(current_board)
            boardCopy.update_board_after_move(move)
            win, next_player_moves = boardCopy.check_if_player_won(current_player)
            if win:
                score = -100
            elif depth == 0:
                score = -self.evaluate(boardCopy, current_player)
            else:
                score, _ = self.max(depth - 1, alpha, beta, boardCopy, next_player_moves, next_player)

            if score <= alpha:
                return alpha,bestMove

            if score < beta:
                beta = score
                bestMove = move

        return beta, bestMove

    def evaluate(self, board, player):
        score = 0
        rowb1, colb1 = board.black1
        rowb2, colb2 = board.black2
        roww1, colw1 = board.white1
        roww2, colw2 = board.white2
        score += board.heights[rowb1][colb1]
        score += board.heights[rowb2][colb2]
        score -= board.heights[roww1][colw1]
        score -= board.heights[roww2][colw2]

        if player == "black":
            return score
        else:
            return -score
