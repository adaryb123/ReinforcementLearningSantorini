from engine.Board import Board
from engine.Move import Move
from engine.Tile import Tile
from ai.RandomBot import RandomBot
from ai.MinMaxBot import MinMaxBot

game_board = Board()
white_moves = []
black_moves = game_board.find_possible_moves("black")

blackBot = MinMaxBot("black")
whiteBot = MinMaxBot("white")

print("starting board")
print(game_board)
while True:
    user_input = input("Continue? Y/N\n")
    if user_input == "N" or user_input == "n":
        break

    print("black turn")
    move = blackBot.make_turn(game_board,black_moves)
    game_board.update_board_after_move(move)
    print(game_board)
    print("black picked move: " + move.__str__())
    game_end, white_moves = game_board.check_if_game_ended("black")
    if game_end:
        print("black won!")
        break

    user_input = input("Continue? Y/N\n")
    if user_input == "N" or user_input == "n":
        break

    print("white turn")
    move = whiteBot.make_turn(game_board, white_moves)
    game_board.update_board_after_move(move)
    print(game_board)
    print("white picked move: " + move.__str__())
    game_end, black_moves = game_board.check_if_game_ended("white")
    if game_end:
        print("white won!")
        break
