from engine.Board import Board
from engine.Move import Move
from engine.Tile import Tile
from ai.RandomBot import RandomBot
from ai.MinMaxBot import MinMaxBot
from datetime import datetime

game_board = Board()
white_moves = []
black_moves = game_board.find_possible_moves("black")

blackBot = MinMaxBot("black")
whiteBot = MinMaxBot("white")
# blackBot = RandomBot("black")
# whiteBot = RandomBot("white")


print("starting board")
print(game_board)
total_thinking_time = 0
moves = 0
while True:
    # user_input = input("Continue? Y/N\n")
    # if user_input == "N" or user_input == "n":
    #     break

    print("black turn")
    start_time = datetime.now()
    move = blackBot.make_turn(game_board,black_moves)
    end_time = datetime.now()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time.seconds
    moves += 1
    game_board.update_board_after_move(move)
    print(game_board)
    print("black picked move: " + move.__str__())
    print("thinking time: "+str(thinking_time.seconds))
    game_end, white_moves = game_board.check_if_game_ended("black")
    if game_end:
        print("black won!")
        print("total thinking time:" + str(total_thinking_time))
        print("total moves: " + str(moves))
        print("average thinking time: " + str(total_thinking_time / moves))
        break

    # user_input = input("Continue? Y/N\n")
    # if user_input == "N" or user_input == "n":
    #     break

    print("white turn")
    start_time = datetime.now()
    move = whiteBot.make_turn(game_board, white_moves)
    end_time = datetime.now()
    thinking_time = end_time - start_time
    total_thinking_time += thinking_time.seconds
    moves += 1
    game_board.update_board_after_move(move)
    print(game_board)
    print("white picked move: " + move.__str__())
    print("thinking time: " + str(thinking_time.seconds))
    game_end, black_moves = game_board.check_if_game_ended("white")
    if game_end:
        print("white won!")
        print("total thinking time:" +str(total_thinking_time))
        print("total moves: " + str(moves))
        print("average thinking time: " + str(total_thinking_time / moves))
        break
