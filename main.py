from engine.Board import Board
from engine.Move import Move, text_to_coords
from ai.RandomBot import RandomBot
from ai.MinMaxBot import MinMaxBot
from ai.HeuristicBot import HeuristicBot
from ai.HeuristicCompetitiveBot import HeuristicCompetitiveBot
from ai.RLBot import RLBot
import random

def get_input_for_player(color):
    number = input("Who will be the " + color + " player? \n" +
                   "Type 0 for human\n" +
                   "Type 1 for Random bot\n" +
                   "Type 2 for Heuristic bot\n" +
                   "Type 3 for Heuristic competitive bot\n" +
                   "Type 4 for Minmax bot\n" +
                   "Type 5 for RL bot\n")
    number = int(number)
    if color == "black":
        if number == 0:
            blackPlayer = "HUMAN"
        elif number == 1:
            blackPlayer = RandomBot(color)
        elif number == 2:
            blackPlayer = HeuristicBot(color)
        elif number == 3:
            blackPlayer = HeuristicCompetitiveBot(color)
        elif number == 4:
            blackPlayer = MinMaxBot(color)
        elif number == 5:
            seed = input("Type RL mode name. Make sure the model is stored in train/models:")
            blackPlayer = RLBot(color,seed)
            if not blackPlayer.check_model_file_exists():
                print("ERROR: model doesnt exist")
                return get_input_for_player(color)
        else:
            print("ERROR: invalid option")
            return get_input_for_player(color)
        return blackPlayer

    elif color == "white":
        if number == 0:
           whitePlayer = "HUMAN"
        elif number == 1:
            whitePlayer = RandomBot(color)
        elif number == 2:
            whitePlayer = HeuristicBot(color)
        elif number == 3:
            whitePlayer = HeuristicCompetitiveBot(color)
        elif number == 4:
            whitePlayer = MinMaxBot(color)
        elif number == 5:
            seed = input("Type RL mode name. Make sure the model is stored in train/models:")
            whitePlayer = RLBot(color,seed)
            if not whitePlayer.check_model_file_exists():
                print("ERROR: model doesnt exist")
                return get_input_for_player(color)
        else:
            print("ERROR: invalid option")
            return get_input_for_player(color)
        return whitePlayer

    else:
        print("ERROR: invalid player color")
        exit(0)


def get_input_for_move(board, color):
    print("Write move coordinates. Example: 1A 2B 2A")
    fromCoordsString = input("Select tile to move from:")
    toCoordsString = input("Select tile to move to:")
    buildCoordsString = input("Select tile to build on:")

    fc_valid, fromCoords = text_to_coords(fromCoordsString)
    tc_valid, toCoords = text_to_coords(toCoordsString)
    bc_valid, buildCoords = text_to_coords(buildCoordsString)

    if fc_valid and tc_valid and bc_valid:
        move = Move(fromCoords,toCoords,buildCoords,color)
        valid, msg = board.check_player_input_move_valid(move)
        if not valid:
            print("ERROR: illegal move: " + msg)
            return get_input_for_move(board,color)
        else:
            return move
    else:
        print("ERROR: invalid input format")
        return get_input_for_move(board,color)


blackPlayer = get_input_for_player("black")
whitePlayer = get_input_for_player("white")

game_board = Board()
white_moves = game_board.find_possible_moves("white")
black_moves = game_board.find_possible_moves("black")
starting_player = "black"
if random.randrange(100) < 50:
    starting_player = "white"

print("starting player:" + str(starting_player))
print("starting board")
print(game_board)

if starting_player == "white":
    print("white turn")
    if whitePlayer == "HUMAN":
        move = get_input_for_move(game_board, "white")
    else:
        move = whitePlayer.make_turn(game_board, white_moves)
    game_board.update_board_after_move(move)
    print(game_board)
    print("white picked move: " + move.__str__())

while True:

    print("black turn")
    if blackPlayer == "HUMAN":
        move = get_input_for_move(game_board,"black")
    else:
        move = blackPlayer.make_turn(game_board, black_moves)
    game_board.update_board_after_move(move)
    print(game_board)
    print("black picked move: " + move.__str__())
    game_end, white_moves = game_board.check_if_player_won("black")
    if game_end:
        print("black won!")
        break

    print("white turn")
    if whitePlayer == "HUMAN":
        move = get_input_for_move(game_board, "white")
    else:
        move = whitePlayer.make_turn(game_board, white_moves)
    game_board.update_board_after_move(move)
    print(game_board)
    print("white picked move: " + move.__str__())
    game_end, black_moves = game_board.check_if_player_won("white")
    if game_end:
        print("white won!")
        break
