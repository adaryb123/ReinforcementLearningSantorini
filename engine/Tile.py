"""
One tile of the game board
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

class Tile:
    def __init__(self):
        self.level = 0      # can be 0,1,2,3,4.   4 means dome
        self.player = 0     # 0 means empty, -1 and -2 means black, 1 and 2 means white,

    def __str__(self):
        output = ""
        spaces = 5
        if self.level == 0:
            spaces = 5
        if self.level == 1:
            output += "C"
            spaces = 4
        elif self.level == 2:
            output += "CC"
            spaces = 3
        elif self.level == 3:
            output += "CCC"
            spaces = 2
        elif self.level == 4:
            output += "CCCD"
            spaces = 1

        if self.player <= -1:
            output += "_X"
            spaces -= 2
        elif self.player >= 1:
            output += "_0"
            spaces -= 2

        for i in range(spaces):
            output += " "

        return output
