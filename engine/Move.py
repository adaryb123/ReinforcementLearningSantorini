"""
Object representing one player's turn
Author: Adam Rybansky (xryban00)
FIT VUT 2023
"""

class Move:
    def __init__(self, fromCoods, toCoords, buildCoords, player):
        self.fromCoords = fromCoods
        self.toCoords = toCoords
        self.buildCoords = buildCoords
        self.player = player

    def __str__(self):
        return self.coords_to_text(self.fromCoords) + " -> " + self.coords_to_text(self.toCoords) + " build: " + self.coords_to_text(self.buildCoords)

    def coords_to_text(self,coords):
        row, col = coords
        return str(row + 1) + chr(ord('A') + col)


def text_to_coords(text):
    if len(text) != 2:
        return False,""

    row = int(text[0]) - 1
    col = int(ord(text[1]) - ord('A'))

    if row < 0 or row > 4 or col < 0 or col > 4:
        return False,""
    else:
        return True, (row,col)