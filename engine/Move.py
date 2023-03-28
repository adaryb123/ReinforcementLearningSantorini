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


