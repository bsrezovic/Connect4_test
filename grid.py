
# just keep the grid in this file, or in the future add the visual interface here
class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]

    def set_value(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = value
        else:
            raise IndexError("Coordinates out of bounds")

    def get_value(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        else:
            raise IndexError("Coordinates out of bounds")

    def display(self):
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))


c4grid = Grid(7, 6)

print(c4grid.display())