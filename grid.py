
# just keep the grid in this file, or in the future add the visual interface here
class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[0 for _ in range(width)] for _ in range(height)]
    def get_value(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        else:
            raise IndexError("Coordinates out of bounds")

    def get_nonempty_columns(self):
        nonempty_columns = []
        for col in range(self.width):
            if any(self.grid[row][col] == 0 for row in range(self.height)):
                nonempty_columns.append(col)
        return nonempty_columns

    def display(self):
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))
    def insert_coin(self,column,value):
        if column < 0 or column >= self.width:
            raise IndexError("Column out of bounds")
        # check for the first empty cell in the column from the bottom up
        for row in reversed(range(self.height)):
            if self.grid[row][column] == 0:
                self.grid[row][column] = value
                return True
        return False  # Column is full



c4grid = Grid(7, 6)

c4grid.insert_coin(4, "R")

print(c4grid.display())

print(c4grid.get_nonempty_columns()) 