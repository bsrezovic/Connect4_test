
import numpy as np
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
    def get_grid(self):
        char_to_int = {
            0: 0,   # empty
            'R': 1,   # player 1
            'B': 2,   # player 2
        }
        numeric_grid = [[char_to_int[cell] for cell in row] for row in self.grid]
        return np.array(numeric_grid, dtype=np.float32) # Return a copy of the grid
    def display(self):
        for row in self.grid:
            print(' '.join(str(cell) for cell in row))
    def insert_coin(self,column,value):
        if column < 0 or column >= self.width:
            raise IndexError("Column out of bounds")
        if column not in self.get_nonempty_columns():
            raise IndexError("Column is full already")
        # check for the first empty cell in the column from the bottom up
        for row in reversed(range(self.height)):
            if self.grid[row][column] == 0:
                self.grid[row][column] = value
                return True
        return False  # if some error occurs
    def define_state(self):
        # convert the grid to a state vector for the qlearning agent
        state_vector = []
        for row in self.grid:
            for cell in row:
                state_vector.append(cell)
        return tuple(state_vector)  # return as a tuple for hashing

#c4grid = Grid(7, 6)

#c4grid.insert_coin(3, "B")

#print(c4grid.display())

#c4grid.get_grid()

#print(c4grid.get_nonempty_columns()) 

#c4grid.get_value(0, 0)

