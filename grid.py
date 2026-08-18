
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
        return [row[:] for row in self.grid]  # Return a copy of the grid
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



c4grid = Grid(7, 6)

c4grid.insert_coin(3, "B")

print(c4grid.display())

c4grid.get_grid()

print(c4grid.get_nonempty_columns()) 

c4grid.get_value(0, 0)
# test grid checking logic
#   use the get methiod

# check all rows and columns
last_val = 0
for row in range(c4grid.height):
    counter = 0
    for col in range(c4grid.width):
        #check rows
        val = c4grid.get_value(col, row)
        if val == last_val and val != 0:
            counter += 1
            if counter >= 4:
                print(f"Found a winning condition for {val} at row {row}, col {col}")
                #break
        elif val != last_val and val !=0:
            counter = 1
            last_val = val
        else:
            counter = 0
            last_val = 0
        
last_val_v = 0
for col in range(c4grid.width):
    counter_v = 0
    for row in range(c4grid.height):
        # check columns
        val_v = c4grid.get_value(col, row)
        if val_v == last_val_v and val_v != 0:
            counter_v += 1
            if counter_v >= 4:
                print(f"Found a winning condition for {val_v} at row {col}, col {row}")
                #break
        elif val_v != last_val_v and val_v !=0:
            counter_v = 1
            last_val_v = val_v
        else:
            counter_v = 0
            last_val_v = 0

# need to figure out a diagonal check now
for col in range(c4grid.width):
    for row in range(c4grid.height):
        # check diagonals
        val_d = c4grid.get_value(col, row)
        if val_d != 0:
            # check diagonal down-right
            counter_d = 1
            for i in range(1, 4):
                if col + i < c4grid.width and row + i < c4grid.height:
                    if c4grid.get_value(col + i, row + i) == val_d:
                        counter_d += 1
                    else:
                        break
            if counter_d >= 4:
                print(f"Found a winning condition for {val_d} at row {row}, col {col} (diagonal down-right)")
                #break

            # check diagonal up-right
            counter_u = 1
            for i in range(1, 4):
                if col + i < c4grid.width and row - i >= 0:
                    if c4grid.get_value(col + i, row - i) == val_d:
                        counter_u += 1
                    else:
                        break
            if counter_u >= 4:
                print(f"Found a winning condition for {val_d} at row {row}, col {col} (diagonal up-right)")
                #break


print(c4grid.display())
