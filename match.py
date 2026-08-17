from grid import Grid  # get the grid class from the grid.py file


class Match:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.grid = Grid(7, 6)  # standard Connect Four grid size

        self.coins = {self.player1 : "R", self.player2 : "B"}
    def play(self):
        current_player = self.player1
        while True:
            print(f"{current_player}'s turn")
            availble_columns = self.grid.get_nonempty_columns()
            column = int(input(f"Choose a column ({availble_columns}): "))
            try:
                self.grid.insert_coin(column, self.coins[current_player])
                self.grid.display()
                if self.check_winner(current_player):
                    print(f"{current_player} wins!")
                    break
                current_player = self.player2 if current_player == self.player1 else self.player1
            except IndexError as e:
                print(e)

    def check_winner(self, player):
        # Check for a winning condition (4 in a row)
        last_val = 0
        for row in range(self.grid.height):
            counter = 0
            for col in range(self.grid.width):
                #check rows
                val = self.grid.get_value(col, row)
                if val == last_val and val != 0:
                    counter += 1
                    if counter >= 4:
                        print(f"Found a winning condition for {val} at row {row}, col {col}")
                        return True
                elif val != last_val and val !=0:
                    counter = 1
                    last_val = val
                else:
                    counter = 0
                    last_val = 0
                
        last_val_v = 0
        for col in range(self.grid.width):
            counter_v = 0
            for row in range(self.grid.height):
                # check columns
                val_v = self.grid.get_value(col, row)
                if val_v == last_val_v and val_v != 0:
                    counter_v += 1
                    if counter_v >= 4:
                        print(f"Found a winning condition for {val_v} at row {col}, col {row}")
                        return True
                elif val_v != last_val_v and val_v !=0:
                    counter_v = 1
                    last_val_v = val_v
                else:
                    counter_v = 0
                    last_val_v = 0

        # need to figure out a diagonal check now
        for col in range(self.grid.width):
            for row in range(self.grid.height):
                # check diagonals
                val_d = self.grid.get_value(col, row)
                if val_d != 0:
                    # check diagonal down-right
                    counter_d = 1
                    for i in range(1, 4):
                        if col + i < self.grid.width and row + i < self.grid.height:
                            if self.grid.get_value(col + i, row + i) == val_d:
                                counter_d += 1
                            else:
                                break
                    if counter_d >= 4:
                        print(f"Found a winning condition for {val_d} at row {row}, col {col} (diagonal down-right)")
                        return True

                    # check diagonal up-right
                    counter_u = 1
                    for i in range(1, 4):
                        if col + i < self.grid.width and row - i >= 0:
                            if self.grid.get_value(col + i, row - i) == val_d:
                                counter_u += 1
                            else:
                                break
                    if counter_u >= 4:
                        print(f"Found a winning condition for {val_d} at row {row}, col {col} (diagonal up-right)")
                        return True
        # check for draw due to full grid
        if self.grid.get_nonempty_columns() == []:
            print("The game is a draw!")
            return True
        return False 


match = Match("Player 1", "Player 2")
match.play()