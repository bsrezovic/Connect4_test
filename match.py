from grid import Grid  # get the grid class from the grid.py file
from match_gui import MatchViewer
import threading
import random
import time
import pickle
class Match:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2
        self.grid = Grid(7, 6)  # standard Connect Four grid size

        self.coins = {self.player1 : "R", self.player2 : "B"}
        self.IsOver = False
        self.winner = "No one won yet"
        self.turn = 1
    def play_vs_bot(self,agent):
        # first decide and announce who goes first 
        if self.turn == 1: # if its the first turn we have to dedide is the random bot player 1 or player 2, and then we will have to make the agent play the other player
            if random.random() < 0.5:
                #botorder = 0
                print("You will go first!")
                human_player = self.player1
                bot_player = self.player2
            else:
                print("The agent will go first")
                #botorder = 1
                human_player = self.player2
                bot_player = self.player1
        
        current_player = self.player1
        while True:
            availble_columns = self.grid.get_nonempty_columns()
            print(f"Its now turn {self.turn}")
            print(f"{current_player}'s turn")
            if current_player == human_player:
                # human player input loop
                column = int(input(f"Choose a column ({availble_columns}): "))
                try:
                    self.grid.insert_coin(column, self.coins[current_player])
                    print("Current boardstate:")
                    self.grid.display()
                    if self.check_winner(current_player):
                        print(f"{current_player} wins!")
                        self.IsOver = True
                        self.winner = current_player
                        break
                    current_player = self.player2 if current_player == self.player1 else self.player1
                    self.turn +=1
                except IndexError as e:
                    print(e)
            else:
                # agent playet input loop
                state = self.grid.get_grid()
                chosen_move = agent.choose_action(state,availble_columns)
                self.grid.insert_coin(chosen_move, self.coins[current_player])
                print("Current boardstate:")
                self.grid.display()
                if self.check_winner(current_player):
                    print(f"{current_player} wins!")
                    self.IsOver = True
                    self.winner = current_player
                    break
                current_player = self.player2 if current_player == self.player1 else self.player1
                self.turn +=1

    def play(self):
        self.turn += 1
        current_player = self.player1
        while True:
            print(f"{current_player}'s turn")
            availble_columns = self.grid.get_nonempty_columns()
            column = int(input(f"Choose a column ({availble_columns}): "))
            try:
                self.grid.insert_coin(column, self.coins[current_player])
                print("Current boardstate:")
                self.grid.display()
                if self.check_winner(current_player):
                    print(f"{current_player} wins!")
                    self.IsOver = True
                    self.winner = current_player
                    break
                current_player = self.player2 if current_player == self.player1 else self.player1
            except IndexError as e:
                print(e)
    # method to get availible columns for the bot to choose from
    def get_available_columns(self):
        return self.grid.get_nonempty_columns()
    # play loop for bots
    def play_bot(self, choice):  # non user input type of playing
        
        if self.turn % 2 == 1:
            current_player = self.player1
        else:
            current_player = self.player2
        #print(f"{current_player}'s turn")
        self.turn += 1
        try:
            self.grid.insert_coin(choice, self.coins[current_player])
            if self.check_winner(current_player):
                #print(f"{current_player} wins!")
                self.winner = current_player
                self.IsOver = True
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
                        #print(f"Found a winning condition for {val} at row {row}, col {col}")
                        self.IsOver = True
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
                        #print(f"Found a winning condition for {val_v} at row {col}, col {row}")
                        self.IsOver = True
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
                        #print(f"Found a winning condition for {val_d} at row {row}, col {col} (diagonal down-right)")
                        self.IsOver = True
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
                        #print(f"Found a winning condition for {val_d} at row {row}, col {col} (diagonal up-right)")
                        return True
        # check for draw due to full grid
        if self.grid.get_nonempty_columns() == []:
            #print("The game is a draw!")
            self.IsOver = True
            return True
        return False 

with open("agent_v1.pkl", "rb") as file:
    loaded_agent = pickle.load(file)

if __name__ == "__main__":
    game = Match("Player 1", "Player 2")
    viewer = MatchViewer(game)

    # Run the original play() in a thread
    game_thread = threading.Thread(target=game.play_vs_bot,args =(loaded_agent,), daemon=True)
    game_thread.start()

    viewer.start()