import tkinter as tk
import threading
import time

class MatchViewer:
    """
    A tkinter window that displays the current state of a Match object's grid.
    It does not interfere with the Match logic; it just polls the grid and redraws.
    """
    def __init__(self, match, refresh_ms=100):
        self.match = match
        self.refresh_ms = refresh_ms

        self.root = tk.Tk()
        self.root.title("Connect Four Viewer")

        # Canvas size based on grid dimensions
        self.cell_size = 80
        self.radius = 30
        self.margin = 40
        canvas_width = self.margin * 2 + match.grid.width * self.cell_size
        canvas_height = self.margin * 2 + match.grid.height * self.cell_size

        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg="lightgray")
        self.canvas.pack()

        self.status_label = tk.Label(self.root, text="Game in progress...", font=("Arial", 14))
        self.status_label.pack(pady=10)

        # Start periodic update
        self.update_board()

    def draw_board(self):
        """Clear canvas and draw grid + coins from match.grid."""
        self.canvas.delete("all")
        grid = self.match.grid

        for row in range(grid.height):
            for col in range(grid.width):
                x0 = self.margin + col * self.cell_size
                y0 = self.margin + row * self.cell_size
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size

                # Cell background
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="lightblue", outline="black")

                # Coin
                val = grid.get_value(col, row)
                color = "white"
                if val == "R":
                    color = "red"
                elif val == "B":
                    color = "blue"
                self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10,
                                        fill=color, outline="black")

        # Update status label if game is over
        if self.match.IsOver:
            if self.match.winner and self.match.winner != "No one won yet":
                self.status_label.config(text=f"{self.match.winner} wins!")
            else:
                self.status_label.config(text="Draw!")
        else:
            self.status_label.config(text="Game in progress...")

    def update_board(self):
        """Called periodically by tkinter to refresh the display."""
        self.draw_board()
        # Schedule next update
        self.root.after(self.refresh_ms, self.update_board)

    def start(self):
        """Start the tkinter main loop."""
        self.root.mainloop()