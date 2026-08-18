from match import Match
import random


epochs = 0
done = False
frames = []

match = Match("Player 1", "Player 2")
while not done:
    if match.IsOver:
        # i guess here we would see who won and learn from it, but for now we will just reset the game
        print(f"Game over! Winner: {match.winner}")
        frames.append(f"Game over! Winner: {match.winner}")
        match = Match("Player 1", "Player 2")

    moves = match.get_available_columns()
    move = random.choice(moves)
    match.play_bot(move)

    frames.append(match.grid.get_grid())

    epochs += 1
    if epochs == 100:
        break