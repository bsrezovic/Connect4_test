import random

from agent1 import Agent

from match import Match

episodes = 0
done = False
frames = []  # useed to store select game states
moveorder = []
results = {"win": 0, "draw": 0, "loss": 0}

win_rates = []
draw_rates = []
loss_rates = []

# first lets try to make agent play a random bot
# lets assign who is which player randomly each episode, then 
botorder = 0
match = Match("Player 1", "Player 2")
agent = Agent(alpha=0.1, gamma=0.9, epsilon=0.1)
players = ["Player 2", "Player 1"]
agent_player = "Player 1" # need to initialize this so my loop works as intended
while not done:
    if match.IsOver:
        # i guess here we would see who won and learn from it, but for now we will just reset the game
        #print(f"Game over! Winner: {match.winner}")
        frames.append(f"Game over! Winner: {match.winner}")
        
        # collect some stats
        if match.winner == "No one won yet":
            results["draw"] += 1
        elif match.winner == agent_player:
            results["win"] += 1
        else:
            results["loss"] += 1
        if (episodes + 1) % 100 == 0:
            total = sum(results.values())
            win_rates.append(results["win"] / total)
            draw_rates.append(results["draw"] / total)
            loss_rates.append(results["loss"] / total)
            print(f"Episode {episodes+1}: Wins {results['win']}, Draws {results['draw']}, Losses {results['loss']}")
            results = {"win": 0, "draw": 0, "loss": 0}
            frames.append(moveorder)
        # moved episode counter here so that we play full games n times
        episodes += 1 
        # start new match
        moveorder = []
        match = Match("Player 1", "Player 2")
    if match.turn == 1: # if its the first turn we have to dedide is the random bot player 1 or player 2, and then we will have to make the agent play the other player
        if random.random() < 0.5:
            botorder = 0
            agent_player = "Player 1"
            bot_player = "Player 2"
        else:
            botorder = 1
            agent_player = "Player 2"
            bot_player = "Player 1"

    moves = match.get_available_columns()
    
    if match.turn % 2 == botorder:
        move = random.choice(moves)
        match.play_bot(move)
    else:
        state = tuple(map(tuple, match.grid.get_grid()))  # convert the grid to a tuple of tuples for hashing
        action = agent.choose_action(state, moves)
        match.play_bot(action)
        next_state = tuple(map(tuple, match.grid.get_grid()))
        # distribute rewards based on the outcome of the game
        # it basically wont update before we have some game wins on our hands
        reward = 1 if match.winner == agent_player else -1 if match.winner == bot_player else 0
        next_moves = match.get_available_columns()
        agent.update(state, action, reward, next_state, next_moves)
    
    moveorder.append(match.grid.get_grid())
    
    if episodes == 1000:
        break



#agent.print_q_table()