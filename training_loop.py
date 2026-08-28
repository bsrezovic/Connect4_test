import random
import pickle
from agent1 import Agent, DQN, DeepAgent
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from match import Match

episodes = 0
done = False
frames = []  # useed to store select game states
moveorder = []
results = {"win": 0, "draw": 0, "loss": 0}

win_rates = []
draw_rates = []
loss_rates = []


# lets assign who is which player randomly each episode, then 
botorder = 0
match = Match("Player 1", "Player 2")
#agent = Agent(alpha=0.1, gamma=0.9, epsilon=0.1)
players = ["Player 2", "Player 1"]
agent_player = "Player 1" # need to initialize this so my loop works as intended
agent = DeepAgent()
steps_done = 0
# Main training loop
while not done:
    # the match over loop should be the same as in the qlearning version
    #print("Doing a move")
    if match.IsOver:
        agent.games_played += 1
        frames.append(f"Game over! Winner: {match.winner}")
        
        # collect some stats
        if match.winner == "No one won yet":
            results["draw"] += 1
        elif match.winner == agent_player:
            results["win"] += 1
            agent.games_won += 1
        else:
            results["loss"] += 1
        
        agent.winrate = agent.games_won / agent.games_played
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
        
        # Decay epsilon
        # we start it high ( to encourgae exploration, even as high as 1 to encourage total randomness)
        # then we decay it to the point the model only uses what it has learned
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)


    if match.turn == 1: # if its the first turn we have to dedide is the random bot player 1 or player 2, and then we will have to make the agent play the other player
        if random.random() < 0.5:
            botorder = 0
            agent_player = "Player 1"
            bot_player = "Player 2"
        else:
            botorder = 1
            agent_player = "Player 2"
            bot_player = "Player 1"
    # currently possible moves 0-6
    moves = match.get_available_columns()


    if match.turn % 2 == botorder:   # random bot makes its move
        move = random.choice(moves)
        match.play_bot(move)
    else:

        state = match.grid.get_grid() 
        
        move = agent.choose_action(state,moves)
        # play the moves that the agent decided on 
        match.play_bot(move)

        # update agent memory
        next_state = torch.FloatTensor(match.grid.get_grid()).flatten()
        last_state = torch.FloatTensor(state).flatten()
        if match.winner == agent_player:
            reward = 1
        elif match.winner == bot_player:
            reward = -1
        else:
            reward = 0

        agent.update_memory(last_state,next_state,move,done,reward)

        # Optimize model
        # model is optimized after batch_size batches. but epsilon is updated only episodically
        agent.optimize()
        # Update target network periodically, lets say every 1000 moves played
        # this has a consequence of tying update freq to speed of winning/losing, perhaps unintended but ok
        if steps_done % agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print("target net updated")
        steps_done += 1

    # remember moves remporarily. save some to list occasionally to check for emergent behaviours
    moveorder.append(match.grid.get_grid())
    
    if episodes == 10000:
        break

agent.winrate


# Save the instance to a file
with open("agent_v1.pkl", "wb") as file:
    pickle.dump(agent, file)

# Load the instance back from the file
#with open("player.pkl", "rb") as file:
#    loaded_player = pickle.load(file)