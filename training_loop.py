import random

from agent1 import Agent, DQN
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

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000
memory_size = 10000
#episodes = 1000

# first lets try to make agent play a random bot
# lets assign who is which player randomly each episode, then 
botorder = 0
match = Match("Player 1", "Player 2")
#agent = Agent(alpha=0.1, gamma=0.9, epsilon=0.1)
players = ["Player 2", "Player 1"]
agent_player = "Player 1" # need to initialize this so my loop works as intended

# Initialize Q-networks
input_dim = 42
output_dim = 7
policy_net = DQN(input_dim, output_dim)
target_net = DQN(input_dim, output_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size)



#match.play_bot(2)
#match.grid.get_grid()

#test = torch.FloatTensor(match.grid.get_grid())
#test_flat = torch.flatten(test).unsqueeze(0)
#test_flat.shape
#test.shape  # cool that works i guees
# Function to optimize the model using experience replay
def optimize_model():
    #print(f"memory: {len(memory)}")
    
    if len(memory) < batch_size:
       # print("memory smaller than batch size")
        # wait unitl replay memory fills to batch size, we only optimize model when it goes through a batch
        return
    
    # break correlation between sequential
    batch = random.sample(memory, batch_size)
    # unzip into 5 separate tuples
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.stack(state_batch).squeeze(1)
    # this is somehow numpys fault that this extra dim gets added
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.stack(next_state_batch).squeeze(1)
    done_batch = torch.FloatTensor(done_batch)

    # Compute Q-values for current states
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Main training loop
while not done:
    # the match over loop should be the same as in the qlearning version
    #print("Doing a move")
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
    # currently possible moves 0-6
    moves = match.get_available_columns()


    if match.turn % 2 == botorder:   # random bot makes its move
        move = random.choice(moves)
        match.play_bot(move)
    else:

        state = torch.FloatTensor(match.grid.get_grid()).flatten()#tensor, and 
        #adds a dim in front for batch size/pruposes
        q_values = policy_net(state.unsqueeze(0)) # only add the dim here cuz you messed up the optimizer func
        # if the move is unavailible due to full columns it needs to be removed
        mask = torch.full_like(q_values, -float('inf'))  # Shape: [1, 4]
        mask[:, moves] = 0.0  # Important: index the 2nd dimension
        masked_q = q_values + mask
        # select from the possible ones
        move = torch.argmax(masked_q).item() 
        # play the move
        match.play_bot(move)
        # update the model with reward functions etc...
        # ideal code
        #next_state, reward, done, _ = env.step(action)
        next_state = torch.FloatTensor(match.grid.get_grid()).flatten()
        
        if match.winner == agent_player:
            reward = 1
        elif match.winner == bot_player:
            reward = -1
        else:
            reward = 0
        # Store transition in memory
        #print("shapes")
        #print(state.shape)
        #print(next_state.shape)
        memory.append((state.squeeze(0), move, reward, next_state.squeeze(0), done))

        # Optimize model
        optimize_model()

    # remember moves remporarily. save some to list occasionally to check for emergent behaviours
    moveorder.append(match.grid.get_grid())
    
    if episodes == 10000:
        break


#agent.print_q_table()