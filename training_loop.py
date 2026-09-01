import random
import pickle
from agent1 import dummy_agent, Agent, DQN, DeepAgent, DeepAgentConvolved
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from match import Match

episodes = 0
done = False
results = {"win": 0, "draw": 0, "loss": 0}

win_rates = []
draw_rates = []
loss_rates = []
tip = 0

# lets assign who is which player randomly each episode, then 
botorder = 0
match = Match("Player 1", "Player 2")
#agent = Agent(alpha=0.1, gamma=0.9, epsilon=0.1)
players = ["Player 2", "Player 1"]
agent_player = "Player 1" # need to initialize this so my loop works as intended
agent = DeepAgentConvolved(epsilon_decay = 0.9995)
dummy = dummy_agent()
steps_done = 0
# training loop vs random oponent
while not done:
    # the match over loop should be the same as in the qlearning version
    #print("Doing a move")
    if match.IsOver:
        agent.games_played += 1
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
        # moved episode counter here so that we play full games n times
        episodes += 1 
        # start new match
        match = Match("Player 1", "Player 2")
        
        # Decay epsilon
        # we start it high ( to encourgae exploration, even as high as 1 to encourage total randomness)
        # then we decay it to the point the model only uses what it has learned
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)
        
        if agent.epsilon == agent.epsilon_min and tip == 0:
            print(f"Minimal epsilon reached at episode {episodes+1}")
            tip += episodes + 1
    if match.turn == 1: # if its the first turn we have to dedide is the random bot player 1 or player 2, and then we will have to make the agent play the other player
        if random.random() < 0.5:
            botorder = 0
            agent_player = "Player 1"
            player_token = 1
            bot_player = "Player 2"
        else:
            botorder = 1
            player_token = 2
            agent_player = "Player 2"
            bot_player = "Player 1"
    # currently possible moves 0-6
    moves = match.get_available_columns()


    if match.turn % 2 == botorder:   # random bot makes its move
        move = dummy.choose_action(moves)
        match.play_bot(move)
    else:

        state = match.grid.get_grid() 
        
        move = agent.choose_action(state,moves,player_token)
        # play the moves that the agent decided on 
        match.play_bot(move)

        # update agent memory
        #next_state = torch.FloatTensor(match.grid.get_grid()).flatten()
        #last_state = torch.FloatTensor(state).flatten()
        next_state = match.grid.get_grid()
        last_state = state
        if match.winner == agent_player:
            reward = 1
        elif match.winner == bot_player:
            reward = -1
        else:
            reward = 0

        agent.update_memory(last_state,next_state,move,done,reward,player_token)

        # Optimize model
        # model is optimized after batch_size batches. but epsilon is updated only episodically
        agent.optimize()
        # Update target network periodically, lets say every 1000 moves played
        # this has a consequence of tying update freq to speed of winning/losing, perhaps unintended but ok
        if steps_done % agent.target_update_freq == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print("target net updated")
        steps_done += 1

    
    if episodes == 10000:
        break

agent.winrate

with open("agent_v4.2.pkl", "wb") as file:
    pickle.dump(agent, file)

################################################################################################################
################################################################################################################

# lets try to pit two agents agents each other
episodes = 0
done = False
# lets assign who is which player randomly each episode, then 
botorder = 0
match = Match("Player 1", "Player 2")
#agent = Agent(alpha=0.1, gamma=0.9, epsilon=0.1)
players = ["Player 2", "Player 1"]
agent1_player = "Player 1" # need to initialize this so my loop works as intended
agent2_player = "Player 2" # need to initialize this so my loop works as intended

agent1 = DeepAgent(epsilon_decay = 0.995)
agent2 = DeepAgent(epsilon_decay = 0.999)
steps_done_agent1 = 0
steps_done_agent2 = 0
c1 = 0
c2 = 0
# training loop vs random oponent
while not done:
    # the match over loop should be the same as in the qlearning version
    #print("Doing a move")
    if match.IsOver:
        agent1.games_played += 1
        agent2.games_played += 1

        # collect some stats

        if match.winner == agent1_player:
            agent1.games_won += 1
            if botorder == 0:         
                agent1.games_won_first += 1
                agent1.winrate_first = agent1.games_won_first / agent1.games_went_first
                
            else:
                agent1.games_won_second += 1     
                agent1.winrate_second = agent1.games_won_second / agent1.games_went_second 
        elif match.winner == agent2_player:
            agent2.games_won += 1
            if botorder == 0:         
                agent2.games_won_first += 1
                agent2.winrate_first = agent2.games_won_first / agent2.games_went_first
            else:
                agent2.games_won_second += 1       
                agent2.winrate_second = agent2.games_won_second / agent2.games_went_second 
              
        

        agent1.winrate = agent1.games_won / agent1.games_played
        agent2.winrate = agent2.games_won / agent2.games_played
        if (episodes + 1) % 100 == 0:
            print(f"Episode {episodes+1} Agent 1 winrate {agent1.winrate}, learning phase {c2}")
            print(f"Episode {episodes+1} Agent 2 winrate {agent2.winrate}, learning phase {c1}")
        # moved episode counter here so that we play full games n times
        episodes += 1 
        # start new match
        moveorder = []
        match = Match("Player 1", "Player 2")
        
        # Decay epsilon
        # we start it high ( to encourgae exploration, even as high as 1 to encourage total randomness)
        # then we decay it to the point the model only uses what it has learned
        agent1.epsilon = max(agent1.epsilon_min, agent1.epsilon_decay * agent1.epsilon)
        agent2.epsilon = max(agent2.epsilon_min, agent2.epsilon_decay * agent2.epsilon)
        if agent2.epsilon == agent2.epsilon_min and c1 == 0:
            print("Minimal epsilon reached for agent 2")
            c1 +=1
        if agent1.epsilon == agent1.epsilon_min and c2 == 0:
            c2 +=1
            print("Minimal epsilon reached for agent 1")
    if match.turn == 1: # if its the first turn we have to dedide is the random bot player 1 or player 2, and then we will have to make the agent play the other player
        if random.random() < 0.5:
            botorder = 0
            agent1_player = "Player 1"
            agent2_player = "Player 2"
            agent1.games_went_first += 1
            agent2.games_went_second += 1
        else:
            botorder = 1
            agent1_player = "Player 2"
            agent2_player = "Player 1"
            agent2.games_went_first += 1
            agent1.games_went_second += 1
    # currently possible moves 0-6
    moves = match.get_available_columns()


    if match.turn % 2 == botorder:   # agent 2 takes its move
        state = match.grid.get_grid() 
        move = agent2.choose_action(state,moves)
        # play the moves that the agent decided on 
        match.play_bot(move)
        # update agent memory
        next_state = torch.FloatTensor(match.grid.get_grid()).flatten()
        last_state = torch.FloatTensor(state).flatten()
        if match.winner == agent2_player:
            reward = 1
        elif match.winner == agent1_player:
            reward = -1
        else:
            reward = 0
        agent2.update_memory(last_state,next_state,move,done,reward)
        # Optimize model
        # model is optimized after batch_size batches. but epsilon is updated only episodically
        agent2.optimize()
        # Update target network periodically, lets say every 1000 moves played
        # this has a consequence of tying update freq to speed of winning/losing, perhaps unintended but ok
        if steps_done_agent2 % agent2.target_update_freq == 0:
            agent2.target_net.load_state_dict(agent2.policy_net.state_dict())
            print("target net updated agent2")
        steps_done_agent2 += 1
    else:
        state = match.grid.get_grid() 
        move = agent1.choose_action(state,moves)
        # play the moves that the agent decided on 
        match.play_bot(move)
        # update agent memory
        next_state = torch.FloatTensor(match.grid.get_grid()).flatten()
        last_state = torch.FloatTensor(state).flatten()
        if match.winner == agent1_player:
            reward = 1
        elif match.winner == agent2_player:
            reward = -1
        else:
            reward = 0
        agent1.update_memory(last_state,next_state,move,done,reward)
        # Optimize model
        # model is optimized after batch_size batches. but epsilon is updated only episodically
        agent1.optimize()
        # Update target network periodically, lets say every 1000 moves played
        # this has a consequence of tying update freq to speed of winning/losing, perhaps unintended but ok
        if steps_done_agent1 % agent1.target_update_freq == 0:
            agent1.target_net.load_state_dict(agent1.policy_net.state_dict())
            print("target net updated for agent 1")
        steps_done_agent1 += 1

    if episodes == 10000:
        break


agent1.games_went_first
agent1.games_went_second

agent2.games_went_first
agent2.games_went_second


agent2.winrate_first
agent2.winrate_second

# Save the instance to a file
with open("agent_v3.3.pkl", "wb") as file:
    pickle.dump(agent1, file)
with open("agent_v3.4.pkl", "wb") as file:
    pickle.dump(agent2, file)

# Load the instance back from the file
#with open("player.pkl", "rb") as file:
#    loaded_player = pickle.load(file)