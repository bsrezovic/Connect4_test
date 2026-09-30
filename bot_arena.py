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
import copy
from itertools import combinations  # for bracket  creating purposes


# here the bots will fight for their lives


class BotArena:

    def __init__(self, num_rounds=100, num_eras = 10, match_limit=101):
        self.era = 1
        self.num_rounds = num_rounds
        self.num_eras = num_eras
        self.match_limit = match_limit
    def populate_arena(self):
        # populate initial arena with some bots and agents
        
        instances =  [DeepAgentConvolved() for _ in range(9)]
        instances.append(dummy_agent())
        return instances

    def round_robin(self, instances):
        
        for agent1, agent2 in combinations(instances, 2):
            duel = Duel(agent1, agent2, self.match_limit)
            duel.run_duel()
    def uptick_era(self):
        self.era += 1
    def print_stats(self, instances):
        print(f"Winrates after era {self.era}:")
        for i, instance in enumerate(instances):
            print(f"Bot {i+1}: {instance.randID}, Winrate: {instance.winrate:.2f}, Games Played: {instance.games_played}, Games Won: {instance.games_won} Games playerd total: {instance.games_played_total}")

    def update_bracket(self, instances):
        # sort the instances by winrate
        instances.sort(key=lambda x: x.winrate, reverse=True)
        # print the winrates of the instances
        print(f"Winrates after era {self.era}:")
        for i, instance in enumerate(instances):
            print(f"Bot {i+1}: {instance.randID}, Winrate: {instance.winrate:.2f}, Games Played: {instance.games_played}, Games Won: {instance.games_won} Games playerd total: {instance.games_played_total}")

        # keep the top 50% of the instances(

        top_4 = instances[:4]
        print(f"Top 4 bots after era {self.era}:")
        for i, instance in enumerate(top_4):
            
            print(f"Bot {i+1}: {instance.randID}, Winrate: {instance.winrate:.2f}, Games Played: {instance.games_played}, Games Won: {instance.games_won} Games playerd total: {instance.games_played_total}")
        
        
        
        new_instances = top_4.copy() 
        # for now make 4 copies and then 2 fresh ones 
        for i in range(4):

            source_agent = top_4[i % 4]
            if isinstance(source_agent, dummy_agent):
                # if the dummy is winning dont copy it, save the 5th place instead
                source_agent = instances[5]
                source_id = source_agent.randID
                new_id = f"{source_id}_era{self.era}" # keeps a string based rep of the agent's lineage
                                                    # will be useful for future "crossbreeding"                      
                print(f"Creating new agent from {source_id} with new ID {new_id}")
                new_agent = copy.deepcopy(source_agent)
                new_agent.randID = new_id               
                new_instances.append(new_agent)
            else:
                source_id = source_agent.randID
                new_id = f"{source_id}_era{self.era}" # keeps a string based rep of the agent's lineage
                                                    # will be useful for future "crossbreeding"                      
                print(f"Creating new agent from {source_id} with new ID {new_id}")
                new_agent = copy.deepcopy(source_agent)
                new_agent.randID = new_id
                
                new_instances.append(new_agent)
        new_instances.append(DeepAgentConvolved())
        # check if we have the required dummy instance
        if any(isinstance(obj, dummy_agent) for obj in new_instances):
            new_instances.append(DeepAgentConvolved())
        else:
            new_instances.append(dummy_agent())
        self.era += 1
        return new_instances



# form a helper class that will run a duel between two agents and return the winner, loser, or draw
# lets make each duel a best of 101?
class Duel:
    
    def __init__(self, agent1, agent2, match_limit=101):
        self.agent1 = agent1
        self.agent2 = agent2
        self.match = Match("Player 1", "Player 2")
        self.winner = None
        self.loser = None
        self.match_limit = match_limit
        self.done = False
        self.agent1_player = "Player 1" # need to initialize this so my loop works as intended
        self.agent2_player = "Player 2" # need to initialize this so my loop works as intended
        self.steps_done_agent1 = 0
        self.steps_done_agent2 = 0
        self.episodes = 0
    def run_duel(self):
        
        while not self.done:
            
            # the match over loop should be the same as in the qlearning version
            #print("Doing a move")
            if self.match.IsOver:
                
                self.agent1.games_played += 1
                self.agent2.games_played += 1
                self.agent1.games_played_total += 1
                self.agent2.games_played_total += 1
                # collect some stats

                if self.match.winner == self.agent1_player:
                    self.agent1.games_won += 1
                    self.agent1.winrate = self.agent1.games_won / self.agent1.games_played
                    if self.botorder == 0:         
                        self.agent1.games_won_first += 1
                        self.agent1.winrate_first = self.agent1.games_won_first / self.agent1.games_went_first
                        
                    else:
                        self.agent1.games_won_second += 1     
                        self.agent1.winrate_second = self.agent1.games_won_second / self.agent1.games_went_second 
                elif self.match.winner == self.agent2_player:
                    self.agent2.games_won += 1
                    self.agent2.winrate = self.agent2.games_won / self.agent2.games_played
                    if self.botorder == 1:         
                        self.agent2.games_won_first += 1
                        self.agent2.winrate_first = self.agent2.games_won_first / self.agent2.games_went_first
                    else:
                        self.agent2.games_won_second += 1       
                        self.agent2.winrate_second = self.agent2.games_won_second / self.agent2.games_went_second 
                    

                # moved episode counter here so that we play full games n times
                self.episodes += 1 
                # start new match
                self.match = Match("Player 1", "Player 2")
                
                # Decay epsilon
                # we start it high ( to encourgae exploration, even as high as 1 to encourage total randomness)
                # then we decay it to the point the model only uses what it has learned
                self.agent1.epsilon = max(self.agent1.epsilon_min, self.agent1.epsilon_decay * self.agent1.epsilon)
                self.agent2.epsilon = max(self.agent2.epsilon_min, self.agent2.epsilon_decay * self.agent2.epsilon)

            if self.match.turn == 1: # if its the first turn we have to dedide is the random bot player 1 or player 2, and then we will have to make the self.agent play the other player
                if random.random() < 0.5:
                    self.botorder = 0
                    self.agent1_player = "Player 1"
                    self.agent2_player = "Player 2"
                    self.agent1.games_went_first += 1
                    self.agent2.games_went_second += 1
                    self.player_token = 1
                else:
                    self.botorder = 1
                    self.agent1_player = "Player 2"
                    self.agent2_player = "Player 1"
                    self.agent2.games_went_first += 1
                    self.agent1.games_went_second += 1
                    self.player_token = 2
            # currently possible moves 0-6
            self.moves = self.match.get_available_columns()
            

            if self.match.turn % 2 == self.botorder:   # self.agent 2 takes its move
                if self.agent2.type != "dummy":
                    self.state = self.match.grid.get_grid() 
            
                    self.move = self.agent2.choose_action(self.state,self.moves,self.player_token)
                    # play the moves that the self.agent decided on 
                    self.match.play_bot(self.move)
            
                    # update self.agent memory
                    self.next_state = self.match.grid.get_grid()
                    self.last_state = self.state
                    if self.match.winner == self.agent2_player:
                        self.reward = 1
                    elif self.match.winner == self.agent1_player:
                        self.reward = -3
                    else:
                        self.reward = 0
                    self.agent2.update_memory(self.last_state,self.next_state,self.move,self.done,self.reward,self.player_token)
                    # Optimize model
                    # model is optimized after batch_size batches. but epsilon is updated only episodically
                    self.agent2.optimize()
                    # Update target network periodically, lets say every 1000 moves played
                    # this has a consequence of tying update freq to speed of winning/losing, perhaps unintended but ok
                    if self.steps_done_agent2 % self.agent2.target_update_freq == 0:
                        self.agent2.target_net.load_state_dict(self.agent2.policy_net.state_dict())

                    self.steps_done_agent2 += 1
                else:
                    self.move = self.agent2.choose_action(self.moves)
                    # play the moves that the self.agent decided on 
                    self.match.play_bot(self.move)
            else:
                if self.agent1.type != "dummy":
                    self.state = self.match.grid.get_grid() 
    
                    self.move = self.agent1.choose_action(self.state,self.moves,self.player_token)
                    # play the moves that the self.agent decided on 
                    self.match.play_bot(self.move)

                    # update self.agent memory
                    self.next_state = self.match.grid.get_grid()
                    self.last_state = self.state
                    if self.match.winner == self.agent1_player:
                        self.reward = 1
                    elif self.match.winner == self.agent2_player:
                        self.reward = -3
                    else:
                        self.reward = 0
                    self.agent1.update_memory(self.last_state,self.next_state,self.move,self.done,self.reward,self.player_token)
                    # Optimize model
                    # model is optimized after batch_size batches. but epsilon is updated only episodically
                    self.agent1.optimize()
                    # Update target network periodically, lets say every 1000 moves played
                    # this has a consequence of tying update freq to speed of winning/losing, perhaps unintended but ok
                    if self.steps_done_agent1 % self.agent1.target_update_freq == 0:
                        self.agent1.target_net.load_state_dict(self.agent1.policy_net.state_dict())
                    self.steps_done_agent1 += 1
                else:
                    self.move = self.agent1.choose_action(self.moves)
                    # play the moves that the self.agent decided on 
                    self.match.play_bot(self.move)
            if self.episodes == self.match_limit:
                #print(f"After 101 episodes, {self.agent1_player} has won {self.agent1.games_won} games and {self.agent2_player} has won {self.agent2.games_won} games")
                self.done = True

#agent1 = DeepAgentConvolved()
#agent2 = DeepAgentConvolved()
#dvoboy = Duel(agent1, agent2)

#dvoboy.run_duel()

if __name__ == "__main__":
    arena = BotArena(num_rounds=10, num_eras=100, match_limit=1000)

    bots = arena.populate_arena()

    for era in range(arena.num_eras):
        print(f"Starting era decade {era+1}")
        arena.round_robin(bots)
        if era % arena.num_rounds == 0:
            # update the bots every 10 rounds
            bots = arena.update_bracket(bots)

        if era % 10 == 0:
            # the bots are sorted after the update_bracket function, so we can save them in order of performance
            for i, bot in enumerate(bots[:3]):
                if bot.type != "dummy":
                    with open(f'bot_{bot.randID}_saved_era{arena.era}_gp_{bot.games_played_total}.pkl', 'wb') as f:
                        pickle.dump(bot, f)
            
        for bot in bots:
            if era <= arena.num_eras * 0.8 and bot.epsilon > bot.epsilon_min: 

                
                bot.epsilon = 0.2
            bot.reset_stats()  # Reset stats for the next era


    #save the agents to disk
    for i, bot in enumerate(bots):
        with open(f'bot_{bot.randID}_era{arena.era}_gp_{bot.games_played_total}.pkl', 'wb') as f:
            pickle.dump(bot, f)


    for i in range(10):
        print(i)