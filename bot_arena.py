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




# here the bots will fight for their lives


class BotArena:

    def __init__(self, num_matches=100, num_eras = 10):
        self.era = 1
        self.num_matches = num_matches
        
        self.num_matches = num_matches
        self.nume_eras = num_eras
    def populate_arena(self):
        # populate initial arena with some bots and agents
        if self.era == 1:
            instances =  [DeepAgentConvolved() for _ in range(8)]
            instances.append(dummy_agent())
            instances.append(DeepAgent())





# form a helper class that will run a duel between two agents and return the winner, loser, or draw
# lets make each duel a best of 3?
class Duel:
    
    def __init__(self, agent1, agent2):
        self.agent1 = agent1
        self.agent2 = agent2
        self.match = Match(agent1, agent2)
        self.winner = None
        self.loser = None
        self.draw = False

    def run_duel(self):
        result = self.match.play()  # not how that works yet but ok
        if result == 1:
            self.winner = self.agent1
            self.loser = self.agent2
        elif result == 2:
            self.winner = self.agent2
            self.loser = self.agent1
        else:
            self.draw = True