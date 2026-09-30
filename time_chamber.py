import pickle
from agent1 import dummy_agent, Agent, DQN, DeepAgent, DeepAgentConvolved
from bot_arena import BotArena,Duel
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from match import Match
import copy
from itertools import combinations

if __name__ == "__main__":
    with open("era2_bots/bot_rDbR6j7R_era6_era7_era8_era11_gp_801000.pkl", "rb") as file:
            loaded_agent = pickle.load(file)


    loaded_agent.epsilon = 0.5  # starting pretty strong
    loaded_agent.epsilon_decay = 0.99999  # decay epsilon over time, this will take aroun 160k rounds to reach 0.1, 

    # run this agianst dummy agent for a while


    arena = BotArena(num_rounds=10, num_eras=200, match_limit=1000)
    dummy_agent = dummy_agent()
    bots = [loaded_agent, dummy_agent]


    for era in range(arena.num_eras):
        print(f"Starting era decade {era+1}")
        arena.round_robin(bots)
        arena.uptick_era()
        if era % 10 == 0:
            arena.print_stats(bots)

        if era % 50 == 0:
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


