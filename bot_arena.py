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
            print(f"Bot {i+1}: {instance.randID}, Winrate: {instance.winrate:.2f}, Games Played: {instance.games_played}, Games Won: {instance.games_won} Games playerd total: {instance.games_played_total} Epsilon: {instance.epsilon} ")

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
        self.agent1_player = "Player 1"
        self.agent2_player = "Player 2"
        self.steps_done_agent1 = 0
        self.steps_done_agent2 = 0
        self.episodes = 0

        # ---- delayed transitions: (state, action, token) ----
        self.pending_agent1 = None
        self.pending_agent2 = None

        # ---- per-agent token (1 if that agent is "Player 1", else 2) ----
        self.agent1_token = 1
        self.agent2_token = 2

        # which physical player goes 2nd (0 or 1), set on turn 1
        self.botorder = 0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _terminal_reward(self, agent_player):
        """Reward from the perspective of agent_player at a terminal state."""
        if self.match.winner == agent_player:
            return 1
        elif self.match.winner in ("Player 1", "Player 2"):
            return -1
        else:
            return 0

    def _resolve_pending(self, which, agent, next_state, done):
        """Finalize agent #which's pending transition using next_state as s'."""
        if agent.type == "dummy":
            return
        attr = "pending_agent1" if which == 1 else "pending_agent2"
        pending = getattr(self, attr)
        if pending is None:
            return
        state, action, token = pending
        if done:
            agent_player = self.agent1_player if which == 1 else self.agent2_player
            reward = self._terminal_reward(agent_player)
        else:
            reward = 0
        agent.update_memory(state, next_state, action, done, reward, token)
        agent.optimize()
        setattr(self, attr, None)

    def _resolve_all_pending_terminal(self):
        """Game just ended - finalize any outstanding pending transitions."""
        grid = self.match.grid.get_grid()
        self._resolve_pending(1, self.agent1, grid, True)
        self._resolve_pending(2, self.agent2, grid, True)

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run_duel(self):
        while not self.done:

            # ---------- game end: finalize pending + stats ----------
            if self.match.IsOver:
                self._resolve_all_pending_terminal()

                self.agent1.games_played += 1
                self.agent2.games_played += 1
                self.agent1.games_played_total += 1
                self.agent2.games_played_total += 1

                if self.match.winner == self.agent1_player:
                    self.agent1.games_won += 1
                    self.agent1.winrate = self.agent1.games_won / self.agent1.games_played
                    if self.botorder == 0:
                        self.agent1.games_won_first += 1
                        self.agent1.winrate_first = (self.agent1.games_won_first /
                                                     self.agent1.games_went_first)
                    else:
                        self.agent1.games_won_second += 1
                        self.agent1.winrate_second = (self.agent1.games_won_second /
                                                      self.agent1.games_went_second)
                elif self.match.winner == self.agent2_player:
                    self.agent2.games_won += 1
                    self.agent2.winrate = self.agent2.games_won / self.agent2.games_played
                    if self.botorder == 1:
                        self.agent2.games_won_first += 1
                        self.agent2.winrate_first = (self.agent2.games_won_first /
                                                     self.agent2.games_went_first)
                    else:
                        self.agent2.games_won_second += 1
                        self.agent2.winrate_second = (self.agent2.games_won_second /
                                                      self.agent2.games_went_second)

                self.episodes += 1
                self.match = Match("Player 1", "Player 2")

                self.agent1.epsilon = max(self.agent1.epsilon_min,
                                          self.agent1.epsilon_decay * self.agent1.epsilon)
                self.agent2.epsilon = max(self.agent2.epsilon_min,
                                          self.agent2.epsilon_decay * self.agent2.epsilon)

            # ---------- coin flip on first turn ----------
            if self.match.turn == 1:
                if random.random() < 0.5:
                    self.botorder = 0
                    self.agent1_player = "Player 1"
                    self.agent2_player = "Player 2"
                    self.agent1.games_went_first += 1
                    self.agent2.games_went_second += 1
                    self.agent1_token = 1
                    self.agent2_token = 2
                else:
                    self.botorder = 1
                    self.agent1_player = "Player 2"
                    self.agent2_player = "Player 1"
                    self.agent2.games_went_first += 1
                    self.agent1.games_went_second += 1
                    self.agent1_token = 2
                    self.agent2_token = 1

            self.moves = self.match.get_available_columns()

            # ==========================================================
            # agent2's turn
            # ==========================================================
            if self.match.turn % 2 == self.botorder:

                # Agent2 is about to play, so the opponent (agent1) has
                # already replied -> the current grid IS s' for agent2's
                # previous pending transition.
                self._resolve_pending(2, self.agent2,
                                      self.match.grid.get_grid(), False)

                if self.agent2.type != "dummy":
                    state = self.match.grid.get_grid()
                    move = self.agent2.choose_action(state, self.moves, self.agent2_token)
                    self.match.play_bot(move)

                    if self.match.IsOver:
                        # agent2's own move ended the game -> finalize now
                        grid = self.match.grid.get_grid()
                        reward = self._terminal_reward(self.agent2_player)
                        self.agent2.update_memory(state, grid, move, True,
                                                  reward, self.agent2_token)
                        self.agent2.optimize()
                        # agent1's pending is also terminal now
                        self._resolve_pending(1, self.agent1, grid, True)
                    else:
                        # stash it; we'll resolve it after agent1 replies
                        self.pending_agent2 = (state, move, self.agent2_token)

                    if self.steps_done_agent2 % self.agent2.target_update_freq == 0:
                        self.agent2.target_net.load_state_dict(
                            self.agent2.policy_net.state_dict())
                    self.steps_done_agent2 += 1
                else:
                    self.match.play_bot(self.agent2.choose_action(self.moves))
                    if self.match.IsOver:
                        self._resolve_pending(1, self.agent1,
                                              self.match.grid.get_grid(), True)

            # ==========================================================
            # agent1's turn
            # ==========================================================
            else:
                # Agent1 is about to play, so agent2 has already replied.
                self._resolve_pending(1, self.agent1,
                                      self.match.grid.get_grid(), False)

                if self.agent1.type != "dummy":
                    state = self.match.grid.get_grid()
                    move = self.agent1.choose_action(state, self.moves, self.agent1_token)
                    self.match.play_bot(move)

                    if self.match.IsOver:
                        grid = self.match.grid.get_grid()
                        reward = self._terminal_reward(self.agent1_player)
                        self.agent1.update_memory(state, grid, move, True,
                                                  reward, self.agent1_token)
                        self.agent1.optimize()
                        self._resolve_pending(2, self.agent2, grid, True)
                    else:
                        self.pending_agent1 = (state, move, self.agent1_token)

                    if self.steps_done_agent1 % self.agent1.target_update_freq == 0:
                        self.agent1.target_net.load_state_dict(
                            self.agent1.policy_net.state_dict())
                    self.steps_done_agent1 += 1
                else:
                    self.match.play_bot(self.agent1.choose_action(self.moves))
                    if self.match.IsOver:
                        self._resolve_pending(2, self.agent2,
                                              self.match.grid.get_grid(), True)

            if self.episodes == self.match_limit:
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