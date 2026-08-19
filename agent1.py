
import random
from collections import defaultdict

class Agent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = defaultdict(float)
        self.alpha = alpha  # training speed
        self.gamma = gamma  # learning speed (future/current reward, 1== future raward bias)
        self.epsilon = epsilon  # exploration rate

    # get q value for state-action pair
    def get_q(self, state, action):
        return self.q_table[(state, action)]
    # print q table for debugging
    def print_q_table(self):
        for key, value in self.q_table.items():
            print(f"State: {key[0]}, Action: {key[1]}, Q-value: {value}")
    def choose_action(self, state, actions):  # default dict lets us update q values for the 7 column moves 
                                              # without initializing them first
                                              # actions are taken randoomly depending on epsilon, otherwise the best q value is chosen
                                              # tiebreakers currently random
                                              # state will be entire board in some fashion
        if random.random() < self.epsilon:
            return random.choice(actions)
        else:
            q_values = [self.get_q(state, a) for a in actions]
            max_q = max(q_values)
            best_actions = [a for a, q in zip(actions, q_values) if q == max_q]  # random tiebreaker
            return random.choice(best_actions)
    
    # updating the q value based on reward
    def update(self, state, action, reward, next_state, next_actions):
        max_q_next = max([self.get_q(next_state, a) for a in next_actions], default=0)
        old_value = self.q_table[(state, action)]
        new_value = old_value + self.alpha * (reward + self.gamma * max_q_next - old_value)
        self.q_table[(state, action)] = new_value

