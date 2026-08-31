
import random
from collections import defaultdict
# switching to deep q learning
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

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



# simple newrual network adapted
class DQN(nn.Module):
    def __init__(self,input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten: (batch, 42)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DeepAgent:
    def __init__(self, learning_rate = 0.001, gamma=0.99,
                 epsilon=1.0, epsilon_min = 0.01, epsilon_decay = 0.995,
                 batch_size = 64, target_update_freq = 1000, memory_size = 10000):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.memory_size = memory_size
        #episodes = 1000
        # initialize the target and the working network as well
        self.input_dim = 42
        self.output_dim = 7
        self.policy_net = DQN(self.input_dim, self.output_dim)
        self.target_net = DQN(self.input_dim, self.output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # training utilities
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=self.memory_size)
        self.winrate = 0
        self.games_played = 0 
        self.games_won = 0
        self.total_reward = 0
        self.games_went_first = 0
        self.games_went_second = 0
        self.games_won_first = 0
        self.games_won_second = 0
        self.winrate_first = 0
        self.winrate_second = 0
    # this just plays moves based on the net, good code i hope for future playing vs human use
    def choose_action(self, state, availible_actions, mode = "learning"): 
        if random.random() < self.epsilon and mode =="learning":
            return random.choice(availible_actions)
        else:  # state should be input as match.grid.get_grid()
            state = torch.FloatTensor(state).flatten()
            q_values = self.policy_net(state.unsqueeze(0)) # only add the dim here cuz you messed up the optimizer func
            # if the move is unavailible due to full columns it needs to be removed
            mask = torch.full_like(q_values, -float('inf'))  # Shape: [1, 4]
            mask[:, availible_actions] = 0.0  # Important: index the 2nd dimension
            masked_q = q_values + mask
             # select from the possible ones
            move = torch.argmax(masked_q).item() 
            # depracated below because torch.argmax returns only first max value
            #best_actions = [a for a, q in zip(actions, q_values) if q == max_q]  # random tiebreaker
            #return random.choice(move)
            return move
    # separated the memory update so its more readable in the training loop
    def update_memory(self,last_state,next_state,move,done,reward):
        self.memory.append((last_state.squeeze(0), move, reward, next_state.squeeze(0), done))
        self.total_reward += reward
    # updating the function based on reward
    # this places the reward breakpoint outside the agent code and into the training loop
    # also good practice i hope?
    def optimize(self):
        if len(self.memory) < self.batch_size:
       # print("memory smaller than batch size")
        # wait unitl replay memory fills to batch size, we only optimize model when it goes through a batch
            return
        # break correlation between sequential
        batch = random.sample(self.memory, self.batch_size)
        # unzip into 5 separate tuples
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.stack(state_batch).squeeze(1)
        # this is somehow numpys fault that this extra dim gets added
        action_batch = torch.LongTensor(action_batch).unsqueeze(1)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.stack(next_state_batch).squeeze(1)
        done_batch = torch.FloatTensor(done_batch)

        # Compute Q-values for current states
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Compute target Q-values using the target network
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        loss = nn.MSELoss()(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



# 2. Helper functions to convert network weights to/from a flat genome vector
def model_to_vector(model):
    """Flattens all model weights and biases into a single 1D numpy array."""
    return np.concatenate([p.data.cpu().numpy().flatten() for p in model.parameters()])

def vector_to_model(vector, model):
    """Restores a flat array of weights back into the PyTorch model structure."""
    pointer = 0
    for p in model.parameters():
        num_param = p.numel()
        p.data = torch.from_numpy(vector[pointer:pointer + num_param]).view(p.size()).float()
        pointer += num_param
    return model

# 3. Evolutionary Operators
def crossover(parent1, parent2):
    """Performs single-point crossover between two parent genomes."""
    point = np.random.randint(1, len(parent1) - 1)
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2

def mutate(genome, mutation_rate=0.1, scale=0.2):
    """Applies random Gaussian mutations to individual genes."""
    mask = np.random.rand(*genome.shape) < mutation_rate
    noise = np.random.normal(0, scale, size=genome.shape)
    genome[mask] += noise[mask]
    return genome

