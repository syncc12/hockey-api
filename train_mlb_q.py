import numpy as np


# Number of states and actions
n_states = 3
n_actions = 2

# Initialize the Q-table to zeros
Q = np.zeros((n_states, n_actions))

# Define the rewards for each state and action pair
# Here, the rows represent states, and the columns represent actions
rewards = np.array([
  [5, -10],  # Rewards when in state 0
  [-1, 2],   # Rewards when in state 1
  [0, 3]     # Rewards when in state 2
])

# Define a simple policy to transition between states
def policy(state, epsilon=0.1):
  """Epsilon-greedy policy for action selection."""
  if np.random.rand() < epsilon:
    return np.random.randint(n_actions)  # Explore
  else:
    return np.argmax(Q[state])  # Exploit best known action


# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.6  # Discount factor
epsilon = 0.1  # Exploration rate
episodes = 10000  # Number of episodes to train

for episode in range(episodes):
  state = np.random.randint(n_states)  # Starting state
  
  action = policy(state, epsilon)
  next_state = np.random.randint(n_states)  # Simulate environment transition
  
  # Q-learning update rule
  reward = rewards[state, action]
  next_max = np.max(Q[next_state])  # Max value for the next state
  Q[state, action] = Q[state, action] + alpha * (reward + gamma * next_max - Q[state, action])

# Print the learned Q-table
print("Learned Q-table:")
print(Q)

# Using the learned Q-table to make decisions
test_state = 1  # Example state
best_action = np.argmax(Q[test_state])
print(f"Best action for state {test_state}: {best_action}")

