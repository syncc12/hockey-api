import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class ActorCritic(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size):
    super(ActorCritic, self).__init__()
    self.critic = nn.Sequential(
      nn.Linear(num_inputs, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1)
    )
    
    self.actor = nn.Sequential(
      nn.Linear(num_inputs, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, num_actions),
      nn.Softmax(dim=-1)
    )

  def forward(self, x):
    value = self.critic(x)
    probs = self.actor(x)
    dist = Categorical(probs)
    return dist, value

def ppo_update(policy_net, optimizer, states, actions, log_probs_old, returns, advantages, clip_param=0.2):
  dist, value = policy_net(states)
  entropy = dist.entropy().mean()
  log_probs = dist.log_prob(actions)
  
  # Ratio for clipping
  ratios = torch.exp(log_probs - log_probs_old)
  
  # Objective function
  surr1 = ratios * advantages
  surr2 = torch.clamp(ratios, 1.0 - clip_param, 1.0 + clip_param) * advantages
  
  # Policy loss
  policy_loss = -torch.min(surr1, surr2).mean()
  
  # Value loss
  value_loss = (returns - value).pow(2).mean()

  # Total loss
  loss = 0.5 * value_loss + policy_loss - 0.01 * entropy
  
  # Perform backpropagation
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
