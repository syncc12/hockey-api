import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomActivation(nn.Module):
  def __init__(self):
    super(CustomActivation, self).__init__()
  
  def forward(self, x):
    # Example: A simple custom activation that squares the inputs
    return torch.pow(x, 2)

class MemoryModule(nn.Module):
  def __init__(self, input_features, memory_size):
    super(MemoryModule, self).__init__()
    self.memory_size = memory_size
    self.memory = nn.Parameter(torch.randn(memory_size, input_features), requires_grad=True)

  def forward(self, x):
    # Compute similarity between input and memory
    similarity = F.cosine_similarity(x.unsqueeze(1), self.memory.unsqueeze(0), dim=2)
    # Retrieve the most similar memory content
    _, indices = similarity.max(dim=1)
    selected_memory = self.memory[indices]
    # Combine input and retrieved memory
    combined = x + selected_memory
    return combined

class NoiseInjection(nn.Module):
  def __init__(self, noise_level=0.1):
    super(NoiseInjection, self).__init__()
    self.noise_level = noise_level

  def forward(self, x):
    if self.training:
      return x + torch.randn_like(x) * self.noise_level
    return x