import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSet(nn.Module):
  def __init__(self, in_features, out_features):
    super(DeepSet, self).__init__()
    # Element-wise processing network
    self.phi = nn.Sequential(
      nn.Linear(in_features, 128),
      nn.ReLU(),
      nn.Linear(128, 128)
    )
    # Aggregation and further processing network
    self.rho = nn.Sequential(
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, out_features)
    )

  def forward(self, x):
    # x is of shape [batch_size, set_size, in_features]
    x = self.phi(x) # Apply phi to each element
    x = x.sum(dim=1) # Aggregate (sum) over the set dimension
    x = self.rho(x) # Further processing
    return x

# Example usage
model = DeepSet(in_features=10, out_features=2)
# Assuming we have a batch of 5 sets, each containing 3 elements with 10 features
input_set = torch.rand(5, 3, 10)
output = model(input_set)
print(output.shape) # Should be [5, 2] indicating the output features for each set
