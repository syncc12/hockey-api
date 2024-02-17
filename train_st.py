import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from sklearn.model_selection import train_test_split
from joblib import load
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import TORCH_VERSION, TORCH_FILE_VERSION, PROJECTED_LINEUP_TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, calculate_accuracy, binary_accuracy, errorAnalysis
import time


NUM_EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(X_INPUTS)
OUTPUT_DIM = 1
HIDDEN_DIM = 204
KERNEL_SIZE = 15
IN_CHANNELS = 71
OUT_CHANNELS = 128

OUTPUT = 'winnerB'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SetTransformer(nn.Module):
  def __init__(self, in_features, out_features, num_heads=4, dim_feedforward=128):
    super(SetTransformer, self).__init__()
    self.encoder = nn.TransformerEncoderLayer(
      d_model=in_features,
      nhead=num_heads,
      dim_feedforward=dim_feedforward,
      batch_first=True
    )
    self.pooling = nn.AdaptiveMaxPool1d(1) # Simple pooling for this example
    self.fc = nn.Linear(in_features, out_features)

  def forward(self, x):
    # x is of shape [batch_size, set_size, in_features]
    x = self.encoder(x)
    # Pooling over the set dimension (assuming the second dimension is the set dimension)
    x = x.transpose(1, 2) # [batch_size, in_features, set_size]
    x = self.pooling(x).squeeze(-1) # [batch_size, in_features]
    x = self.fc(x)
    return x

# Example usage
model = SetTransformer(in_features=10, out_features=2)
input_set = torch.rand(5, 3, 10) # 5 sets, each with 3 elements of 10 features
output = model(input_set)
print(output.shape) # Should be [5, 2]
