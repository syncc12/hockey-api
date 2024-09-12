import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from joblib import dump, load
import pandas as pd
import numpy as np
from pages.mlb.inputs import X_INPUTS_MLB, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_TEAM_VERSION, MLB_TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
mlb_db = client["mlb"]
TEST_SEASONS = [
  2023,
  2022,
  2021,
]
SEASONS = [
  2023,
  2022,
  2021,
  2020,
  2019,
  2018,
  2017,
  2016,
  # 2015,
  # 2014,
  # 2013,
  # 2012,
  # 2011,
  # 2010,
  # 2009,
  # 2008,
  # 2007,
  # 2006,
  # 2005,
  # 2004,
  # 2003,
  # 2002,
  # 2001,
  # 2000,
]

OUTPUT = 'winner'

USE_TEST_SPLIT = True

TEAM = 143

BATCH_SIZE = len(X_INPUTS_MLB)

class GameDataset(Dataset):
  def __init__(self, sequences, labels):
    self.sequences = torch.from_numpy(sequences.astype(np.float32))
    self.labels = torch.from_numpy(labels.astype(np.float32))
    self.len = self.sequences.shape[0]

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.sequences[idx], self.labels[idx]

class BaseballCNN(nn.Module):
  def __init__(self, num_input_channels):
    super(BaseballCNN, self).__init__()
    self.conv1 = nn.Conv1d(in_channels=num_input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
    self.relu = nn.ReLU()
    self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
    self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

    self._to_linear = self._get_conv_output_size(num_input_channels)
    # print(f"Conv Output Size: {self._to_linear}")

    self.fc1 = nn.Linear(self._to_linear, 120)
    self.fc2 = nn.Linear(120, 2) # Using 2 output units with softmax

  def _get_conv_output_size(self, shape):
    # Pass a dummy input through the conv + pool layers to get the output size
    bs = 1  # Batch size of 1
    input = torch.rand(bs, shape, len(X_INPUTS_MLB))
    # print(f"Input Size: {input.size()}")
    output = self.pool(self.conv1(input))
    # print(f"Output Size: {output.size()}")
    output = self.pool(self.conv2(output))
    # print(f"Output Size: {output.size()}")
    flat_output = int(output.size(2))
    # print(f"Flat Output Size: {flat_output}")
    return flat_output

  def forward(self, x):
    x = self.pool(self.relu(self.conv1(x)))
    x = self.pool(self.relu(self.conv2(x)))
    x = torch.flatten(x, 1) # Flatten the tensor for the fully connected layer
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# model = BaseballCNN()
# criterion = nn.CrossEntropyLoss() # Suitable for binary classification with 2 output units
# optimizer = optim.Adam(model.parameters(), lr=0.001)

INPUT_SIZE = len(X_INPUTS_MLB)
HIDDEN_SIZE = 1000
NUM_LAYERS = 500
LR = 0.001
EPOCHS = 50

def train(x_train,y_train,x_test,y_test):
  train_dataset = GameDataset(x_train, y_train)
  # print(f"Train Dataset: {len(train_dataset)}")
  train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
  test_dataset = GameDataset(x_test, y_test)
  test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  # model = BaseballCNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
  model = BaseballCNN(num_input_channels=len(X_INPUTS_MLB))
  model = model.to(device)
  criterion = torch.nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=LR)

  for epoch in range(EPOCHS):
    correct = 0
    total = 0
    for inputs, labels in train_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      optimizer.zero_grad() # Zero the parameter gradients
      outputs = model(inputs) # Forward pass
      loss = criterion(outputs, labels) # Calculate loss
      loss.backward() # Backward pass
      optimizer.step() # Optimize

      # Convert outputs to predicted class (0 or 1)
      predicted = (outputs >= 0.5).float()
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    epoch_accuracy = correct / total

    print(f'Epoch {epoch+1}, Loss: {round(loss.item()*100,2)}%, Accuracy: {round(epoch_accuracy*100,2)}%')

  correct = 0
  total = 0
  model.eval()  # Set the model to evaluation mode
  with torch.no_grad():  # Instructs PyTorch to not compute gradients
    for sequences, labels in test_loader:
      sequences, labels = sequences.to(device), labels.to(device)
      outputs = model(sequences)
      predicted = (outputs.squeeze() >= 0.5).float()
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  final_accuracy = correct / total
  print(f'Final Accuracy: {round(final_accuracy*100,2)}%')

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  TRAINING_DATA = mlb_training_input(SEASONS)
  data = pd.DataFrame(TRAINING_DATA)
  for column in ENCODE_COLUMNS:
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data = data[data[column] != -1]
    data[column] = encoder.transform(data[column])

  x_train = data [X_INPUTS_MLB].to_numpy()
  y_train = data [[OUTPUT]].to_numpy()
  if USE_TEST_SPLIT:
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
  
  if not USE_TEST_SPLIT:
    TEST_DATA = mlb_test_input(TEST_SEASONS)
    test_data = pd.DataFrame(TEST_DATA)
    for column in ENCODE_COLUMNS:
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      test_data = test_data[test_data[column] != -1]
      test_data[column] = encoder.transform(test_data[column])

    x_test = test_data [X_INPUTS_MLB].to_numpy()
    y_test = test_data [[OUTPUT]].to_numpy()

  train(x_train,y_train,x_test,y_test)
