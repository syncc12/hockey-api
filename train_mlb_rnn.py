import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from joblib import dump, load
import pandas as pd
import numpy as np
from pages.mlb.inputs import X_INPUTS_MLB_T, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_TEAM_VERSION, MLB_TEAM_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
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
  2015,
  2014,
  2013,
  2012,
  2011,
  2010,
  2009,
  2008,
  2007,
  2006,
  2005,
  2004,
  2003,
  2002,
  2001,
  2000,
]

OUTPUT = 'winner'

USE_TEST_SPLIT = True

TEAM = 143

BATCH_SIZE = 19

class GameDataset(Dataset):
  def __init__(self, sequences, labels):
    # Check if sequences is a DataFrame or Series and convert to numpy array
    if isinstance(sequences, (pd.DataFrame, pd.Series)):
      sequences = sequences.to_numpy(dtype=float)

    # Check if labels is a DataFrame or Series and convert to numpy array
    if isinstance(labels, (pd.DataFrame, pd.Series)):
      labels = labels.to_numpy(dtype=float)

    self.sequences = torch.tensor(sequences, dtype=torch.float)
    self.labels = torch.tensor(labels, dtype=torch.float)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return self.sequences[idx], self.labels[idx]

class BaseballRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(BaseballRNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    print(type(x))
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
    x = x.unsqueeze(1)
    # print(f"x shape: {x.shape}, h0 shape: {h0.shape}")
    out, _ = self.rnn(x, h0)
    out = out[:, -1, :] # Take the last time step
    out = self.fc(out)
    return torch.sigmoid(out)



INPUT_SIZE = len(X_INPUTS_MLB_T)
HIDDEN_SIZE = 1000
NUM_LAYERS = 500
LR = 0.001
EPOCHS = 50

def train(dtrain,dtest):
  train_dataset = GameDataset(dtrain['x_train'], dtrain['y_train'])
  # print(f"Train Dataset: {len(train_dataset)}")
  train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
  test_dataset = GameDataset(dtest['x_test'], dtest['y_test'])
  test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
  model = BaseballRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS)
  model = model.to(device)
  criterion = torch.nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=LR)

  for epoch in range(EPOCHS):
    correct = 0
    total = 0
    for sequences, labels in train_loader:
      sequences, labels = sequences.to(device), labels.to(device)
      optimizer.zero_grad() # Clear gradients
      outputs = model(sequences) # Forward pass: Compute predicted outputs by passing inputs to the model
      outputs = outputs.squeeze()
      loss = criterion(outputs, labels) # Calculate Loss
      loss.backward() # Backward pass: compute gradient of the loss with respect to model parameters
      optimizer.step() # Perform a single optimization step (parameter update)
        
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
  teamLookup = team_lookup(mlb_db,only_active_mlb=True)
  TRAINING_DATA = mlb_training_input(SEASONS)
  data1 = pd.DataFrame(TRAINING_DATA)
  data2 = pd.DataFrame(TRAINING_DATA)

  data1.rename(columns=home_rename, inplace=True)
  data2.rename(columns=away_rename, inplace=True)
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  for column in ENCODE_COLUMNS:
    data = data[data[column] != -1]
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data[column] = encoder.transform(data[column])
  keep_teams = list(teamLookup.keys())
  data = data[data['team'].isin(keep_teams)]
  data = data[data['opponent'].isin(keep_teams)]
  data.reset_index(drop=True, inplace=True)
  all_data = data
  teams = data.groupby('team')

  # if TEAM:
  #   teams = [(TEAM, teams.get_group(TEAM))]

  dtrains = {}
  dtests = {}
  for team, team_data in teams:
    x_train = team_data [X_INPUTS_MLB_T]
    y_train = team_data [[OUTPUT]].values.ravel()
    if USE_TEST_SPLIT:
      x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
      dtests[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    dtrains[team] = {'x_train':x_train,'y_train':y_train,'len':len(x_train)}
  
  if not USE_TEST_SPLIT:
    TEST_DATA = mlb_test_input(TEST_SEASONS)
    test_data1 = pd.DataFrame(TEST_DATA)
    test_data2 = pd.DataFrame(TEST_DATA)
    test_data1.rename(columns=home_rename, inplace=True)
    test_data2.rename(columns=away_rename, inplace=True)
    test_data = pd.concat([test_data1, test_data2], axis=0)
    test_data.reset_index(drop=True, inplace=True)
    for column in ENCODE_COLUMNS:
      test_data = test_data[test_data[column] != -1]
      encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
      test_data[column] = encoder.transform(test_data[column])
    test_data = test_data[test_data['team'].isin(keep_teams)]
    test_data = test_data[test_data['opponent'].isin(keep_teams)]
    test_data.reset_index(drop=True, inplace=True)
    all_test_data = test_data
    test_teams = test_data.groupby('team')

    # if TEAM:
    #   test_teams = [(TEAM, test_teams.get_group(TEAM))]

    for team, team_data in test_teams:
      x_test = team_data [X_INPUTS_MLB_T]
      y_test = team_data [[OUTPUT]].values.ravel()
      dtests[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}

  train(dtrains[TEAM],dtests[TEAM])