import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from joblib import load
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import TORCH_VERSION, TORCH_FILE_VERSION, TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy, errorAnalysis
import time
from training_input import training_input, test_input


NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(X_INPUTS)
# HIDDEN_DIM = 256
HIDDEN_DIM = BATCH_SIZE
OUTPUT_DIM = 1

USE_PARTIAL_SEASONS = False

OUTPUT = 'winnerB'

class CustomDataset(Dataset):
  def __init__(self, x_train, y_train):
    self.x = torch.from_numpy(x_train.astype(np.float32))
    self.y = torch.from_numpy(y_train.astype(np.float32))
    self.len = self.x.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return self.x[index], self.y[index]


class BinaryClassificationRNN(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim):
    super(BinaryClassificationRNN, self).__init__()
    
    
    # LSTM layer
    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
    
    # Linear layer
    self.fc = nn.Linear(hidden_dim, output_dim)
    
    # Sigmoid activation
    self.sigmoid = nn.Sigmoid()
      
  def forward(self, x):
    # text = text.long()
    lstm_out, (hidden, cell) = self.lstm(x)
    
    # We only want the last output from the sequence for binary classification
    # print(hidden)
    # print(hidden.shape)
    # hidden = hidden[-1]
    
    dense_outputs = self.fc(hidden)
    outputs = self.sigmoid(dense_outputs)
    # print(outputs)
    # outputs = outputs.view(-1, 1)
    print(outputs.shape)
    return outputs



def train():
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]

  SEASONS = [
    # 20052006,
    # 20062007,
    # 20072008,
    20082009,
    20092010,
    20102011,
    20112012,
    20122013,
    20132014,
    20142015,
    20152016,
    20162017,
    20172018,
    20182019,
    20192020,
    20202021,
    20212022,
    # 20222023,
  ]
  training_data = training_input(SEASONS)

  data = pd.DataFrame(training_data)
  x_train = data [X_INPUTS].to_numpy()
  y_train = data [[OUTPUT]].to_numpy()

  x_test, y_test = test_input(X_INPUTS,[OUTPUT], season=20222023)

  x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

  train_dataset = CustomDataset(x_train, y_train)
  validation_dataset = CustomDataset(x_validation, y_validation)
  test_dataset = CustomDataset(x_test, y_test)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = BinaryClassificationRNN(input_dim=len(X_INPUTS),hidden_dim=HIDDEN_DIM,output_dim=OUTPUT_DIM).to(device)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # Loss function and optimizer
  # criterion = nn.BCEWithLogitsLoss ()
  criterion = nn.BCELoss ()
  # criterion = nn.MSELoss ()
  # criterion = HingeLoss ()
  # criterion = FocalLoss ()
  optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=L2)
  # optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=L2)

  # Training loop
  running_total_accuracy = []
  for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_accuracy = 0.0
    running_batches = 0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      # set optimizer to zero grad to remove previous epoch gradients
      optimizer.zero_grad()
      # forward propagation
      outputs = model(inputs)
      # print(f"Output shape: {outputs.shape}, Target shape: {labels.shape}")
      loss = criterion(outputs, labels)
      # backward propagation
      loss.backward()
      # optimize
      optimizer.step()
      running_loss += loss.item()
      running_accuracy += binary_accuracy(outputs, labels)
      running_batches += 1
    running_total_accuracy.append(running_accuracy / running_batches)
    running_average_accuracy = sum(running_total_accuracy) / len(running_total_accuracy) if len(running_total_accuracy) > 0 else 0.0
    # display statistics
    print(f'[{epoch + 1}/{NUM_EPOCHS}] accuracy: {running_accuracy / running_batches:.2f}% ({running_average_accuracy:.2f}%), loss: {running_loss / running_batches:.4f}')

  # Test Loop
  # model.eval()  # Set the model to evaluation mode
  correct, total = 0, 0
  total_accuracy = 0
  test_batches = 0
  with torch.no_grad():
    for data in test_loader:
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      # calculate output by running through the network
      outputs = model(inputs)
      # get the predictions
      __, predicted = torch.max(outputs.data, 1)
      # update results
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      current_accuracy = binary_accuracy(outputs, labels)
      total_accuracy += current_accuracy
      test_batches += 1

  average_accuracy = total_accuracy / test_batches
  print(f'Average Accuracy: {average_accuracy}')


  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'version': TORCH_VERSION,
    'inputs': X_INPUTS,
    'outputs': OUTPUT,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'seasons': SEASONS,
    'projectedLineup': False,
    'model': 'PyTorch Binary Recurrent Neural Network Classifier',
    'hyperparameters': {
      'num_epochs': NUM_EPOCHS,
      'hidden_dim': HIDDEN_DIM,
      'output_dim': OUTPUT_DIM,
      'batch_size': BATCH_SIZE,
      'num_workers': NUM_WORKERS,
      'lr': LR,
      'l2': L2,
      'input_dim': INPUT_DIM,
      'output_dim': OUTPUT_DIM,
    },
    'accuracies': {
      OUTPUT: average_accuracy,
    },
  })

  # Save/Load the model
  torch.save(model.state_dict(), f'./models/nhl_ai_v{TORCH_FILE_VERSION}_rnn_{OUTPUT}.pt')



if __name__ == '__main__':
  train()