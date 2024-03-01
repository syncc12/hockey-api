import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
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
NUM_WORKERS = 1
INPUT_DIM = len(X_INPUTS)
LR=0.001
L2=1e-4
MAX_LENGTH = len(X_INPUTS)
NUM_LAYERS = 2
HEADS = 8
EMBED_SIZE = HEADS * 4
FORWARD_EXPANSION = 1
DROPOUT = 0.1
OUTPUT_DIM = 1

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

HIDDEN_LAYERS = (
  len(X_INPUTS)*2,
  len(X_INPUTS)*4,
  len(X_INPUTS)*6,
  len(X_INPUTS)*8,
  len(X_INPUTS)*10,
  len(X_INPUTS)*8,
  len(X_INPUTS)*6,
  len(X_INPUTS)*4,
)
  
class DBN(nn.Module):
  def __init__(self):
    super(DBN, self).__init__()
    # Define the architecture of the DBN
    self.layer1 = nn.Sequential(
      nn.Linear(INPUT_DIM, HIDDEN_LAYERS[0]),
      nn.ReLU(),
    )
    self.layer2 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
      nn.ReLU(),
    )
    self.layer3 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2]),
      nn.ReLU(),
    )
    self.layer4 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[2], HIDDEN_LAYERS[3]),
      nn.ReLU(),
    )
    self.layer5 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[3], HIDDEN_LAYERS[4]),
      nn.ReLU(),
    )
    self.layer6 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[4], HIDDEN_LAYERS[5]),
      nn.ReLU(),
    )
    self.layer7 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[5], HIDDEN_LAYERS[6]),
      nn.ReLU(),
    )
    self.layer8 = nn.Sequential(
      nn.Linear(HIDDEN_LAYERS[6], HIDDEN_LAYERS[7]),
      nn.ReLU(),
    )
        
    self.classifier = nn.Linear(HIDDEN_LAYERS[7], OUTPUT_DIM)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    x = self.classifier(x)
    x = self.sigmoid(x)
    return x


def train():
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]

  SEASONS = [
    # 20052006,
    # 20062007,
    # 20072008,
    # 20082009,
    # 20092010,
    # 20102011,
    # 20112012,
    # 20122013,
    # 20132014,
    # 20142015,
    # 20152016,
    # 20162017,
    20172018,
    20182019,
    20192020,
    20202021,
    20212022,
    20222023,
  ]
  training_data = training_input(SEASONS)

  data = pd.DataFrame(training_data)
  x_train = data [X_INPUTS].to_numpy()
  y_train = data [[OUTPUT]].to_numpy()

  # x_test, y_test = test_input(X_INPUTS,[OUTPUT], season=20222023)
  x_test, y_test = test_input(X_INPUTS,[OUTPUT])

  # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)

  train_dataset = CustomDataset(x_train, y_train)
  # validation_dataset = CustomDataset(x_validation, y_validation)
  test_dataset = CustomDataset(x_test, y_test)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  # Initialize the DBN
  model = DBN().to(device)

  # Loss and optimizer
  criterion = nn.BCELoss()
  optimizer = optim.Adam(model.parameters(), lr=LR)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  # validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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
    'model': 'PyTorch Binary Deep Belief Network',
    'file': 'train_dl_dbn.py',
    'hyperparameters': {
      'hidden_layers': HIDDEN_LAYERS,
      'num_epochs': NUM_EPOCHS,
      'embed_size': EMBED_SIZE,
      'num_layers': NUM_LAYERS,
      'heads': HEADS,
      'device': device,
      'forward_expansion': FORWARD_EXPANSION,
      'dropout': DROPOUT, 
      'max_length': MAX_LENGTH, 
      'lr': LR,
      'l2': L2,
      'output_dim': OUTPUT_DIM,
    },
    'accuracies': {
      OUTPUT: average_accuracy,
    },
  })

  # Save/Load the model
  torch.save(model.state_dict(), f'./models/nhl_ai_v{TORCH_FILE_VERSION}_dbn_{OUTPUT}.pt')



if __name__ == '__main__':
  train()