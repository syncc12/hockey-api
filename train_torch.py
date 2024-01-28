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
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy
import time

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]

NUM_EPOCHS = 100
BATCH_SIZE = 16
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(X_INPUTS)
OUTPUT_DIM = 1

training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
data = pd.DataFrame(training_data)
x = data [X_INPUTS].to_numpy()
y = data [['winnerB']].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  
class CustomDataset(Dataset):
  def __init__(self, x_train, y_train):
    self.x = torch.from_numpy(x_train.astype(np.float32))
    self.y = torch.from_numpy(y_train.astype(np.float32))
    self.len = self.x.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return self.x[index], self.y[index]


HIDDEN_LAYERS = (len(X_INPUTS)*5,len(X_INPUTS)*2)
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.linearStart = nn.Linear(INPUT_DIM, HIDDEN_LAYERS[0])
    self.actStart = nn.Sigmoid()

    self.linear2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
    self.act2 = nn.Sigmoid()

    self.norm3 = nn.BatchNorm1d(HIDDEN_LAYERS[1])

    self.dropout2 = nn.Dropout(0.5)

    self.linearEnd = nn.Linear(HIDDEN_LAYERS[1], OUTPUT_DIM)

  def forward(self, x):
    x = self.actStart(self.linearStart(x))
    x = self.act2(self.linear2(x))
    x = self.norm3(x)
    x = self.dropout2(x)
    x = self.linearEnd(x)
    return x

def train():
  train_dataset = CustomDataset(x_train, y_train)
  test_dataset = CustomDataset(x_test, y_test)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = Net().to(device)


  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # Loss function and optimizer
  # criterion = nn.BCEWithLogitsLoss ()
  # criterion = nn.MSELoss ()
  # criterion = HingeLoss ()
  criterion = FocalLoss ()
  optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=L2)

  # Training loop
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
    # display statistics
    print(f'[{epoch + 1}/{NUM_EPOCHS}] accuracy: {running_accuracy / running_batches:.2f}, loss: {running_loss / running_batches:.4f}')

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
  # print(f'Accuracy of the model on the test set: {100 * correct / total}%')

  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'version': VERSION,
    'inputs': X_INPUTS,
    'outputs': 'winnerB',
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'projectedLineup': False,
    'model': 'PyTorch Artificial Neural Network Classifier',
    'hyperparameters': {
      'hidden_layers': HIDDEN_LAYERS,
      'num_epochs': NUM_EPOCHS,
      'batch_size': BATCH_SIZE,
      'num_workers': NUM_WORKERS,
      'lr': LR,
      'l2': L2,
      'input_dim': INPUT_DIM,
      'output_dim': OUTPUT_DIM,
    },
    'accuracies': {
      'winnerB': average_accuracy,
    },
  })

  # Save/Load the model
  torch.save(model.state_dict(), f'./models/nhl_ai_v{FILE_VERSION}_torch_winnerB.pt')

if __name__ == '__main__':
  train()