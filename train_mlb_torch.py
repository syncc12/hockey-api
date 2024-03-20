import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from joblib import load
from pages.mlb.inputs import X_INPUTS_MLB, X_INPUTS_MLB_S, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_VERSION, FILE_MLB_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy, errorAnalysis
from util.torch_layers import MemoryModule, NoiseInjection
import time


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

USE_X_INPUTS_MLB = X_INPUTS_MLB

NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(USE_X_INPUTS_MLB)
OUTPUT_DIM = 1

USE_TEST_SPLIT = True

OUTPUT = 'winner'

class PredictionDataset(Dataset):
  def __init__(self, x):
    # print(x)
    x = np.array(x)
    # print(x)
    # self.x = torch.from_numpy(x.astype(np.float32))
    self.x = x.astype(np.float32)
    # print(self.x)
    self.len = self.x.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return self.x[index]

class CustomDataset(Dataset):
  def __init__(self, x_train, y_train):
    self.x = torch.from_numpy(x_train.astype(np.float32))
    self.y = torch.from_numpy(y_train.astype(np.float32))
    self.len = self.x.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return self.x[index], self.y[index]


# HIDDEN_LAYERS = [
#   len(USE_X_INPUTS_MLB)*2,
#   len(USE_X_INPUTS_MLB)*4,
#   len(USE_X_INPUTS_MLB)*6,
#   len(USE_X_INPUTS_MLB)*8,
#   len(USE_X_INPUTS_MLB)*10,
#   len(USE_X_INPUTS_MLB)*8,
#   len(USE_X_INPUTS_MLB)*6,
#   len(USE_X_INPUTS_MLB)*8,
#   len(USE_X_INPUTS_MLB)*10,
#   len(USE_X_INPUTS_MLB)*8,
#   len(USE_X_INPUTS_MLB)*6,
#   len(USE_X_INPUTS_MLB)*4,
# ]
HIDDEN_LAYERS = [
  len(USE_X_INPUTS_MLB)*2,
  len(USE_X_INPUTS_MLB)*1,
  len(USE_X_INPUTS_MLB)*2,
  len(USE_X_INPUTS_MLB)*1,
  # len(USE_X_INPUTS_MLB)*2,
  # len(USE_X_INPUTS_MLB)*1,
  # len(USE_X_INPUTS_MLB)*2,
  # len(USE_X_INPUTS_MLB)*1,
  # len(USE_X_INPUTS_MLB)*2,
  # len(USE_X_INPUTS_MLB)*1,
  # len(USE_X_INPUTS_MLB)*2,
  # len(USE_X_INPUTS_MLB)*1,
]
LAYERS = [
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0.5,'noise':0,'memory':0,'norm':1},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
  {'dropout':0,'noise':0,'memory':0,'norm':0},
]
LAYERS = [layer for layer in LAYERS for i in range(0,len(HIDDEN_LAYERS))]
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.layers = nn.ModuleDict()
    for i in range(0,len(HIDDEN_LAYERS)):
      if i == 0:
        hiddenLayer1 = INPUT_DIM
        hiddenLayer2 = HIDDEN_LAYERS[i]
      else:
        hiddenLayer1 = HIDDEN_LAYERS[i-1]
        hiddenLayer2 = HIDDEN_LAYERS[i]
      self.layers[f'linear{i}'] = nn.Linear(hiddenLayer1, hiddenLayer2)
      self.layers[f'act{i}'] = nn.ReLU()
      if LAYERS[i]['memory'] > 0:
        self.layers[f'mm{i}'] = MemoryModule(input_features=hiddenLayer2, memory_size=LAYERS[i]['memory'])
      if LAYERS[i]['noise'] > 0:
        self.layers[f'ni{i}'] = NoiseInjection(noise_level=LAYERS[i]['noise'])
      if LAYERS[i]['norm'] > 0:
        self.layers[f'norm{i}'] = nn.BatchNorm1d(hiddenLayer2)
      if LAYERS[i]['dropout'] > 0:
        self.layers[f'dropout{i}'] = nn.Dropout(LAYERS[i]['dropout'])
    self.layers['end'] = nn.Linear(HIDDEN_LAYERS[-1], OUTPUT_DIM)

  def forward(self, x):
    for i in range(len(HIDDEN_LAYERS)):
      x = self.layers[f'act{i}'](self.layers[f'linear{i}'](x))
      if LAYERS[i]['memory'] > 0:
        x = self.layers[f'mm{i}'](x)
      if LAYERS[i]['noise'] > 0:
        x = self.layers[f'ni{i}'](x)
      if LAYERS[i]['norm'] > 0:
        x = self.layers[f'norm{i}'](x)
      if LAYERS[i]['dropout'] > 0:
        x = self.layers[f'dropout{i}'](x)
    x = self.layers['end'](x)
    return x

def train(x_train,y_train,x_test,y_test):
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
  # print(f'Accuracy of the model on the test set: {100 * correct / total}%')

  # Error Analysis
  # errorAnalysis(model,device)

  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'sport': 'mlb',
    'savedAt': timestamp,
    'version': MLB_VERSION,
    'inputs': USE_X_INPUTS_MLB,
    'outputs': OUTPUT,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'projectedLineup': False,
    'model': 'PyTorch Artificial Neural Network Classifier',
    'hyperparameters': {
      'hidden_layers': HIDDEN_LAYERS,
      'layers': LAYERS,
      'num_epochs': NUM_EPOCHS,
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
  torch.save(model.state_dict(), f'./models/mlb_ai_v{FILE_MLB_VERSION}_torch_{OUTPUT}.pt')



def predict_model(input_data=[],threshold=0.489,include_confidence=False):
  # Instantiate the model
  model = Net()

  # Load the trained model weights
  model.load_state_dict(torch.load(f'./models/mlb_ai_v{FILE_MLB_VERSION}_torch_{OUTPUT}.pt'))

  # Set the model to evaluation mode
  model.eval()

  # Assume you have a function to preprocess your input data
  def preprocess_data(raw_data):
    # Implement preprocessing here (e.g., scaling, normalization)
    return PredictionDataset(raw_data)
    # return raw_data

  # Load and preprocess your input data
  # raw_data = ... (load your data here)
  processed_data = preprocess_data(input_data)

  # Convert your processed data to a PyTorch tensor
  input_tensor = torch.tensor(processed_data)

  # If you're using a GPU, move your data to the GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  input_tensor = input_tensor.to(device)
  model = model.to(device)

  # Make predictions
  with torch.no_grad():
    predictions = model(input_tensor)

  # Convert predictions to numpy array (if needed)
  # predictions = predictions.cpu().numpy()

  confidence = torch.sigmoid(predictions)
  # print('confidence',confidence[0][0])
  prediction = (confidence >= threshold).int()

  if not include_confidence:
    return prediction
  else:
    return prediction, confidence




if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  teamLookup = team_lookup(mlb_db,only_active_mlb=True)
  TRAINING_DATA = mlb_training_input(SEASONS)
  data = pd.DataFrame(TRAINING_DATA)
  for column in ENCODE_COLUMNS:
    encoder = load(f'pages/mlb/encoders/{column}_encoder.joblib')
    data = data[data[column] != -1]
    data[column] = encoder.transform(data[column])

  x_train = data [USE_X_INPUTS_MLB].to_numpy()
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

    x_test = test_data [USE_X_INPUTS_MLB].to_numpy()
    y_test = test_data [[OUTPUT]].to_numpy()

  train(x_train,y_train,x_test,y_test)