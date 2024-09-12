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
from constants.inputConstants import X_INPUTS_P, Y_OUTPUTS_P
from constants.constants import PROJECTED_LINEUP_VERSION, PROJECTED_LINEUP_FILE_VERSION, PROJECTED_LINEUP_TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, calculate_accuracy, binary_accuracy, errorAnalysis
import time


NUM_EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(X_INPUTS_P)
OUTPUT_DIM = len(Y_OUTPUTS_P)
HIDDEN_DIM = 142
KERNEL_SIZE = 32
IN_CHANNELS = 71
OUT_CHANNELS = 128

USE_PARTIAL_SEASONS = False

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
    self.y = torch.argmax(torch.from_numpy(y_train.astype(np.float32)),dim=1)
    self.len = self.x.shape[0]

  def __len__(self):
    return self.len

  def __getitem__(self, index):
    return self.x[index], self.y[index]

# print('~~~~~~~~~~~ Feature Count:',len(X_INPUTS_P))
HIDDEN_LAYERS = (
  len(X_INPUTS_P)*2,
  len(X_INPUTS_P)*4,
  len(X_INPUTS_P)*6,
  len(X_INPUTS_P)*8,
  len(X_INPUTS_P)*10,
  len(X_INPUTS_P)*8,
  len(X_INPUTS_P)*6,
  len(X_INPUTS_P)*4,
  )
class Net(nn.Module):
  def __init__(self, input_size, hidden_dim, output_size):
    super(Net, self).__init__()
    num_layers = len(HIDDEN_LAYERS)
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    
    # Define an LSTM layer
    self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_LAYERS[0], num_layers, batch_first=True)

    self.l2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
    self.act2 = nn.ReLU()

    self.l3 = nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2])
    self.act3 = nn.ReLU()

    self.l4 = nn.Linear(HIDDEN_LAYERS[2], HIDDEN_LAYERS[3])
    self.act4 = nn.ReLU()

    self.do1 = nn.Dropout(0.5)

    self.l5 = nn.Linear(HIDDEN_LAYERS[3], HIDDEN_LAYERS[4])
    self.act5 = nn.ReLU()

    self.l6 = nn.Linear(HIDDEN_LAYERS[4], HIDDEN_LAYERS[5])
    self.act6 = nn.ReLU()

    self.l7 = nn.Linear(HIDDEN_LAYERS[5], HIDDEN_LAYERS[6])
    self.act7 = nn.ReLU()

    self.l8 = nn.Linear(HIDDEN_LAYERS[6], HIDDEN_LAYERS[7])
    self.act8 = nn.ReLU()

    self.c1d1 = nn.Conv1d(in_channels=HIDDEN_LAYERS[7], out_channels=OUT_CHANNELS, kernel_size=KERNEL_SIZE)

    self.n2 = nn.BatchNorm1d(OUT_CHANNELS)

    self.do2 = nn.Dropout(0.5)

    # Define the output layer
    self.fc = nn.Linear(OUT_CHANNELS, OUTPUT_DIM)

  def forward(self, x):
    # Initialize hidden and cell states
    h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
    
    # Forward propagate the LSTM
    out, _ = self.lstm(x, (h0, c0))
    
    out = self.act2(self.l2(out))
    out = self.act3(self.l3(out))
    out = self.act4(self.l4(out))
    out = self.do1(out)
    out = self.act5(self.l5(out))
    out = self.act6(self.l6(out))
    out = self.act7(self.l7(out))
    out = self.act8(self.l8(out))
    out = out.transpose(1,2)
    out = self.c1d1(out)
    out = self.n2(out)
    out = out.transpose(2,1)
    # out = torch.flatten(out, start_dim=1)
    out = self.do2(out)

    # Pass the output of the last time step to the classifier
    out = self.fc(out[:, -1, :])
    # out = self.fc(out)
    return out

def train():
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
  db = client["hockey"]

  if USE_PARTIAL_SEASONS:
    seasons = [
      20172018,
      20182019,
      20192020,
      20202021,
      20212022,
      20222023,
    ]
    result = []
    for i in seasons:
      result.append(load(f'training_data/v{PROJECTED_LINEUP_VERSION}/projected_lineup/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_{i}.joblib'))
    training_data = np.concatenate(result).tolist()
  else:
    training_data = load(f'training_data/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib')
  
  future_test_data = load(f'test_data/test_data_v{PROJECTED_LINEUP_TEST_DATA_FILE_VERSION}_projectedLineup.joblib')
  
  data = pd.DataFrame(training_data)
  test_data = pd.DataFrame(future_test_data)
  x_train = data [X_INPUTS_P].to_numpy()
  y_train = data [Y_OUTPUTS_P].to_numpy()
  x_test = test_data [X_INPUTS_P].to_numpy()
  y_test = test_data [Y_OUTPUTS_P].to_numpy()
  # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.2, random_state=RANDOM_STATE)
  
  train_dataset = CustomDataset(x_train, y_train)
  validation_dataset = CustomDataset(x_validation, y_validation)
  test_dataset = CustomDataset(x_test, y_test)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = Net(input_size=INPUT_DIM, hidden_dim=HIDDEN_DIM, output_size=OUTPUT_DIM).to(device)


  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # Loss function and optimizer
  # criterion = nn.BCEWithLogitsLoss ()
  criterion = nn.CrossEntropyLoss ()
  # criterion = nn.MSELoss ()
  # criterion = HingeLoss ()
  # criterion = FocalLoss ()
  # optimizer = torch.optim.SGD(model.parameters(),lr=LR,weight_decay=L2)
  # optimizer = torch.optim.Adam(model.parameters(),lr=LR,weight_decay=L2)
  optimizer = optim.Adam(model.parameters(), lr=LR)
  
  # Training loop
  for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_accuracy = 0.0
    running_batches = 0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      inputs = inputs.unsqueeze(0)
      # set optimizer to zero grad to remove previous epoch gradients
      optimizer.zero_grad()
      # forward propagation
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      # backward propagation
      loss.backward()
      # optimize
      optimizer.step()
      print("Output shape:", outputs.shape)
      print("Labels shape:", labels.shape)
      running_loss += loss.item()
      running_accuracy += calculate_accuracy(outputs, labels)
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
      inputs = inputs.unsqueeze(0)
      labels = labels.unsqueeze(0)
      # calculate output by running through the network
      outputs = model(inputs)
      # get the predictions
      __, predicted = torch.max(outputs.data, 1)
      # update results
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      current_accuracy = calculate_accuracy(outputs, labels)
      total_accuracy += current_accuracy
      test_batches += 1

  average_accuracy = total_accuracy / test_batches
  print(f'Average Accuracy: {average_accuracy}')
  # print(f'Accuracy of the model on the test set: {100 * correct / total}%')

  # Error Analysis
  errorAnalysis(model,validation_loader,device)

  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'version': PROJECTED_LINEUP_VERSION,
    'inputs': X_INPUTS_P,
    'outputs': Y_OUTPUTS_P,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'projectedLineup': True,
    'model': 'PyTorch Artificial Neural Network Multi-Class Single-Label Classifier',
    'hyperparameters': {
      'hidden_layers': HIDDEN_LAYERS,
      'num_epochs': NUM_EPOCHS,
      'batch_size': BATCH_SIZE,
      'num_workers': NUM_WORKERS,
      'hidden_dim': HIDDEN_DIM,
      'kernel_size': KERNEL_SIZE,
      'in_channels': IN_CHANNELS,
      'out_channels': OUT_CHANNELS,
      'lr': LR,
      'l2': L2,
      'input_dim': INPUT_DIM,
      'output_dim': OUTPUT_DIM,
    },
    'accuracies': {
      'projectedLineup': average_accuracy,
    },
  })

  # Save/Load the model
  torch.save(model.state_dict(), f'./models/nhl_ai_v{PROJECTED_LINEUP_FILE_VERSION}_torch_projectedLineup.pt')



def predict_model(input_data=[],threshold=0.489,include_confidence=False):
  # Instantiate the model
  model = Net()

  # Load the trained model weights
  model.load_state_dict(torch.load(f'./models/nhl_ai_v{PROJECTED_LINEUP_FILE_VERSION}_torch_projectedLineup.pt'))

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
  print('confidence',confidence[0][0])
  prediction = (confidence >= threshold).int()

  if not include_confidence:
    return prediction
  else:
    return prediction, confidence




if __name__ == '__main__':
  train()