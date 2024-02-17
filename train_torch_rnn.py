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

USE_PARTIAL_SEASONS = False

OUTPUT = 'winnerB'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
class BinaryRNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers):
    super(BinaryRNN, self).__init__()
    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=False)
    self.fc = nn.Linear(hidden_size, 1)  # Output layer
  
  def forward(self, x):
    # x shape: (batch_size, seq_length, input_size)
    out, _ = self.rnn(x)
    # Decode the hidden state of the last time step
    out = self.fc(out[:, -1, :])
    out = torch.sigmoid(out)
    return out.squeeze(-1)

def train():
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]

  training_data = load(f'training_data/training_data_v{TORCH_FILE_VERSION}.joblib')
  future_test_data = load(f'test_data/test_data_v{TORCH_FILE_VERSION}.joblib')
  
  data = pd.DataFrame(training_data)
  test_data = pd.DataFrame(future_test_data)
  x_train = data [X_INPUTS].to_numpy()
  y_train = data [[OUTPUT]].to_numpy()
  x_test = test_data [X_INPUTS].to_numpy()
  y_test = test_data [[OUTPUT]].to_numpy()
  # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=0.2, random_state=RANDOM_STATE)
  
  train_dataset = CustomDataset(x_train, y_train)
  validation_dataset = CustomDataset(x_validation, y_validation)
  test_dataset = CustomDataset(x_test, y_test)

  
  model = BinaryRNN(input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, num_layers=len(HIDDEN_LAYERS)).to(device)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # Loss function and optimizer
  # criterion = nn.BCEWithLogitsLoss ()
  # criterion = nn.CrossEntropyLoss ()
  # criterion = nn.MSELoss ()
  # criterion = HingeLoss ()
  criterion = nn.BCELoss()
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
      sequences, labels = data
      sequences, labels = sequences.to(device), labels.to(device)
      # set optimizer to zero grad to remove previous epoch gradients
      optimizer.zero_grad()
      # forward propagation
      outputs = model(sequences).squeeze()
      loss = criterion(outputs, labels)
      
      loss.backward()
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
      predicted = (outputs > 0.5).float()
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
  errorAnalysis(model,validation_loader,device)

  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'version': TORCH_VERSION,
    'inputs': X_INPUTS,
    'outputs': Y_OUTPUTS,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'projectedLineup': False,
    'model': 'PyTorch Recurrent Artificial Neural Network Binary Classifier',
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
  torch.save(model.state_dict(), f'./models/nhl_ai_v{TORCH_FILE_VERSION}_torch_rnn_{OUTPUT}.pt')



def predict_model(input_data=[],threshold=0.489,include_confidence=False):
  # Instantiate the model
  model = Net()

  # Load the trained model weights
  model.load_state_dict(torch.load(f'./models/nhl_ai_v{TORCH_FILE_VERSION}_torch_rnn_{OUTPUT}.pt'))

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