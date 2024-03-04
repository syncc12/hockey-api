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
from constants.inputConstants import X_INPUTS, Y_OUTPUTS, X_INPUTS_T
from constants.constants import TORCH_VERSION, TORCH_FILE_VERSION, TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy, errorAnalysis
from util.torch_layers import MemoryModule, NoiseInjection
import time
from training_input import training_input, test_input
from util.team_helpers import away_rename, home_rename
from util.helpers import team_lookup

DRY_RUN = True

NUM_EPOCHS = 50
BATCH_SIZE = 16
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(X_INPUTS_T)
OUTPUT_DIM = 1

OUTPUT = 'winB'

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
class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    self.linearStart = nn.Linear(INPUT_DIM, HIDDEN_LAYERS[0])
    self.actStart = nn.ReLU()
    self.mm1 = MemoryModule(input_features=HIDDEN_LAYERS[0], memory_size=1000)
    self.ni1 = NoiseInjection(noise_level=0.1)

    self.norm1 = nn.BatchNorm1d(HIDDEN_LAYERS[0])
    self.dropout1 = nn.Dropout(0.3)

    self.linear2 = nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1])
    self.act2 = nn.ReLU()
    
    self.norm2 = nn.BatchNorm1d(HIDDEN_LAYERS[1])
    self.dropout2 = nn.Dropout(0.3)

    self.linear3 = nn.Linear(HIDDEN_LAYERS[1], HIDDEN_LAYERS[2])
    self.act3 = nn.ReLU()

    self.norm3 = nn.BatchNorm1d(HIDDEN_LAYERS[2])
    self.dropout3 = nn.Dropout(0.3)

    self.linear4 = nn.Linear(HIDDEN_LAYERS[2], HIDDEN_LAYERS[3])
    self.act4 = nn.ReLU()
    
    self.norm4 = nn.BatchNorm1d(HIDDEN_LAYERS[3])
    self.dropout4 = nn.Dropout(0.5)

    self.linear5 = nn.Linear(HIDDEN_LAYERS[3], HIDDEN_LAYERS[4])
    self.act5 = nn.ReLU()
    
    self.norm5 = nn.BatchNorm1d(HIDDEN_LAYERS[4])
    self.dropout5 = nn.Dropout(0.3)

    self.linear6 = nn.Linear(HIDDEN_LAYERS[4], HIDDEN_LAYERS[5])
    self.act6 = nn.ReLU()
    
    self.norm6 = nn.BatchNorm1d(HIDDEN_LAYERS[5])
    self.dropout6 = nn.Dropout(0.3)

    self.linear7 = nn.Linear(HIDDEN_LAYERS[5], HIDDEN_LAYERS[6])
    self.act7 = nn.ReLU()
    
    self.norm7 = nn.BatchNorm1d(HIDDEN_LAYERS[6])
    self.dropout7 = nn.Dropout(0.3)

    self.linear8 = nn.Linear(HIDDEN_LAYERS[6], HIDDEN_LAYERS[7])
    self.act8 = nn.ReLU()

    self.norm8 = nn.BatchNorm1d(HIDDEN_LAYERS[7])
    self.dropout8 = nn.Dropout(0.5)

    self.linearEnd = nn.Linear(HIDDEN_LAYERS[7], OUTPUT_DIM)

  def forward(self, x):
    x = self.actStart(self.linearStart(x))
    x = self.mm1(x)
    # x = self.ni1(x)
    # x = self.norm1(x)
    # x = self.dropout1(x)
    x = self.act2(self.linear2(x))
    # x = self.norm2(x)
    # x = self.dropout2(x)
    x = self.act3(self.linear3(x))
    # x = self.norm3(x)
    # x = self.dropout3(x)
    x = self.act4(self.linear4(x))
    x = self.norm4(x)
    x = self.dropout4(x)
    x = self.act5(self.linear5(x))
    # x = self.norm5(x)
    # x = self.dropout5(x)
    x = self.act6(self.linear6(x))
    # x = self.norm6(x)
    # x = self.dropout6(x)
    x = self.act7(self.linear7(x))
    # x = self.norm7(x)
    # x = self.dropout7(x)
    x = self.act8(self.linear8(x))
    x = self.norm8(x)
    x = self.dropout8(x)
    x = self.linearEnd(x)
    return x

def train():
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]

  seasons = [
    # 20052006,
    # 20062007,
    # 20072008,
    # 20082009,
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
    20222023,
  ]
  teamLookup = team_lookup(db)
  training_data = training_input(seasons)
  data1 = pd.DataFrame(training_data)
  data2 = pd.DataFrame(training_data)

  data1.rename(columns=home_rename, inplace=True)
  data1['winB'] = 1 - data1['winB']
  data1['lossB'] = 1 - data1['winB']
  data2.rename(columns=away_rename, inplace=True)
  data2['lossB'] = 1 - data2['winB']
  data = pd.concat([data1, data2], axis=0)
  data.reset_index(drop=True, inplace=True)
  teams = data.groupby('team')

  dtrains = {}
  dtrains_winB = {}
  dtrains_lossB = {}
  for team, team_data in teams:
    x_train = team_data [X_INPUTS_T].to_numpy()
    y_train = team_data [[OUTPUT]].to_numpy()
    y_train_winB = team_data [['winB']].to_numpy()
    y_train_lossB = team_data [['lossB']].to_numpy()
    dtrains[team] = (x_train,y_train)
    dtrains_winB[team] = (x_train,y_train_winB)
    dtrains_lossB[team] = (x_train,y_train_lossB)

  test_data = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
  test_data1 = pd.DataFrame(test_data)
  test_data2 = pd.DataFrame(test_data)
  test_data1.rename(columns=home_rename, inplace=True)
  test_data1['winB'] = 1 - test_data1['winB']
  test_data1['lossB'] = 1 - test_data1['winB']
  test_data2.rename(columns=away_rename, inplace=True)
  test_data2['lossB'] = 1 - test_data2['winB']
  test_data = pd.concat([test_data1, test_data2], axis=0)
  test_data.reset_index(drop=True, inplace=True)
  test_teams = test_data.groupby('team')

  dtests = {}
  dtests_winB = {}
  dtests_lossB = {}
  for team, team_data in test_teams:
    x_test = team_data [X_INPUTS_T].to_numpy()
    y_test = team_data [[OUTPUT]].to_numpy()
    y_test_winB = team_data [['winB']].to_numpy()
    y_test_lossB = team_data [['lossB']].to_numpy()
    dtests[team] = {'x_test':x_test,'y_test':y_test,'len':len(x_test)}
    dtests_winB[team] = {'x_test':x_test,'y_test':y_test_winB,'len':len(x_test)}
    dtests_lossB[team] = {'x_test':x_test,'y_test':y_test_lossB,'len':len(x_test)}

  for j, (team, dtrain) in enumerate(dtrains.items()):
    team_name = teamLookup[team]['abbrev']
    print('Team:',team_name)
    # data = pd.DataFrame(training_data)
    # x_train = data [X_INPUTS].to_numpy()
    # y_train = data [[OUTPUT]].to_numpy()
    x_train, y_train = dtrain

    x_test, y_test = dtests[team]['x_test'], dtests[team]['y_test']
    # x_test, y_test = test_input(X_INPUTS,[OUTPUT])

    # x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
    
    train_dataset = CustomDataset(x_train, y_train)
    # validation_dataset = CustomDataset(x_validation, y_validation)
    test_dataset = CustomDataset(x_test, y_test)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    # validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Loss function and optimizer
    # criterion = nn.BCEWithLogitsLoss ()
    # criterion = nn.MSELoss ()
    # nn.
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
      print(f'[{epoch + 1}/{NUM_EPOCHS}] {team_name} {OUTPUT} accuracy: {running_accuracy / running_batches:.2f}% ({running_average_accuracy:.2f}%), loss: {running_loss / running_batches:.4f}')

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
    # errorAnalysis(model,validation_loader,device)

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
      'projectedLineup': False,
      'teamName': team_name,
      'model': 'PyTorch Artificial Neural Network Classifier (Team)',
      'fileName': 'train_torch_team.py',
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
        OUTPUT: average_accuracy,
      },
    })

    if not DRY_RUN:
      # Save/Load the model
      torch.save(model.state_dict(), f'./models/nhl_ai_v{TORCH_FILE_VERSION}_torch_team{team}_{OUTPUT}.pt')



def predict_model(input_data=[],threshold=0.489,include_confidence=False):
  # Instantiate the model
  model = Net()

  # Load the trained model weights
  model.load_state_dict(torch.load(f'./models/nhl_ai_v{TORCH_FILE_VERSION}_torch_{OUTPUT}.pt'))

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