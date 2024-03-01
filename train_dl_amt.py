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

class SelfAttention(nn.Module):
  def __init__(self, embed_size, heads):
    super(SelfAttention, self).__init__()
    self.embed_size = embed_size
    self.heads = heads
    self.head_dim = embed_size // heads

    assert (
      self.head_dim * heads == embed_size
    ), "Embed size needs to be divisible by heads"

    self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
    self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

  def forward(self, value, key, query):
    N = query.shape[0]
    value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

    # Split the embedding into self.heads different pieces
    # print("Value shape:", value.shape)
    # print("Key shape:", key.shape)
    # print("Query shape:", query.shape)
    # print('N:', N)
    # print('value_len:', value_len)
    # print('self.heads:', self.heads)
    # print('self.head_dim:', self.head_dim)
    values = value.reshape(N, value_len, self.heads, self.head_dim)
    keys = key.reshape(N, key_len, self.heads, self.head_dim)
    queries = query.reshape(N, query_len, self.heads, self.head_dim)

    values = self.values(values)
    keys = self.keys(keys)
    queries = self.queries(queries)

    # Einsum does matrix multiplication for query*keys for each training example
    # with every other training example, don't be confused by einsum
    # it's just a way to do batch matrix multiplication
    energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
    attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

    out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
      N, query_len, self.heads * self.head_dim
    )

    out = self.fc_out(out)
    return out

class TransformerBlock(nn.Module):
  def __init__(self, embed_size, heads, dropout, forward_expansion):
    super(TransformerBlock, self).__init__()
    self.attention = SelfAttention(embed_size, heads)
    self.norm1 = nn.LayerNorm(embed_size)
    self.norm2 = nn.LayerNorm(embed_size)

    self.feed_forward = nn.Sequential(
      nn.Linear(embed_size, forward_expansion * embed_size),
      nn.ReLU(),
      nn.Linear(forward_expansion * embed_size, embed_size),
    )

    self.dropout = nn.Dropout(dropout)

  def forward(self, value, key, query):
    attention = self.attention(value, key, query)
    x = self.dropout(self.norm1(attention + query))
    forward = self.feed_forward(x)
    out = self.dropout(self.norm2(forward + x))
    return out

class Classifier(nn.Module):
  def __init__(self, input_dim, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, num_classes):
    super(Classifier, self).__init__()
    self.device = device
    
    # Add a Linear layer to project input vectors to the desired embedding size
    self.input_projection = nn.Linear(input_dim, embed_size)
    
    self.layers = nn.ModuleList(
      [
        TransformerBlock(
          embed_size,
          heads,
          dropout=dropout,
          forward_expansion=forward_expansion,
        )
        for _ in range(num_layers)
      ]
    )
    self.dropout = nn.Dropout(dropout)
    # Adjust the input to fc_out based on your architecture; might need changes based on how you process/pool the transformer outputs
    self.fc_out = nn.Linear(max_length * embed_size, num_classes)

  def forward(self, x):
    # Project input vectors to the embedding size
    x = self.input_projection(x)
    
    N, seq_length = x.shape[0], x.shape[1]
    
    for layer in self.layers:
      x = layer(x, x, x)

    x = x.flatten(start_dim=1)
    x = self.dropout(x)
    x = self.fc_out(x)
    return x


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
  model = Classifier(
    input_dim=len(X_INPUTS),
    embed_size=EMBED_SIZE,
    num_layers=NUM_LAYERS,
    heads=HEADS,
    device=device,
    forward_expansion=FORWARD_EXPANSION,
    dropout=DROPOUT, 
    max_length=x_train.shape[1], 
    num_classes=1  # For binary classification, output is 1
  ).to(device)

  # Loss and optimizer
  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(model.parameters(), lr=LR)

  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
  test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

  # Training loop
  running_total_accuracy = []
  for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    running_accuracy = 0.0
    running_batches = 0
    for batch_idx, (data, targets) in enumerate(train_loader):
      data = data.to(device)
      targets = targets.to(device)

      # forward pass
      scores = model(data)
      loss = criterion(scores, targets)

      # backward pass
      optimizer.zero_grad()
      loss.backward()
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
    'model': 'PyTorch Binary Transformer Model with Attention Mechanism Neural Network Classifier',
    'file': 'train_dl_amt.py',
    'hyperparameters': {
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
  torch.save(model.state_dict(), f'./models/nhl_ai_v{TORCH_FILE_VERSION}_rnn_{OUTPUT}.pt')



if __name__ == '__main__':
  train()