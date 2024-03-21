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
from sklearn.metrics import accuracy_score
from joblib import load
from pages.mlb.inputs import X_INPUTS_MLB, X_INPUTS_MLB_S, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_VERSION, FILE_MLB_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy, errorAnalysis
from util.torch_layers import MemoryModule, NoiseInjection
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


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

NUM_EPOCHS = 100
BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 100
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(USE_X_INPUTS_MLB)
OUTPUT_DIM = 1

N = 10

USE_TEST_SPLIT = True

OUTPUT = 'winner'



def train(x_train,y_train,x_test,y_test):
  # Define the model
  model = Sequential([
    LSTM(50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
  ])

  # Compile the model
  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  # Print model summary
  model.summary()

  # Train the model (using mock data here, replace with your dataset)
  model.fit(x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

  preds = model.predict(x_test)
  predictions = [1 if i > 0.5 else 0 for i in preds]
  accuracy = accuracy_score(y_test, predictions)
  print(f'Accuracy: {accuracy}')


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

  x_train = data [USE_X_INPUTS_MLB]
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

    x_test = test_data [USE_X_INPUTS_MLB]
    y_test = test_data [[OUTPUT]].to_numpy()

  x_train_s = []
  y_train_s = []
  for i in range(len(x_train) - N):
    sequence = x_train.iloc[i:i+N].to_numpy()
    x_train_s.append(sequence)

  x_train = np.array(x_train_s)
  y_train = np.squeeze(y_train[N:])

  x_test_s = []
  y_test_s = []
  for i in range(len(x_test) - N):
    sequence = x_test.iloc[i:i+N].to_numpy()
    x_test_s.append(sequence)

  x_test = np.array(x_test_s)
  y_test = np.squeeze(y_test[N:])

  train(x_train,y_train,x_test,y_test)