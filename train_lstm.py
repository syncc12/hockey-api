import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import load
from constants.inputConstants import X_INPUTS, X_INPUTS_S, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from training_input import training_input, test_input
import pandas as pd
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]


SEASONS = [
  20052006,
  20062007,
  20072008,
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
  20222023,
]

USE_X_INPUTS = X_INPUTS

NUM_EPOCHS = 100
BATCH_SIZE = 1
MAX_SEQUENCE_LENGTH = 100
NUM_WORKERS = 4
LR=0.001
L2=1e-4
INPUT_DIM = len(USE_X_INPUTS)
OUTPUT_DIM = 1

N = 20

OUTPUT = 'winnerB'


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
  TRAINING_DATA = training_input(SEASONS)
  data = pd.DataFrame(TRAINING_DATA)

  x_train = data [USE_X_INPUTS]
  y_train = data [[OUTPUT]].to_numpy()

  TEST_DATA = test_input(no_format=True)
  test_data = pd.DataFrame(TEST_DATA)

  x_test = test_data [USE_X_INPUTS]
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