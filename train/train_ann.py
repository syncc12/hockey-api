import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from joblib import load, dump
from constants.inputConstants import X_V4_INPUTS, Y_V4_OUTPUTS

VERSION = 3

training_data = load(f'training_data/training_data_v{VERSION}.joblib')

def train():
  data = pd.DataFrame(training_data)

  x = data [X_V4_INPUTS].values
  y = data [Y_V4_OUTPUTS].values

  model = Sequential()
  model.add(Dense(64, activation='relu', input_shape=(x.shape[1],)))
  model.add(Dropout(0.5))
  model.add(Dense(32, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

  model.compile(optimizer='adam',
                loss='binary_crossentropy',  # Binary crossentropy for binary classification
                metrics=['accuracy'])
  
  model.fit(x, y, epochs=10, batch_size=32)
  dump(model,f'models/nhl_ai_v{VERSION}_ann.joblib')
