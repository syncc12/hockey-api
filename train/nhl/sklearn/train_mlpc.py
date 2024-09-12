import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from joblib import dump, load
from sklearn.impute import SimpleImputer
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time

HIDDEN_LAYER_SIZES=(50,100,150,100,50)
MAX_ITER=10000
ACTIVATION='tanh'
SOLVER='adam'
LEARNING_RATE=''
EARLY_STOPPING=True

X_DATA = 'winnerB'

def train(db, inData):
  data = pd.DataFrame(inData)
  x = data [X_INPUTS]
  y_winnerB = data [[X_DATA]].values.ravel()
  x_train, x_test, y_train, y_test = train_test_split(x, y_winnerB, test_size=0.3, random_state=RANDOM_STATE)
  winnerB_mlpc = MLPClassifier(hidden_layer_sizes=HIDDEN_LAYER_SIZES, max_iter=MAX_ITER, activation=ACTIVATION, solver=SOLVER, early_stopping=EARLY_STOPPING, verbose=VERBOSE, random_state=RANDOM_STATE)
  winnerB_mlpc.fit(x_train, y_train)
  y_pred = winnerB_mlpc.predict(x_test)
  winnerB_accuracy = accuracy_score(y_test, y_pred)
  print(f"Accuracy: {winnerB_accuracy}")
  
  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'lastTrainedId': inData[len(inData)-1]['id'],
    'version': VERSION,
    'inputs': X_INPUTS,
    'outputs': Y_OUTPUTS,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'model': 'Multi-layer Perceptron Classifier',
    'constants': {
      X_DATA: {
        'hidden_layer_sizes': HIDDEN_LAYER_SIZES,
        'max_iter': MAX_ITER,
        'activation': ACTIVATION,
        'solver': SOLVER,
        'learning_rate': LEARNING_RATE,
        'early_stopping': EARLY_STOPPING,
      },
    },
    'accuracies': {
      X_DATA: winnerB_accuracy,
    },
  })

  
  dump(winnerB_mlpc, f'models/nhl_ai_v{FILE_VERSION}_mlpc_{X_DATA}.joblib')



client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')

train(db,training_data)