import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
import sklearn
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time

N_ESTIMATORS=5000

X_DATA = 'winnerB'

def train(db, inData):
  data = pd.DataFrame(inData)
  x = data [X_INPUTS]
  y = data [[X_DATA]].values.ravel()
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)
  ada = AdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
  ada.fit(x_train, y_train)
  predictions  = ada.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print(f"Accuracy: {accuracy}")
  
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
    'model': 'Ada Boost Classifier',
    'constants': {
      X_DATA: {
        'n_estimators': N_ESTIMATORS,
      },
    },
    'accuracies': {
      X_DATA: accuracy,
    },
  })

  
  dump(ada, f'models/nhl_ai_v{FILE_VERSION}_ada_{X_DATA}.joblib')



client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')

train(db,training_data)