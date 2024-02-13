import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time

MAX_ITER = 1000

def train(db, inData, inTestData):
  data = pd.DataFrame(inData)
  test_data = pd.DataFrame(inTestData)
  x_train = data [X_INPUTS]
  y_train = data [['winnerB']].values.ravel()
  x_test = test_data [X_INPUTS]
  y_test = test_data [['winnerB']].values.ravel()
  # x_train, x_test, y_train, y_test = train_test_split(x, y_winner, test_size=0.3, random_state=RANDOM_STATE)
  base_learners = [
    ('svc', SVC(probability=True)),
    ('rfc', RandomForestClassifier()),
    ('knn', KNeighborsClassifier()),
    ('mlp', MLPClassifier(max_iter=MAX_ITER)),
  ]
  final_estimator = LogisticRegression()
  winnerB_stacked_clf = StackingClassifier(estimators=base_learners, final_estimator=final_estimator)
  winnerB_stacked_clf.fit(x_train, y_train)
  winnerB_accuracy = winnerB_stacked_clf.score(x_test, y_test)
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
    'model': 'Stacked Classifier',
    'accuracies': {
      'winnerB': winnerB_accuracy,
    },
  })

  
  dump(winnerB_stacked_clf, f'models/nhl_ai_v{FILE_VERSION}_stacked_winnerB_4.joblib')



client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
test_data = load(f'test_data/test_data_v{FILE_VERSION}.joblib')

train(db,training_data,test_data)