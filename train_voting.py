import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier,HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time
from sklearn.metrics import accuracy_score

MAX_ITER = 5000
VOTE_TYPE = 'hard'

OUTPUT = 'winnerB'

def train(db, inData, inTestData):
  data = pd.DataFrame(inData)
  test_data = pd.DataFrame(inTestData)
  x_train = data [X_INPUTS]
  y_train = data [[OUTPUT]].values.ravel()
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()

  # Create the individual classifiers
  svc = SVC(probability=True, random_state=RANDOM_STATE)
  rfc = RandomForestClassifier(random_state=RANDOM_STATE)
  # etc = ExtraTreeClassifier(random_state=RANDOM_STATE)
  # gbc = GradientBoostingClassifier(random_state=RANDOM_STATE)
  # dtc = DecisionTreeClassifier(random_state=RANDOM_STATE)
  # rfr = RandomForestRegressor(random_state=RANDOM_STATE)
  # dtr = DecisionTreeRegressor(random_state=RANDOM_STATE)
  knn = KNeighborsClassifier()
  mlp = MLPClassifier(max_iter=MAX_ITER, random_state=RANDOM_STATE)
  ESTIMATORS = [
    ('svc', svc),
    ('rfc', rfc),
    # ('etc', etc),
    ('knn', knn),
    ('mlp', mlp)
  ]
  clf = VotingClassifier(estimators=ESTIMATORS, voting=VOTE_TYPE)
  clf.fit(x_train, y_train)
  y_prediction = clf.predict(x_test)
  accuracy = accuracy_score(y_test, y_prediction)

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
    'voteType': VOTE_TYPE,
    'model': 'Voting Classifier',
    'estimators': [i[0] for i in ESTIMATORS],
    'maxIter': MAX_ITER,
    'accuracies': {
      OUTPUT: accuracy,
    },
  })

  
  dump(clf, f'models/nhl_ai_v{FILE_VERSION}_{VOTE_TYPE}_voting_{OUTPUT}.joblib')



client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
test_data = load(f'test_data/test_data_v{FILE_VERSION}.joblib')

train(db,training_data,test_data)