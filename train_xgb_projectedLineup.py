import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS_P, Y_OUTPUTS_P
from constants.constants import PROJECTED_LINEUP_VERSION, PROJECTED_LINEUP_FILE_VERSION, PROJECTED_LINEUP_TEST_DATA_VERSION, PROJECTED_LINEUP_TEST_DATA_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import time
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]
TRAINING_DATA = load(f'training_data/training_data_v{PROJECTED_LINEUP_FILE_VERSION}_projectedLineup.joblib')
TEST_DATA = load(f'test_data/test_data_v{PROJECTED_LINEUP_TEST_DATA_FILE_VERSION}_projectedLineup.joblib')


PARAMS = {
  'max_depth': 10,  # the maximum depth of each tree
  'device': 'cuda',
  'tree_method': 'hist',
  'eta': 0.01,  # the training step for each iteration
  # 'num_class': data[Y_OUTPUTS_P].nunique(),
  'objective': 'multi:softprob',
  'eval_metric': 'mlogloss'  # evaluation metric
}
EPOCHS = 10  # the number of training iterations

def train(db):

  accuracies = {}

  for output in Y_OUTPUTS_P:
    data = pd.DataFrame(TRAINING_DATA)
    test_data = pd.DataFrame(TEST_DATA)
    x_train = data [X_INPUTS_P]
    x_test = test_data [X_INPUTS_P]
    y_train = data[output]
    y_test = test_data[output]
    # y_train, _ = pd.factorize(y_train)
    # y_test, _ = pd.factorize(y_test)

    # Calculate the number of unique classes for the current output
    unique_classes = y_train.nunique()
    unique_labels = y_train.unique()
    print(f"Unique labels for forward1: {unique_labels}")
    PARAMS['num_class'] = unique_classes  # Make sure this is an int

    # Convert y_train and y_test to be 1D arrays if they are not already
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    print(f"Number of unique classes for {output}: {unique_classes}")
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtest = xgb.DMatrix(x_test, label=y_test)

    bst = xgb.train(PARAMS, dtrain, EPOCHS)

    preds = bst.predict(dtest)
    predictions = np.asarray([np.argmax(line) for line in preds])
    accuracy = accuracy_score(y_test, predictions)
    print(f"{output} Predictions: {preds}")
    print(f"{output} Accuracy: {'%.2f%%' % (accuracy * 100.0)}")

    accuracies[output] = accuracy

  print(accuracies)


  TrainingRecords = db['dev_training_records']

  timestamp = time.time()
  TrainingRecords.insert_one({
    'savedAt': timestamp,
    'lastTrainedId': TRAINING_DATA[len(TRAINING_DATA)-1]['id'],
    'version': PROJECTED_LINEUP_VERSION,
    'testVersion': PROJECTED_LINEUP_TEST_DATA_VERSION,
    'inputs': X_INPUTS_P,
    'outputs': Y_OUTPUTS_P,
    'randomState': RANDOM_STATE,
    'startingSeason': START_SEASON,
    'finalSeason': END_SEASON,
    'projectedLineup': True,
    'model': 'XGBoost Multi-Class Classifier',
    'params': PARAMS,
    'epochs': EPOCHS,
    'accuracies': accuracies,
  })

  
  # dump(bst, f'models/nhl_ai_v{PROJECTED_LINEUP_FILE_VERSION}_xgboost_projectedLineup.joblib')



train(db)

