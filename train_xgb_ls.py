import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import pandas as pd
import numpy as np
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, TEST_DATA_VERSION,TEST_DATA_FILE_VERSION, XGB_VERSION, XGB_FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from util.xgb_helpers import learning_rate_schedule, update_learning_rate, LearningRateScheduler
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json
from multiprocessing import Pool

# winnerB Accuracy: 53.89% | eta: 0.75 | max_depth: 20 | initial_lr: 0.2 | final_lr: 0.1 || Best So Far: Accuracy: 59.64% | eta: 0.01 | max_depth: 13 | initial_lr: 0.19 | final_lr: 0.019

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
TRAINING_DATA = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
# TRAINING_DATA = load(f'training_data/training_data_v{FILE_VERSION}_{START_SEASON}_{END_SEASON}.joblib')
TEST_DATA = load(f'test_data/test_data_v{TEST_DATA_FILE_VERSION}.joblib')


OUTPUT = 'winnerB'

TRIAL = True
DRY_RUN = False

NUM_BOOST_ROUND = 500
N_TRIALS = 100

INITIAL_LR = 0.1
FINAL_LR = round(INITIAL_LR*0.1, 2)

PARAMS = {
  'max_depth': 23,
  'eta': 0.18,
  'objective': 'binary:logistic',
  # 'objective': 'reg:logistic',
  'eval_metric': 'logloss',
  # 'eval_metric': 'aucpr',
  'device': 'cuda',
  'tree_method': 'hist',
}
EPOCHS = 10
THRESHOLD = 0.5

def train(db,params,initial_lr,final_lr,x_train,y_train,x_test,y_test,x_validate,y_validate,trial=False):
  if not trial:
    print('Inputs:', X_INPUTS)
    print('Output:', OUTPUT)
    print('Params:', params)
    print('Initial LR:', initial_lr)
    print('Final LR:', final_lr)

  dtrain = xgb.DMatrix(x_train, label=y_train)
  dtest = xgb.DMatrix(x_test, label=y_test)
  lr_scheduler = LearningRateScheduler(num_boost_round=NUM_BOOST_ROUND, initial_lr=initial_lr, final_lr=final_lr)
  bst = xgb.train(params, dtrain, EPOCHS, num_boost_round=NUM_BOOST_ROUND, callbacks=[lr_scheduler])

  preds = bst.predict(dtest)
  validate_accuracy = -1
  if isinstance(x_validate, pd.DataFrame):
    dvalidate = xgb.DMatrix(x_validate, label=y_validate)
    validation = bst.predict(dvalidate)
    validate_predictions = [1 if i > 0.5 else 0 for i in validation]
    validate_accuracy = accuracy_score(y_validate, validate_predictions)

  # Convert probabilities to binary output with a threshold of 0.5
  predictions = [1 if i > 0.5 else 0 for i in preds]
  # predictions = preds
  # predictions = [round(i) for i in preds]

  accuracy = accuracy_score(y_test, predictions)
  # model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)} | eta: {eta} | max_depth: {max_depth} | epochs: {epochs}"
  if not trial:
    model_data = f"Accuracy: {'%.2f%%' % (accuracy * 100.0)}"
    # records.append(model_data)
    # f.write(f'{model_data}\n')
    print(model_data)

    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'savedAt': timestamp,
      'lastTrainedId': TRAINING_DATA[len(TRAINING_DATA)-1]['id'],
      'version': VERSION,
      'XGBVersion': XGB_VERSION,
      'testDataVersion': TEST_DATA_VERSION,
      'inputs': X_INPUTS,
      'outputs': Y_OUTPUTS,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'model': 'XGBoost Classifier (Learning Schedule)',
      'threshold': THRESHOLD,
      'params': params,
      'initial_lr': initial_lr,
      'final_lr': final_lr,
      'epochs': EPOCHS,
      'accuracies': {
        OUTPUT: accuracy,
      },
    })
    if not DRY_RUN:
      dump(bst, f'models/nhl_ai_v{XGB_FILE_VERSION}_xgboost_ls_{OUTPUT}.joblib')
  return accuracy, validate_accuracy

def trial_loop(*args):
  # print(args)
  x_train = args[0][0]
  y_train = args[0][1]
  x_test = args[0][2]
  y_test = args[0][3]
  x_validate = args[0][4]
  y_validate = args[0][5]
  max_depth = args[1]
  eta = args[2]
  initial_lr = args[3]
  final_lrp = args[4]
  # # for max_depth,eta,initial_lr,final_lrp in zip(range(10,101),np.arange(0.01, 0.91, 0.01),np.arange(0.01, 0.31, 0.01),np.arange(0.1, 0.6, 0.1)):
  initial_lr = round(initial_lr, 3)
  final_lr = round(initial_lr*final_lrp, 4)
  params = {
    'max_depth': max_depth,  # the maximum depth of each tree
    'eta': eta,  # the training step for each iteration
    'objective': 'binary:logistic',  # binary classification
    'eval_metric': 'aucpr',  # evaluation metric
    'device': 'cuda',
    'tree_method': 'hist',
  }
  accuracy,validation = train(db,params,initial_lr,final_lr,x_train,y_train,x_test,y_test,x_validate=x_validate,y_validate=y_validate,trial=True)
  p_accuracy = f'{OUTPUT} Accuracy:{(accuracy*100):.2f}%|Validation:{(validation*100):.2f}%|eta:{eta}|max_depth:{max_depth}|initial_lr:{initial_lr}|final_lr: {final_lr}'
  print(p_accuracy)
  return {
    'accuracy': accuracy,
    'validation': validation,
    'params': params,
    'initial_lr': initial_lr,
    'final_lr': final_lr,
  }

if __name__ == '__main__':
  data = pd.DataFrame(TRAINING_DATA)
  test_data = pd.DataFrame(TEST_DATA)
  data = data.sort_values(by='id')
  test_data = test_data.sort_values(by='id')
  x_train = data [X_INPUTS]
  y_train = data [[OUTPUT]].values.ravel()
  x_test = test_data [X_INPUTS]
  y_test = test_data [[OUTPUT]].values.ravel()
  x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_STATE)
  if TRIAL:
    best = {
      'max_depth': 0,
      'eta': 0,
      'accuracy': 0,
    }
    best_initial_lr = 0
    best_final_lr = 0
    # pool = Pool(processes=4)
    # result = pool.apply_async(trial_loop, args=(eta,max_depth,x_train,y_train,x_test,y_test,x_validate,y_validate))
    # max_depth = range(10,101)
    # eta = np.arange(0.01, 0.91, 0.01)
    # initial_lr = np.arange(0.01, 0.31, 0.01)
    # final_lrp = np.arange(0.1, 0.6, 0.1)
    # args = zip((x_train,y_train,x_test,y_test,x_validate,y_validate),max_depth,eta,initial_lr,final_lrp)
    # result = pool.starmap(trial_loop,args)
    for max_depth in range(10,101):
      for eta in np.arange(0.01, 0.91, 0.01):
        for initial_lr in np.arange(0.01, 0.31, 0.01):
          for final_lrp in np.arange(0.1, 0.6, 0.1):
            initial_lr = round(initial_lr, 3)
            final_lr = round(initial_lr*final_lrp, 4)
            params = {
              'max_depth': max_depth,  # the maximum depth of each tree
              'eta': eta,  # the training step for each iteration
              'objective': 'binary:logistic',  # binary classification
              'eval_metric': 'aucpr',  # evaluation metric
              'device': 'cuda',
              'tree_method': 'hist',
            }
            accuracy, validation = train(db,params,initial_lr,final_lr,x_train,y_train,x_test,y_test,x_validate=x_validate,y_validate=y_validate,trial=True)
            if accuracy > best['accuracy']:
              best['max_depth'] = max_depth
              best['eta'] = eta
              best['accuracy'] = accuracy
              best_initial_lr = initial_lr
              best_final_lr = final_lr
            p_accuracy = f'{OUTPUT} Accuracy:{(accuracy*100):.2f}%|Validation:{(validation*100):.2f}%|eta:{eta}|max_depth:{max_depth}|initial_lr:{initial_lr}|final_lr: {final_lr}'
            p_best = f'Best So Far: Accuracy:{(best["accuracy"]*100):.2f}%|eta:{best["eta"]}|max_depth:{best["max_depth"]}|initial_lr:{best_initial_lr}|final_lr: {best_final_lr}'
            p_validation = f'Validation: Accuracy:{(accuracy*100):.2f}%|eta:{eta}|max_depth:{max_depth}|initial_lr:{initial_lr}|final_lr:{final_lr}'
            print(f'{p_accuracy}||{p_best}')
    best_params = {
      'max_depth': best['max_depth'],
      'eta': best['eta'],
      'objective': 'binary:logistic',
      'eval_metric': params['eval_metric'],
      'device': params['device'],
      'tree_method': params['tree_method'],
    }
    
    # result = np.concatenate(result).tolist()
    # pool.close()
    # best_result = max(result, key=lambda x: x['accuracy'])
    # print('Best Result:', best_result)

    best_params = {
      'max_depth': best['max_depth'],
      'eta': best['eta'],
      'objective': 'binary:logistic',
      'eval_metric': 'aucpr',
      'device': 'cuda',
      'tree_method': 'hist',
    }
    train(db,best_params,best_initial_lr,best_final_lr,x_train,y_train,x_test,y_test,trial=False)
  else:
    train(db,PARAMS,INITIAL_LR,FINAL_LR,x_train,y_train,x_test,y_test,trial=False)

