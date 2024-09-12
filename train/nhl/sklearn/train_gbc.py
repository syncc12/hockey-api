import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE

RE_PULL = True
UPDATE = False

N_ESTIMATORS=200
LEARNING_RATE=0.1
MAX_DEPTH=1

WINNER_N_ESTIMATORS=200
WINNER_LEARNING_RATE=0.01
WINNER_MAX_DEPTH=1

WINNER_B_N_ESTIMATORS=1000
WINNER_B_LEARNING_RATE=0.01
WINNER_B_MAX_DEPTH=50

HOME_SCORE_N_ESTIMATORS=150
HOME_SCORE_LEARNING_RATE=0.2
HOME_SCORE_MAX_DEPTH=MAX_DEPTH

AWAY_SCORE_N_ESTIMATORS=150
AWAY_SCORE_LEARNING_RATE=0.2
AWAY_SCORE_MAX_DEPTH=MAX_DEPTH

TOTAL_GOALS_N_ESTIMATORS=N_ESTIMATORS
TOTAL_GOALS_LEARNING_RATE=0.2
TOTAL_GOALS_MAX_DEPTH=1

GOAL_DIFFERENTIAL_N_ESTIMATORS=N_ESTIMATORS
GOAL_DIFFERENTIAL_LEARNING_RATE=0.2
GOAL_DIFFERENTIAL_MAX_DEPTH=1


def train_winner(x, y, n_estimators, learning_rate, max_depth):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE, verbose=VERBOSE)
  gbc.fit(x_train,y_train)
  predictions = gbc.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print("Winner Accuracy:", accuracy)
  dump(gbc, f'models/nhl_ai_v{FILE_VERSION}_gbc_winner.joblib')
  return accuracy

def train_winnerB(x, y, n_estimators, learning_rate, max_depth):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE, verbose=VERBOSE)
  gbc.fit(x_train,y_train)
  predictions = gbc.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print("WinnerB Accuracy:", accuracy)
  dump(gbc, f'models/nhl_ai_v{FILE_VERSION}_gbc_winnerB.joblib')
  return accuracy

def train_homeScore(x, y, n_estimators, learning_rate, max_depth):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE, verbose=VERBOSE)
  gbc.fit(x_train,y_train)
  predictions = gbc.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print("Home Score Accuracy:", accuracy)
  dump(gbc, f'models/nhl_ai_v{FILE_VERSION}_gbc_homeScore.joblib')
  return accuracy

def train_awayScore(x, y, n_estimators, learning_rate, max_depth):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE, verbose=VERBOSE)
  gbc.fit(x_train,y_train)
  predictions = gbc.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print("Away Score Accuracy:", accuracy)
  dump(gbc, f'models/nhl_ai_v{FILE_VERSION}_gbc_awayScore.joblib')
  return accuracy

def train_totalGoals(x, y, n_estimators, learning_rate, max_depth):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE, verbose=VERBOSE)
  gbc.fit(x_train,y_train)
  predictions = gbc.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print("Total Goals Accuracy:", accuracy)
  dump(gbc, f'models/nhl_ai_v{FILE_VERSION}_gbc_totalGoals.joblib')
  return accuracy

def train_goalDifferential(x, y, n_estimators, learning_rate, max_depth):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_STATE)
  gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=RANDOM_STATE, verbose=VERBOSE)
  gbc.fit(x_train,y_train)
  predictions = gbc.predict(x_test)
  accuracy = accuracy_score(y_test, predictions)
  print("Goal Differential Accuracy:", accuracy)
  dump(gbc, f'models/nhl_ai_v{FILE_VERSION}_gbc_goalDifferential.joblib')
  return accuracy

def train(db, inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  x = data [X_INPUTS]
  
  # y_winner = data [['winner']].values.ravel()
  y_winnerB = data [['winnerB']].values.ravel()
  # y_homeScore = data [['homeScore']].values.ravel()
  # y_awayScore = data [['awayScore']].values.ravel()
  # y_totalGoals = data [['totalGoals']].values.ravel()
  # y_goalDifferential = data [['goalDifferential']].values.ravel()
  imputer.fit(x)
  x = imputer.transform(x)
  # x_winner = x
  x_winnerB = x
  # x_homeScore = x
  # x_awayScore = x
  # x_totalGoals = x
  # x_goalDifferential = x

  # winner_accuracy = train_winner(x_winner, y_winner, WINNER_N_ESTIMATORS, WINNER_LEARNING_RATE, WINNER_MAX_DEPTH)
  winnerB_accuracy = train_winner(x_winnerB, y_winnerB, WINNER_B_N_ESTIMATORS, WINNER_B_LEARNING_RATE, WINNER_B_MAX_DEPTH)
  # homeScore_accuracy = train_homeScore(x_homeScore, y_homeScore, HOME_SCORE_N_ESTIMATORS, HOME_SCORE_LEARNING_RATE, HOME_SCORE_MAX_DEPTH)
  # awayScore_accuracy = train_awayScore(x_awayScore, y_awayScore, AWAY_SCORE_N_ESTIMATORS, AWAY_SCORE_LEARNING_RATE, AWAY_SCORE_MAX_DEPTH)
  # totalGoals_accuracy = train_totalGoals(x_totalGoals, y_totalGoals, TOTAL_GOALS_N_ESTIMATORS, TOTAL_GOALS_LEARNING_RATE, TOTAL_GOALS_MAX_DEPTH)
  # goalDifferential_accuracy = train_goalDifferential(x_goalDifferential, y_goalDifferential, GOAL_DIFFERENTIAL_N_ESTIMATORS, GOAL_DIFFERENTIAL_LEARNING_RATE, GOAL_DIFFERENTIAL_MAX_DEPTH)

  TrainingRecords = db['dev_training_records']
  # Metadata = db['dev_metadata']

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
    'constants': {
      # 'winner': {
      #   'n_estimators': WINNER_N_ESTIMATORS,
      #   'learning_rate': WINNER_LEARNING_RATE,
      #   'max_depth': WINNER_MAX_DEPTH,
      # },
      'winnerB': {
        'n_estimators': WINNER_B_N_ESTIMATORS,
        'learning_rate': WINNER_B_LEARNING_RATE,
        'max_depth': WINNER_B_MAX_DEPTH,
      },
      # 'homeScore': {
      #   'n_estimators': HOME_SCORE_N_ESTIMATORS,
      #   'learning_rate': HOME_SCORE_LEARNING_RATE,
      #   'max_depth': HOME_SCORE_MAX_DEPTH,
      # },
      # 'awayScore': {
      #   'n_estimators': AWAY_SCORE_N_ESTIMATORS,
      #   'learning_rate': AWAY_SCORE_LEARNING_RATE,
      #   'max_depth': AWAY_SCORE_MAX_DEPTH,
      # },
      # 'totalGoals': {
      #   'n_estimators': TOTAL_GOALS_N_ESTIMATORS,
      #   'learning_rate': TOTAL_GOALS_LEARNING_RATE,
      #   'max_depth': TOTAL_GOALS_MAX_DEPTH,
      # },
      # 'goalDifferential': {
      #   'n_estimators': GOAL_DIFFERENTIAL_N_ESTIMATORS,
      #   'learning_rate': GOAL_DIFFERENTIAL_LEARNING_RATE,
      #   'max_depth': GOAL_DIFFERENTIAL_MAX_DEPTH,
      # },
    },
    'model': 'Gradient Boosted Random Forest Classifier',
    'accuracies': {
      # 'winner': winner_accuracy,
      'winnerB': winnerB_accuracy,
      # 'homeScore': homeScore_accuracy,
      # 'awayScore': awayScore_accuracy,
      # 'totalGoals': totalGoals_accuracy,
      # 'goalDifferential': goalDifferential_accuracy,
    },
  })



client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')

train(db,training_data)