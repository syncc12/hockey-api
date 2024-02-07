import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

import requests
import json
from pymongo import MongoClient
import math
from datetime import datetime
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from joblib import dump, load
import pandas as pd
from multiprocessing import Pool
from util.training_data import season_training_data, game_training_data, update_training_data
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from util.helpers import latestIDs
import time
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE

RE_PULL = False
UPDATE = False
TEST_DATA = False

def train(db, inData):
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  data = pd.DataFrame(inData)
  x = data [X_INPUTS]
  # y = data [['homeScore','awayScore','winner','totalGoals','goalDifferential']].values
  y_winner = data [['winner']].values.ravel()
  y_winnerB = data [['winnerB']].values.ravel()
  y_homeScore = data [['homeScore']].values.ravel()
  y_awayScore = data [['awayScore']].values.ravel()
  y_totalGoals = data [['totalGoals']].values.ravel()
  y_goalDifferential = data [['goalDifferential']].values.ravel()
  # y_finalPeriod = data [['finalPeriod']].values.ravel()
  # y_pastRegulation = data [['pastRegulation']].values.ravel()
  # y_awayShots = data [['awayShots']].values.ravel()
  # y_homeShots = data [['homeShots']].values.ravel()
  # y_awayShotsPeriod1 = data [['awayShotsPeriod1']].values.ravel()
  # y_homeShotsPeriod1 = data [['homeShotsPeriod1']].values.ravel()
  # y_awayShotsPeriod2 = data [['awayShotsPeriod2']].values.ravel()
  # y_homeShotsPeriod2 = data [['homeShotsPeriod2']].values.ravel()
  # y_awayShotsPeriod3 = data [['awayShotsPeriod3']].values.ravel()
  # y_homeShotsPeriod3 = data [['homeShotsPeriod3']].values.ravel()
  # y_awayShotsPeriod4 = data [['awayShotsPeriod4']].values.ravel()
  # y_homeShotsPeriod4 = data [['homeShotsPeriod4']].values.ravel()
  # y_awayShotsPeriod5 = data [['awayShotsPeriod5']].values.ravel()
  # y_homeShotsPeriod5 = data [['homeShotsPeriod5']].values.ravel()
  # y_awayScorePeriod1 = data [['awayScorePeriod1']].values.ravel()
  # y_homeScorePeriod1 = data [['homeScorePeriod1']].values.ravel()
  # y_awayScorePeriod2 = data [['awayScorePeriod2']].values.ravel()
  # y_homeScorePeriod2 = data [['homeScorePeriod2']].values.ravel()
  # y_awayScorePeriod3 = data [['awayScorePeriod3']].values.ravel()
  # y_homeScorePeriod3 = data [['homeScorePeriod3']].values.ravel()
  # y_awayScorePeriod4 = data [['awayScorePeriod4']].values.ravel()
  # y_homeScorePeriod4 = data [['homeScorePeriod4']].values.ravel()
  # y_awayScorePeriod5 = data [['awayScorePeriod5']].values.ravel()
  # y_homeScorePeriod5 = data [['homeScorePeriod5']].values.ravel()
  # y_period1PuckLine = data [['period1PuckLine']].values.ravel()
  # y_period2PuckLine = data [['period2PuckLine']].values.ravel()
  # y_period3PuckLine = data [['period3PuckLine']].values.ravel()
  imputer.fit(x)
  x = imputer.transform(x)
  x_winner = x
  x_winnerB = x
  x_homeScore = x
  x_awayScore = x
  x_totalGoals = x
  x_goalDifferential = x
  # x_finalPeriod = x
  # x_pastRegulation = x
  # x_awayShots = x
  # x_homeShots = x
  # x_awayShotsPeriod1 = x
  # x_homeShotsPeriod1 = x
  # x_awayShotsPeriod2 = x
  # x_homeShotsPeriod2 = x
  # x_awayShotsPeriod3 = x
  # x_homeShotsPeriod3 = x
  # x_awayShotsPeriod4 = x
  # x_homeShotsPeriod4 = x
  # x_awayShotsPeriod5 = x
  # x_homeShotsPeriod5 = x
  # x_awayScorePeriod1 = x
  # x_homeScorePeriod1 = x
  # x_awayScorePeriod2 = x
  # x_homeScorePeriod2 = x
  # x_awayScorePeriod3 = x
  # x_homeScorePeriod3 = x
  # x_awayScorePeriod4 = x
  # x_homeScorePeriod4 = x
  # x_awayScorePeriod5 = x
  # x_homeScorePeriod5 = x
  # x_period1PuckLine = x
  # x_period2PuckLine = x
  # x_period3PuckLine = x

  x_train_winner, x_test_winner, y_train_winner, y_test_winner = train_test_split(x_winner, y_winner, test_size=0.2, random_state=RANDOM_STATE)
  x_train_winnerB, x_test_winnerB, y_train_winnerB, y_test_winnerB = train_test_split(x_winnerB, y_winnerB, test_size=0.2, random_state=RANDOM_STATE)
  x_train_homeScore, x_test_homeScore, y_train_homeScore, y_test_homeScore = train_test_split(x_homeScore, y_homeScore, test_size=0.2, random_state=RANDOM_STATE)
  x_train_awayScore, x_test_awayScore, y_train_awayScore, y_test_awayScore = train_test_split(x_awayScore, y_awayScore, test_size=0.2, random_state=RANDOM_STATE)
  x_train_totalGoals, x_test_totalGoals, y_train_totalGoals, y_test_totalGoals = train_test_split(x_totalGoals, y_totalGoals, test_size=0.2, random_state=RANDOM_STATE)
  x_train_goalDifferential, x_test_goalDifferential, y_train_goalDifferential, y_test_goalDifferential = train_test_split(x_goalDifferential, y_goalDifferential, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_finalPeriod, x_test_finalPeriod, y_train_finalPeriod, y_test_finalPeriod = train_test_split(x_finalPeriod, y_finalPeriod, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_pastRegulation, x_test_pastRegulation, y_train_pastRegulation, y_test_pastRegulation = train_test_split(x_pastRegulation, y_pastRegulation, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayShots, x_test_awayShots, y_train_awayShots, y_test_awayShots = train_test_split(x_awayShots, y_awayShots, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeShots, x_test_homeShots, y_train_homeShots, y_test_homeShots = train_test_split(x_homeShots, y_homeShots, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayShotsPeriod1, x_test_awayShotsPeriod1, y_train_awayShotsPeriod1, y_test_awayShotsPeriod1 = train_test_split(x_awayShotsPeriod1, y_awayShotsPeriod1, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeShotsPeriod1, x_test_homeShotsPeriod1, y_train_homeShotsPeriod1, y_test_homeShotsPeriod1 = train_test_split(x_homeShotsPeriod1, y_homeShotsPeriod1, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayShotsPeriod2, x_test_awayShotsPeriod2, y_train_awayShotsPeriod2, y_test_awayShotsPeriod2 = train_test_split(x_awayShotsPeriod2, y_awayShotsPeriod2, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeShotsPeriod2, x_test_homeShotsPeriod2, y_train_homeShotsPeriod2, y_test_homeShotsPeriod2 = train_test_split(x_homeShotsPeriod2, y_homeShotsPeriod2, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayShotsPeriod3, x_test_awayShotsPeriod3, y_train_awayShotsPeriod3, y_test_awayShotsPeriod3 = train_test_split(x_awayShotsPeriod3, y_awayShotsPeriod3, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeShotsPeriod3, x_test_homeShotsPeriod3, y_train_homeShotsPeriod3, y_test_homeShotsPeriod3 = train_test_split(x_homeShotsPeriod3, y_homeShotsPeriod3, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayShotsPeriod4, x_test_awayShotsPeriod4, y_train_awayShotsPeriod4, y_test_awayShotsPeriod4 = train_test_split(x_awayShotsPeriod4, y_awayShotsPeriod4, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeShotsPeriod4, x_test_homeShotsPeriod4, y_train_homeShotsPeriod4, y_test_homeShotsPeriod4 = train_test_split(x_homeShotsPeriod4, y_homeShotsPeriod4, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayShotsPeriod5, x_test_awayShotsPeriod5, y_train_awayShotsPeriod5, y_test_awayShotsPeriod5 = train_test_split(x_awayShotsPeriod5, y_awayShotsPeriod5, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeShotsPeriod5, x_test_homeShotsPeriod5, y_train_homeShotsPeriod5, y_test_homeShotsPeriod5 = train_test_split(x_homeShotsPeriod5, y_homeShotsPeriod5, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayScorePeriod1, x_test_awayScorePeriod1, y_train_awayScorePeriod1, y_test_awayScorePeriod1 = train_test_split(x_awayScorePeriod1, y_awayScorePeriod1, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeScorePeriod1, x_test_homeScorePeriod1, y_train_homeScorePeriod1, y_test_homeScorePeriod1 = train_test_split(x_homeScorePeriod1, y_homeScorePeriod1, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayScorePeriod2, x_test_awayScorePeriod2, y_train_awayScorePeriod2, y_test_awayScorePeriod2 = train_test_split(x_awayScorePeriod2, y_awayScorePeriod2, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeScorePeriod2, x_test_homeScorePeriod2, y_train_homeScorePeriod2, y_test_homeScorePeriod2 = train_test_split(x_homeScorePeriod2, y_homeScorePeriod2, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayScorePeriod3, x_test_awayScorePeriod3, y_train_awayScorePeriod3, y_test_awayScorePeriod3 = train_test_split(x_awayScorePeriod3, y_awayScorePeriod3, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeScorePeriod3, x_test_homeScorePeriod3, y_train_homeScorePeriod3, y_test_homeScorePeriod3 = train_test_split(x_homeScorePeriod3, y_homeScorePeriod3, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayScorePeriod4, x_test_awayScorePeriod4, y_train_awayScorePeriod4, y_test_awayScorePeriod4 = train_test_split(x_awayScorePeriod4, y_awayScorePeriod4, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeScorePeriod4, x_test_homeScorePeriod4, y_train_homeScorePeriod4, y_test_homeScorePeriod4 = train_test_split(x_homeScorePeriod4, y_homeScorePeriod4, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_awayScorePeriod5, x_test_awayScorePeriod5, y_train_awayScorePeriod5, y_test_awayScorePeriod5 = train_test_split(x_awayScorePeriod5, y_awayScorePeriod5, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_homeScorePeriod5, x_test_homeScorePeriod5, y_train_homeScorePeriod5, y_test_homeScorePeriod5 = train_test_split(x_homeScorePeriod5, y_homeScorePeriod5, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_period1PuckLine, x_test_period1PuckLine, y_train_period1PuckLine, y_test_period1PuckLine = train_test_split(x_period1PuckLine, y_period1PuckLine, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_period2PuckLine, x_test_period2PuckLine, y_train_period2PuckLine, y_test_period2PuckLine = train_test_split(x_period2PuckLine, y_period2PuckLine, test_size=0.2, random_state=RANDOM_STATE)
  # x_train_period3PuckLine, x_test_period3PuckLine, y_train_period3PuckLine, y_test_period3PuckLine = train_test_split(x_period3PuckLine, y_period3PuckLine, test_size=0.2, random_state=RANDOM_STATE)

  # clf = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_winner = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_winnerB = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_homeScore = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_awayScore = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_totalGoals = RandomForestClassifier(random_state=RANDOM_STATE)
  clf_goalDifferential = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_finalPeriod = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_pastRegulation = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayShots = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeShots = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayShotsPeriod1 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeShotsPeriod1 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayShotsPeriod2 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeShotsPeriod2 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayShotsPeriod3 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeShotsPeriod3 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayShotsPeriod4 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeShotsPeriod4 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayShotsPeriod5 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeShotsPeriod5 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayScorePeriod1 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeScorePeriod1 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayScorePeriod2 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeScorePeriod2 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayScorePeriod3 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeScorePeriod3 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayScorePeriod4 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeScorePeriod4 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_awayScorePeriod5 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_homeScorePeriod5 = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_period1PuckLine = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_period2PuckLine = RandomForestClassifier(random_state=RANDOM_STATE)
  # clf_period3PuckLine = RandomForestClassifier(random_state=RANDOM_STATE)

  # clf.fit(x,y)
  clf_winner.fit(x_train_winner,y_train_winner)
  clf_winnerB.fit(x_train_winnerB,y_train_winnerB)
  clf_homeScore.fit(x_train_homeScore,y_train_homeScore)
  clf_awayScore.fit(x_train_awayScore,y_train_awayScore)
  clf_totalGoals.fit(x_train_totalGoals,y_train_totalGoals)
  clf_goalDifferential.fit(x_train_goalDifferential,y_train_goalDifferential)
  # clf_finalPeriod.fit(x_train_finalPeriod,y_train_finalPeriod)
  # clf_pastRegulation.fit(x_train_pastRegulation,y_train_pastRegulation)
  # clf_awayShots.fit(x_train_awayShots,y_train_awayShots)
  # clf_homeShots.fit(x_train_homeShots,y_train_homeShots)
  # clf_awayShotsPeriod1.fit(x_train_awayShotsPeriod1,y_train_awayShotsPeriod1)
  # clf_homeShotsPeriod1.fit(x_train_homeShotsPeriod1,y_train_homeShotsPeriod1)
  # clf_awayShotsPeriod2.fit(x_train_awayShotsPeriod2,y_train_awayShotsPeriod2)
  # clf_homeShotsPeriod2.fit(x_train_homeShotsPeriod2,y_train_homeShotsPeriod2)
  # clf_awayShotsPeriod3.fit(x_train_awayShotsPeriod3,y_train_awayShotsPeriod3)
  # clf_homeShotsPeriod3.fit(x_train_homeShotsPeriod3,y_train_homeShotsPeriod3)
  # clf_awayShotsPeriod4.fit(x_train_awayShotsPeriod4,y_train_awayShotsPeriod4)
  # clf_homeShotsPeriod4.fit(x_train_homeShotsPeriod4,y_train_homeShotsPeriod4)
  # clf_awayShotsPeriod5.fit(x_train_awayShotsPeriod5,y_train_awayShotsPeriod5)
  # clf_homeShotsPeriod5.fit(x_train_homeShotsPeriod5,y_train_homeShotsPeriod5)
  # clf_awayScorePeriod1.fit(x_train_awayScorePeriod1,y_train_awayScorePeriod1)
  # clf_homeScorePeriod1.fit(x_train_homeScorePeriod1,y_train_homeScorePeriod1)
  # clf_awayScorePeriod2.fit(x_train_awayScorePeriod2,y_train_awayScorePeriod2)
  # clf_homeScorePeriod2.fit(x_train_homeScorePeriod2,y_train_homeScorePeriod2)
  # clf_awayScorePeriod3.fit(x_train_awayScorePeriod3,y_train_awayScorePeriod3)
  # clf_homeScorePeriod3.fit(x_train_homeScorePeriod3,y_train_homeScorePeriod3)
  # clf_awayScorePeriod4.fit(x_train_awayScorePeriod4,y_train_awayScorePeriod4)
  # clf_homeScorePeriod4.fit(x_train_homeScorePeriod4,y_train_homeScorePeriod4)
  # clf_awayScorePeriod5.fit(x_train_awayScorePeriod5,y_train_awayScorePeriod5)
  # clf_homeScorePeriod5.fit(x_train_homeScorePeriod5,y_train_homeScorePeriod5)
  # clf_period1PuckLine.fit(x_train_period1PuckLine,y_train_period1PuckLine)
  # clf_period2PuckLine.fit(x_train_period2PuckLine,y_train_period2PuckLine)
  # clf_period3PuckLine.fit(x_train_period3PuckLine,y_train_period3PuckLine)

  predictions_winner = clf_winner.predict(x_test_winner)
  predictions_winnerB = clf_winnerB.predict(x_test_winnerB)
  predictions_homeScore = clf_homeScore.predict(x_test_homeScore)
  predictions_awayScore = clf_awayScore.predict(x_test_awayScore)
  predictions_totalGoals = clf_totalGoals.predict(x_test_totalGoals)
  predictions_goalDifferential = clf_goalDifferential.predict(x_test_goalDifferential)
  # predictions_finalPeriod = clf_finalPeriod.predict(x_test_finalPeriod)
  # predictions_pastRegulation = clf_pastRegulation.predict(x_test_pastRegulation)
  # predictions_awayShots = clf_awayShots.predict(x_test_awayShots)
  # predictions_homeShots = clf_homeShots.predict(x_test_homeShots)
  # predictions_awayShotsPeriod1 = clf_awayShotsPeriod1.predict(x_test_awayShotsPeriod1)
  # predictions_homeShotsPeriod1 = clf_homeShotsPeriod1.predict(x_test_homeShotsPeriod1)
  # predictions_awayShotsPeriod2 = clf_awayShotsPeriod2.predict(x_test_awayShotsPeriod2)
  # predictions_homeShotsPeriod2 = clf_homeShotsPeriod2.predict(x_test_homeShotsPeriod2)
  # predictions_awayShotsPeriod3 = clf_awayShotsPeriod3.predict(x_test_awayShotsPeriod3)
  # predictions_homeShotsPeriod3 = clf_homeShotsPeriod3.predict(x_test_homeShotsPeriod3)
  # predictions_awayShotsPeriod4 = clf_awayShotsPeriod4.predict(x_test_awayShotsPeriod4)
  # predictions_homeShotsPeriod4 = clf_homeShotsPeriod4.predict(x_test_homeShotsPeriod4)
  # predictions_awayShotsPeriod5 = clf_awayShotsPeriod5.predict(x_test_awayShotsPeriod5)
  # predictions_homeShotsPeriod5 = clf_homeShotsPeriod5.predict(x_test_homeShotsPeriod5)
  # predictions_awayScorePeriod1 = clf_awayScorePeriod1.predict(x_test_awayScorePeriod1)
  # predictions_homeScorePeriod1 = clf_homeScorePeriod1.predict(x_test_homeScorePeriod1)
  # predictions_awayScorePeriod2 = clf_awayScorePeriod2.predict(x_test_awayScorePeriod2)
  # predictions_homeScorePeriod2 = clf_homeScorePeriod2.predict(x_test_homeScorePeriod2)
  # predictions_awayScorePeriod3 = clf_awayScorePeriod3.predict(x_test_awayScorePeriod3)
  # predictions_homeScorePeriod3 = clf_homeScorePeriod3.predict(x_test_homeScorePeriod3)
  # predictions_awayScorePeriod4 = clf_awayScorePeriod4.predict(x_test_awayScorePeriod4)
  # predictions_homeScorePeriod4 = clf_homeScorePeriod4.predict(x_test_homeScorePeriod4)
  # predictions_awayScorePeriod5 = clf_awayScorePeriod5.predict(x_test_awayScorePeriod5)
  # predictions_homeScorePeriod5 = clf_homeScorePeriod5.predict(x_test_homeScorePeriod5)
  # predictions_period1PuckLine = clf_period1PuckLine.predict(x_test_period1PuckLine)
  # predictions_period2PuckLine = clf_period2PuckLine.predict(x_test_period2PuckLine)
  # predictions_period3PuckLine = clf_period3PuckLine.predict(x_test_period3PuckLine)

  winner_accuracy = accuracy_score(y_test_winner, predictions_winner)
  winnerB_accuracy = accuracy_score(y_test_winnerB, predictions_winnerB)
  homeScore_accuracy = accuracy_score(y_test_homeScore, predictions_homeScore)
  awayScore_accuracy = accuracy_score(y_test_awayScore, predictions_awayScore)
  totalGoals_accuracy = accuracy_score(y_test_totalGoals, predictions_totalGoals)
  goalDifferential_accuracy = accuracy_score(y_test_goalDifferential, predictions_goalDifferential)
  # finalPeriod_accuracy = accuracy_score(y_test_finalPeriod, predictions_finalPeriod)
  # pastRegulation_accuracy = accuracy_score(y_test_pastRegulation, predictions_pastRegulation)
  # awayShots_accuracy = accuracy_score(y_test_awayShots, predictions_awayShots)
  # homeShots_accuracy = accuracy_score(y_test_homeShots, predictions_homeShots)
  # awayShotsPeriod1_accuracy = accuracy_score(y_test_awayShotsPeriod1, predictions_awayShotsPeriod1)
  # homeShotsPeriod1_accuracy = accuracy_score(y_test_homeShotsPeriod1, predictions_homeShotsPeriod1)
  # awayShotsPeriod2_accuracy = accuracy_score(y_test_awayShotsPeriod2, predictions_awayShotsPeriod2)
  # homeShotsPeriod2_accuracy = accuracy_score(y_test_homeShotsPeriod2, predictions_homeShotsPeriod2)
  # awayShotsPeriod3_accuracy = accuracy_score(y_test_awayShotsPeriod3, predictions_awayShotsPeriod3)
  # homeShotsPeriod3_accuracy = accuracy_score(y_test_homeShotsPeriod3, predictions_homeShotsPeriod3)
  # awayShotsPeriod4_accuracy = accuracy_score(y_test_awayShotsPeriod4, predictions_awayShotsPeriod4)
  # homeShotsPeriod4_accuracy = accuracy_score(y_test_homeShotsPeriod4, predictions_homeShotsPeriod4)
  # awayShotsPeriod5_accuracy = accuracy_score(y_test_awayShotsPeriod5, predictions_awayShotsPeriod5)
  # homeShotsPeriod5_accuracy = accuracy_score(y_test_homeShotsPeriod5, predictions_homeShotsPeriod5)
  # awayScorePeriod1_accuracy = accuracy_score(y_test_awayScorePeriod1, predictions_awayScorePeriod1)
  # homeScorePeriod1_accuracy = accuracy_score(y_test_homeScorePeriod1, predictions_homeScorePeriod1)
  # awayScorePeriod2_accuracy = accuracy_score(y_test_awayScorePeriod2, predictions_awayScorePeriod2)
  # homeScorePeriod2_accuracy = accuracy_score(y_test_homeScorePeriod2, predictions_homeScorePeriod2)
  # awayScorePeriod3_accuracy = accuracy_score(y_test_awayScorePeriod3, predictions_awayScorePeriod3)
  # homeScorePeriod3_accuracy = accuracy_score(y_test_homeScorePeriod3, predictions_homeScorePeriod3)
  # awayScorePeriod4_accuracy = accuracy_score(y_test_awayScorePeriod4, predictions_awayScorePeriod4)
  # homeScorePeriod4_accuracy = accuracy_score(y_test_homeScorePeriod4, predictions_homeScorePeriod4)
  # awayScorePeriod5_accuracy = accuracy_score(y_test_awayScorePeriod5, predictions_awayScorePeriod5)
  # homeScorePeriod5_accuracy = accuracy_score(y_test_homeScorePeriod5, predictions_homeScorePeriod5)
  # period1PuckLine_accuracy = accuracy_score(y_test_period1PuckLine, predictions_period1PuckLine)
  # period2PuckLine_accuracy = accuracy_score(y_test_period2PuckLine, predictions_period2PuckLine)
  # period3PuckLine_accuracy = accuracy_score(y_test_period3PuckLine, predictions_period3PuckLine)
  print("Winner Accuracy:", winner_accuracy)
  print("Winner Binary Accuracy:", winnerB_accuracy)
  print("Home Score Accuracy:", homeScore_accuracy)
  print("Away Score Accuracy:", awayScore_accuracy)
  print("Total Goals Accuracy:", totalGoals_accuracy)
  print("Goal Differential Accuracy:", goalDifferential_accuracy)
  # print("finalPeriod Accuracy:", finalPeriod_accuracy)
  # print("pastRegulation Accuracy:", pastRegulation_accuracy)
  # print("awayShots Accuracy:", awayShots_accuracy)
  # print("homeShots Accuracy:", homeShots_accuracy)
  # print("awayShotsPeriod1 Accuracy:", awayShotsPeriod1_accuracy)
  # print("homeShotsPeriod1 Accuracy:", homeShotsPeriod1_accuracy)
  # print("awayShotsPeriod2 Accuracy:", awayShotsPeriod2_accuracy)
  # print("homeShotsPeriod2 Accuracy:", homeShotsPeriod2_accuracy)
  # print("awayShotsPeriod3 Accuracy:", awayShotsPeriod3_accuracy)
  # print("homeShotsPeriod3 Accuracy:", homeShotsPeriod3_accuracy)
  # print("awayShotsPeriod4 Accuracy:", awayShotsPeriod4_accuracy)
  # print("homeShotsPeriod4 Accuracy:", homeShotsPeriod4_accuracy)
  # print("awayShotsPeriod5 Accuracy:", awayShotsPeriod5_accuracy)
  # print("homeShotsPeriod5 Accuracy:", homeShotsPeriod5_accuracy)
  # print("awayScorePeriod1 Accuracy:", awayScorePeriod1_accuracy)
  # print("homeScorePeriod1 Accuracy:", homeScorePeriod1_accuracy)
  # print("awayScorePeriod2 Accuracy:", awayScorePeriod2_accuracy)
  # print("homeScorePeriod2 Accuracy:", homeScorePeriod2_accuracy)
  # print("awayScorePeriod3 Accuracy:", awayScorePeriod3_accuracy)
  # print("homeScorePeriod3 Accuracy:", homeScorePeriod3_accuracy)
  # print("awayScorePeriod4 Accuracy:", awayScorePeriod4_accuracy)
  # print("homeScorePeriod4 Accuracy:", homeScorePeriod4_accuracy)
  # print("awayScorePeriod5 Accuracy:", awayScorePeriod5_accuracy)
  # print("homeScorePeriod5 Accuracy:", homeScorePeriod5_accuracy)
  # print("period1PuckLine Accuracy:", period1PuckLine_accuracy)
  # print("period2PuckLine Accuracy:", period2PuckLine_accuracy)
  # print("period3PuckLine Accuracy:", period3PuckLine_accuracy)
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
    'model': 'Random Forest Classifier',
    'accuracies': {
      'winner': winner_accuracy,
      'winnerB': winnerB_accuracy,
      'homeScore': homeScore_accuracy,
      'awayScore': awayScore_accuracy,
      'totalGoals': totalGoals_accuracy,
      'goalDifferential': goalDifferential_accuracy,
      # 'finalPeriod': finalPeriod_accuracy,
      # 'pastRegulation': pastRegulation_accuracy,
      # 'awayShots': awayShots_accuracy,
      # 'homeShots': homeShots_accuracy,
      # 'awayShotsPeriod1': awayShotsPeriod1_accuracy,
      # 'homeShotsPeriod1': homeShotsPeriod1_accuracy,
      # 'awayShotsPeriod2': awayShotsPeriod2_accuracy,
      # 'homeShotsPeriod2': homeShotsPeriod2_accuracy,
      # 'awayShotsPeriod3': awayShotsPeriod3_accuracy,
      # 'homeShotsPeriod3': homeShotsPeriod3_accuracy,
      # 'awayShotsPeriod4': awayShotsPeriod4_accuracy,
      # 'homeShotsPeriod4': homeShotsPeriod4_accuracy,
      # 'awayShotsPeriod5': awayShotsPeriod5_accuracy,
      # 'homeShotsPeriod5': homeShotsPeriod5_accuracy,
      # 'awayScorePeriod1': awayScorePeriod1_accuracy,
      # 'homeScorePeriod1': homeScorePeriod1_accuracy,
      # 'awayScorePeriod2': awayScorePeriod2_accuracy,
      # 'homeScorePeriod2': homeScorePeriod2_accuracy,
      # 'awayScorePeriod3': awayScorePeriod3_accuracy,
      # 'homeScorePeriod3': homeScorePeriod3_accuracy,
      # 'awayScorePeriod4': awayScorePeriod4_accuracy,
      # 'homeScorePeriod4': homeScorePeriod4_accuracy,
      # 'awayScorePeriod5': awayScorePeriod5_accuracy,
      # 'homeScorePeriod5': homeScorePeriod5_accuracy,
      # 'period1PuckLine': period1PuckLine_accuracy,
      # 'period2PuckLine': period2PuckLine_accuracy,
      # 'period3PuckLine': period3PuckLine_accuracy,
    },
  })

  # dump(clf, f'models/nhl_ai_v{FILE_VERSION}.joblib')
  dump(clf_winner, f'models/nhl_ai_v{FILE_VERSION}_winner.joblib')
  dump(clf_winnerB, f'models/nhl_ai_v{FILE_VERSION}_winnerB.joblib')
  dump(clf_homeScore, f'models/nhl_ai_v{FILE_VERSION}_homeScore.joblib')
  dump(clf_awayScore, f'models/nhl_ai_v{FILE_VERSION}_awayScore.joblib')
  dump(clf_totalGoals, f'models/nhl_ai_v{FILE_VERSION}_totalGoals.joblib')
  dump(clf_goalDifferential, f'models/nhl_ai_v{FILE_VERSION}_goalDifferential.joblib')
  # dump(clf_finalPeriod, f'models/nhl_ai_v{FILE_VERSION}_finalPeriod.joblib')
  # dump(clf_pastRegulation, f'models/nhl_ai_v{FILE_VERSION}_pastRegulation.joblib')
  # dump(clf_awayShots, f'models/nhl_ai_v{FILE_VERSION}_awayShots.joblib')
  # dump(clf_homeShots, f'models/nhl_ai_v{FILE_VERSION}_homeShots.joblib')
  # dump(clf_awayShotsPeriod1, f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod1.joblib')
  # dump(clf_homeShotsPeriod1, f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod1.joblib')
  # dump(clf_awayShotsPeriod2, f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod2.joblib')
  # dump(clf_homeShotsPeriod2, f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod2.joblib')
  # dump(clf_awayShotsPeriod3, f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod3.joblib')
  # dump(clf_homeShotsPeriod3, f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod3.joblib')
  # dump(clf_awayShotsPeriod4, f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod4.joblib')
  # dump(clf_homeShotsPeriod4, f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod4.joblib')
  # dump(clf_awayShotsPeriod5, f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod5.joblib')
  # dump(clf_homeShotsPeriod5, f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod5.joblib')
  # dump(clf_awayScorePeriod1, f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod1.joblib')
  # dump(clf_homeScorePeriod1, f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod1.joblib')
  # dump(clf_awayScorePeriod2, f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod2.joblib')
  # dump(clf_homeScorePeriod2, f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod2.joblib')
  # dump(clf_awayScorePeriod3, f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod3.joblib')
  # dump(clf_homeScorePeriod3, f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod3.joblib')
  # dump(clf_awayScorePeriod4, f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod4.joblib')
  # dump(clf_homeScorePeriod4, f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod4.joblib')
  # dump(clf_awayScorePeriod5, f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod5.joblib')
  # dump(clf_homeScorePeriod5, f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod5.joblib')
  # dump(clf_period1PuckLine, f'models/nhl_ai_v{FILE_VERSION}_period1PuckLine.joblib')
  # dump(clf_period2PuckLine, f'models/nhl_ai_v{FILE_VERSION}_period2PuckLine.joblib')
  # dump(clf_period3PuckLine, f'models/nhl_ai_v{FILE_VERSION}_period3PuckLine.joblib')


dir_path = f'training_data/v{VERSION}'
if not os.path.exists(dir_path):
  os.makedirs(dir_path)

tdList = os.listdir(f'training_data/v{VERSION}')

USE_SEASONS = True
SKIP_SEASONS = [int(td.replace(f'training_data_v{FILE_VERSION}_','').replace('.joblib','')) for td in tdList] if len(tdList) > 0 and not f'training_data_v{FILE_VERSION}.joblib' in os.listdir('training_data') else []

LATEST_SEASON = 20232024
MAX_ID = 2023020514

if __name__ == '__main__':
  # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
  if RE_PULL:
    if USE_SEASONS:
      if TEST_DATA:
        seasons = [END_SEASON]
      else:
        seasons = list(db["dev_seasons"].find(
          {
            'seasonId': {
              '$gte': START_SEASON,
              '$lte': END_SEASON,
            }
          },
          {'_id':0,'seasonId': 1}
        ))
        seasons = [int(season['seasonId']) for season in seasons]
        print(seasons)
        if (len(SKIP_SEASONS) > 0):
          for season in SKIP_SEASONS:
            seasons.remove(season)
          print(seasons)
    else:
      ids = latestIDs()
      startID = ids['saved']['training']
      endID = MAX_ID
      games = list(db["dev_games"].find(
        {'id':{'$gte':startID,'$lt':endID+1}},
        # {'id':{'$lt':endID+1}},
        {'id': 1, '_id': 0}
      ))
    
    pool = Pool(processes=4)
    if USE_SEASONS:
      result = pool.map(season_training_data,seasons)
    else:
      result = pool.map(game_training_data,games)
    if len(SKIP_SEASONS) > 0 and not TEST_DATA:
      for skip_season in SKIP_SEASONS:
        season_data = load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{skip_season}.joblib')
        result.append(season_data)
    result = np.concatenate(result).tolist()
    pool.close()
    if TEST_DATA:
      dump(result,f'training_data/test_data_v{FILE_VERSION}.joblib')
    else:
      dump(result,f'training_data/training_data_v{FILE_VERSION}.joblib')
      f = open('training_data/training_data_text.txt', 'w')
      f.write(json.dumps(result[len(result)-200:len(result)]))
  else:
    training_data_path = f'training_data/training_data_v{FILE_VERSION}.joblib'
    print(training_data_path)
    result = load(training_data_path)
    f = open('training_data/training_data_text.txt', 'w')
    f.write(json.dumps(result[len(result)-200:len(result)]))
  print('Games Collected')
  if not TEST_DATA:
    train(db, result)