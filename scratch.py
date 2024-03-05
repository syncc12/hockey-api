import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from joblib import dump, load
import warnings
from util.team_models import PREDICT, PREDICT_H2H, PREDICT_SCORE_H2H, PREDICT_CALIBRATED_H2H, PREDICT_CALIBRATED_SCORE_H2H, W_MODELS, L_MODELS, W_MODELS_C, L_MODELS_C
from util.team_helpers import XGBWrapper, XGBWrapperInverse
# from util.helpers import team_lookup
from sklearn.metrics import accuracy_score
from training_input import test_input
from constants.inputConstants import X_INPUTS_T, Y_OUTPUTS
import pandas as pd

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
db = client["hockey"]

# # Suppress specific UserWarning from sklearn
# warnings.filterwarnings("ignore", message="X does not have valid feature names")
# warnings.filterwarnings("ignore", message="X has feature names")


# data_rename_1 = {
#   'homeTeam': 'team',
#   'awayTeam': 'opponent',
#   'homeScore': 'score',
#   'awayScore': 'opponentScore',
#   'homeHeadCoach': 'headCoach',
#   'awayHeadCoach': 'opponentHeadCoach',
#   'homeForwardAverage': 'forwardAverage',
#   'homeDefenseAverage': 'defenseAverage',
#   'homeGoalieAverage': 'goalieAverage',
#   'awayForwardAverage': 'opponentForwardAverage',
#   'awayDefenseAverage': 'opponentDefenseAverage',
#   'awayGoalieAverage': 'opponentGoalieAverage',
#   'homeForwardAverageAge': 'forwardAverageAge',
#   'homeDefenseAverageAge': 'defenseAverageAge',
#   'homeGoalieAverageAge': 'goalieAverageAge',
#   'awayForwardAverageAge': 'opponentForwardAverageAge',
#   'awayDefenseAverageAge': 'opponentDefenseAverageAge',
#   'awayGoalieAverageAge': 'opponentGoalieAverageAge',
#   'winner': 'win',
#   'winnerB': 'winB',
# }
# data_rename_2 = {
#   'homeTeam': 'opponent',
#   'awayTeam': 'team',
#   'homeScore': 'opponentScore',
#   'awayScore': 'score',
#   'homeHeadCoach': 'opponentHeadCoach',
#   'awayHeadCoach': 'headCoach',
#   'homeForwardAverage': 'opponentForwardAverage',
#   'homeDefenseAverage': 'opponentDefenseAverage',
#   'homeGoalieAverage': 'opponentGoalieAverage',
#   'awayForwardAverage': 'forwardAverage',
#   'awayDefenseAverage': 'defenseAverage',
#   'awayGoalieAverage': 'goalieAverage',
#   'homeForwardAverageAge': 'opponentForwardAverageAge',
#   'homeDefenseAverageAge': 'opponentDefenseAverageAge',
#   'homeGoalieAverageAge': 'opponentGoalieAverageAge',
#   'awayForwardAverageAge': 'forwardAverageAge',
#   'awayDefenseAverageAge': 'defenseAverageAge',
#   'awayGoalieAverageAge': 'goalieAverageAge',
#   'winner': 'win',
#   'winnerB': 'winB',
# }
# OUTPUT = 'winnerB'
# TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
# test_df = pd.DataFrame(TEST_DATA)
# test_data1 = pd.DataFrame(TEST_DATA)
# test_data2 = pd.DataFrame(TEST_DATA)
# test_data1.rename(columns=data_rename_1, inplace=True)
# test_data1['winB'] = 1 - test_data1['winB']
# test_data1['lossB'] = 1 - test_data1['winB']
# test_data2.rename(columns=data_rename_2, inplace=True)
# test_data2['lossB'] = 1 - test_data2['winB']
# test_data = pd.concat([test_data1, test_data2], axis=0)
# test_data.reset_index(drop=True, inplace=True)
# # test_teams = test_data.groupby('team')
# # for team, team_data in test_teams:
# #   data = team_data [X_INPUTS_T]
# #   y_test = team_data [[OUTPUT]].values.ravel()
# #   svp,sv,w_prediction,w_probability,l_prediction,l_probability = PREDICT(data, team, W_MODELS, L_MODELS)
# #   # print('y_test',y_test)
# #   # print('svp',svp)
# #   svp_accuracy = accuracy_score(y_test, svp)
# #   w_accuracy = accuracy_score(y_test, w_prediction)
# #   l_accuracy = accuracy_score(y_test, l_prediction)
# #   print(f'{team} - {(svp_accuracy*100):.2f}% - {(w_accuracy*100):.2f}% - {(l_accuracy*100):.2f}% - {len(y_test)} games')





# # prediction,confidences, *other = PREDICT_H2H(TEST_DATA, W_MODELS, L_MODELS, test=True)
# score_prediction,score_confidences, *other = PREDICT_SCORE_H2H(TEST_DATA, W_MODELS, L_MODELS, test=True)
# calibrated_prediction,calibrated_confidences, *other = PREDICT_CALIBRATED_H2H(TEST_DATA, W_MODELS_C, L_MODELS_C, test=True)
# calibrated_score_prediction,calibrated_score_confidences, *other = PREDICT_CALIBRATED_SCORE_H2H(TEST_DATA, W_MODELS_C, L_MODELS_C, test=True)
# y_test = test_df [['winnerB']].values.ravel()

# # accuracy = accuracy_score(y_test, prediction)
# score_accuracy = accuracy_score(y_test, score_prediction)
# calibrated_accuracy = accuracy_score(y_test, calibrated_prediction)
# calibrated_score_accuracy = accuracy_score(y_test, calibrated_score_prediction)
# # print(f'Accuracy: {(accuracy*100):.2f}% - {len(y_test)} games')
# print(f'Score Accuracy: {(score_accuracy*100):.2f}% - {len(y_test)} games')
# print(f'Calibrated Accuracy: {(calibrated_accuracy*100):.2f}% - {len(y_test)} games')
# print(f'Calibrated Score Accuracy: {(calibrated_score_accuracy*100):.2f}% - {len(y_test)} games')

# # accuracies = {
# #   'NJD': 0.6034,
# #   'NYI': 0.6379,
# #   'NYR': 0.6610,
# #   'PHI': 0.6333,
# #   'PIT': 0.6545,
# #   'BOS': 0.6333,
# #   'BUF': 0.6102,
# #   'MTL': 0.6271,
# #   'OTT': 0.6140,
# #   'TOR': 0.6724,
# #   'CAR': 0.6271,
# #   'FLA': 0.7586,
# #   'TBL': 0.6557,
# #   'WSH': 0.6207,
# #   'CHI': 0.7627,
# #   'DET': 0.6441,
# #   'NSH': 0.6167,
# #   'STL': 0.6552,
# #   'CGY': 0.6034,
# #   'COL': 0.6610,
# #   'EDM': 0.7321,
# #   'VAN': 0.7167,
# #   'ANA': 0.7069,
# #   'DAL': 0.7500,
# #   'LAK': 0.6316,
# #   'SJS': 0.7500,
# #   'CBJ': 0.7018,
# #   'MIN': 0.6441,
# #   'WPG': 0.7018,
# #   'ARI': 0.6897,
# #   'VGK': 0.6207,
# #   'SEA': 0.6379,
# # }

# # adjusted_accuracies = []

# # for team, accuracy in accuracies.items():
# #   if accuracy < 0.5:
# #     adjusted_accuracies.append(1 - accuracy)
# #   else:
# #     adjusted_accuracies.append(accuracy)
# # average_accuracy = (sum(adjusted_accuracies)/len(adjusted_accuracies))*100
# # print(f'Adjusted Accuracy: {average_accuracy:.2f}% - {len(adjusted_accuracies)} teams')

Training_Records = db['dev_training_records']
training_records = list(Training_Records.find(
  {'calibrationMethod': 'isotonic'},
  {'_id': 0, 'team': 1, 'params': 1, 'accuracies': 1}
))

print(training_records)