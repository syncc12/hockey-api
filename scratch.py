import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from joblib import dump, load
import warnings
from util.team_models import PREDICT, PREDICT_H2H, W_MODELS, L_MODELS
# from util.helpers import team_lookup
from sklearn.metrics import accuracy_score
from training_input import test_input
from constants.inputConstants import X_INPUTS_T, Y_OUTPUTS
import pandas as pd

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")


data_rename_1 = {
  'homeTeam': 'team',
  'awayTeam': 'opponent',
  'homeScore': 'score',
  'awayScore': 'opponentScore',
  'homeHeadCoach': 'headCoach',
  'awayHeadCoach': 'opponentHeadCoach',
  'homeForwardAverage': 'forwardAverage',
  'homeDefenseAverage': 'defenseAverage',
  'homeGoalieAverage': 'goalieAverage',
  'awayForwardAverage': 'opponentForwardAverage',
  'awayDefenseAverage': 'opponentDefenseAverage',
  'awayGoalieAverage': 'opponentGoalieAverage',
  'homeForwardAverageAge': 'forwardAverageAge',
  'homeDefenseAverageAge': 'defenseAverageAge',
  'homeGoalieAverageAge': 'goalieAverageAge',
  'awayForwardAverageAge': 'opponentForwardAverageAge',
  'awayDefenseAverageAge': 'opponentDefenseAverageAge',
  'awayGoalieAverageAge': 'opponentGoalieAverageAge',
  'winner': 'win',
  'winnerB': 'winB',
}
data_rename_2 = {
  'homeTeam': 'opponent',
  'awayTeam': 'team',
  'homeScore': 'opponentScore',
  'awayScore': 'score',
  'homeHeadCoach': 'opponentHeadCoach',
  'awayHeadCoach': 'headCoach',
  'homeForwardAverage': 'opponentForwardAverage',
  'homeDefenseAverage': 'opponentDefenseAverage',
  'homeGoalieAverage': 'opponentGoalieAverage',
  'awayForwardAverage': 'forwardAverage',
  'awayDefenseAverage': 'defenseAverage',
  'awayGoalieAverage': 'goalieAverage',
  'homeForwardAverageAge': 'opponentForwardAverageAge',
  'homeDefenseAverageAge': 'opponentDefenseAverageAge',
  'homeGoalieAverageAge': 'opponentGoalieAverageAge',
  'awayForwardAverageAge': 'forwardAverageAge',
  'awayDefenseAverageAge': 'defenseAverageAge',
  'awayGoalieAverageAge': 'goalieAverageAge',
  'winner': 'win',
  'winnerB': 'winB',
}
OUTPUT = 'winB'
TEST_DATA = test_input(X_INPUTS_T,[OUTPUT],no_format=True)
test_df = pd.DataFrame(TEST_DATA)
test_data1 = pd.DataFrame(TEST_DATA)
test_data2 = pd.DataFrame(TEST_DATA)
test_data1.rename(columns=data_rename_1, inplace=True)
test_data1['winB'] = 1 - test_data1['winB']
test_data1['lossB'] = 1 - test_data1['winB']
test_data2.rename(columns=data_rename_2, inplace=True)
test_data2['lossB'] = 1 - test_data2['winB']
test_data = pd.concat([test_data1, test_data2], axis=0)
test_data.reset_index(drop=True, inplace=True)
# test_teams = test_data.groupby('team')
# for team, team_data in test_teams:
#   data = team_data [X_INPUTS_T]
#   y_test = team_data [[OUTPUT]].values.ravel()
#   svp,sv,w_prediction,w_probability,l_prediction,l_probability = PREDICT(data, team, W_MODELS, L_MODELS)
#   # print('y_test',y_test)
#   # print('svp',svp)
#   svp_accuracy = accuracy_score(y_test, svp)
#   w_accuracy = accuracy_score(y_test, w_prediction)
#   l_accuracy = accuracy_score(y_test, l_prediction)
#   print(f'{team} - {(svp_accuracy*100):.2f}% - {(w_accuracy*100):.2f}% - {(l_accuracy*100):.2f}% - {len(y_test)} games')

prediction,away_prediction,home_prediction,away_probability,home_probability,w_prediction_away,l_prediction_away,w_prediction_home,l_prediction_home,w_probability_away,l_probability_away,w_probability_home,l_probability_home = PREDICT_H2H(TEST_DATA, W_MODELS, L_MODELS)
y_test = test_df [['winnerB']].values.ravel()

accuracy = accuracy_score(y_test, prediction)
print(f'Accuracy: {(accuracy*100):.2f}% - {len(y_test)} games')