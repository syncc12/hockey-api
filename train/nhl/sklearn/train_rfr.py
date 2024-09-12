import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, Lars, LarsCV, LassoLars, LassoLarsCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
from sklearn.naive_bayes import GaussianNB
import time


OUTPUT = [
  'winnerB',
  'winner',
  'goalDifferential',
  'totalGoals',
]


def train(db,inData,inTestData):
  data = pd.DataFrame(inData)
  test_data = pd.DataFrame(inTestData)
  for op in OUTPUT:
    x = data [X_INPUTS]
    y = data [[op]].values.ravel()
    x_season_test = test_data [X_INPUTS]
    y_season_test = test_data [[op]].values.ravel()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)

    reg = GradientBoostingClassifier(random_state=RANDOM_STATE)
    reg.fit(x_train, y_train)


    # Calculate metrics
    # y_pred = reg.predict(x_test)
    # y_season_pred = reg.predict(x_season_test)
    # mse = mean_squared_error(y_test, y_pred)
    # mae = mean_absolute_error(y_test, y_pred)
    # r2 = r2_score(y_test, y_pred)
    # mse_season = mean_squared_error(y_season_test, y_season_pred)
    # mae_season = mean_absolute_error(y_season_test, y_season_pred)
    # r2_season = r2_score(y_season_test, y_season_pred)

    # print(f"{op} Mean Squared Error: {mse} | Season: {mse_season}")
    # print(f"{op} Mean Absolute Error: {mae} | Season: {mae_season}")
    # print(f"{op} R-squared: {r2} | Season: {r2_season}")

    accuracy = reg.score(x_test, y_test)
    season_accuracy = reg.score(x_season_test, y_season_test)
    print(f"{op} Accuracy: {accuracy} | Season Accuracy: {season_accuracy}")
    
    
    TrainingRecords = db['dev_training_records']

    timestamp = time.time()
    TrainingRecords.insert_one({
      'savedAt': timestamp,
      'version': VERSION,
      'inputs': X_INPUTS,
      'outputs': op,
      'randomState': RANDOM_STATE,
      'startingSeason': START_SEASON,
      'finalSeason': END_SEASON,
      'projectedLineup': False,
      # 'model': 'Random Forest Regressor',
      'model': 'Gradient Boosting Classifier',
      'accuracies': {
        # op: {
        #   'mse': mse,
        #   'mae': mae,
        #   'r2': r2,
        #   'mse_season': mse_season,
        #   'mae_season': mae_season,
        #   'r2_season': r2_season,
        # },
        op: accuracy
      },
    })
    
    dump(reg, f'models/nhl_ai_v{FILE_VERSION}_gbc_rfr_{op}.joblib')



client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
test_data = load(f'training_data/test_data_v{FILE_VERSION}.joblib')

train(db,training_data,test_data)


