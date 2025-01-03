import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from joblib import load
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split

def hyperparameter_tuning():
  imputer = SimpleImputer(strategy='constant', fill_value=-1)
  training_data = load('training_data/training_data_v6.joblib')
  data = pd.DataFrame(training_data)
  x = data [[
    'id','season','gameType','venue','neutralSite','homeTeam','awayTeam','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Age',
    'awayForward2','awayForward2Age','awayForward3','awayForward3Age','awayForward4','awayForward4Age','awayForward5','awayForward5Age','awayForward6','awayForward6Age','awayForward7',
    'awayForward7Age','awayForward8','awayForward8Age','awayForward9','awayForward9Age','awayForward10','awayForward10Age','awayForward11','awayForward11Age','awayForward12','awayForward12Age',
    'awayForward13','awayForward13Age','awayDefenseman1','awayDefenseman1Age','awayDefenseman2','awayDefenseman2Age','awayDefenseman3','awayDefenseman3Age','awayDefenseman4','awayDefenseman4Age',
    'awayDefenseman5','awayDefenseman5Age','awayDefenseman6','awayDefenseman6Age','awayDefenseman7','awayDefenseman7Age','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge',
    'awayStartingGoalieHeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','homeForward1','homeForward1Age','homeForward2','homeForward2Age',
    'homeForward3','homeForward3Age','homeForward4','homeForward4Age','homeForward5','homeForward5Age','homeForward6','homeForward6Age','homeForward7','homeForward7Age','homeForward8',
    'homeForward8Age','homeForward9','homeForward9Age','homeForward10','homeForward10Age','homeForward11','homeForward11Age','homeForward12','homeForward12Age','homeForward13','homeForward13Age',
    'homeDefenseman1','homeDefenseman1Age','homeDefenseman2','homeDefenseman2Age','homeDefenseman3','homeDefenseman3Age','homeDefenseman4','homeDefenseman4Age','homeDefenseman5',
    'homeDefenseman5Age','homeDefenseman6','homeDefenseman6Age','homeDefenseman7','homeDefenseman7Age','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge',
    'homeStartingGoalieHeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight'
  ]].values
  # y = data [['homeScore','awayScore','winner','totalGoals','goalDifferential']].values
  y = data [['winner']].values.ravel()

  imputer.fit(x)
  x = imputer.transform(x)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
  # Define the parameter grid
  print(y_train.shape)
  param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
  }

  # Instantiate RandomForestClassifier
  clf = RandomForestClassifier(random_state=12)

  # Instantiate GridSearchCV
  grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
  
  # Fit GridSearchCV
  grid_search.fit(x_train, y_train)

  # Best parameters and best score
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_

  print('best_params',best_params)
  print('best_score',best_score)


hyperparameter_tuning()