import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, LassoCV, Lars, LarsCV, LassoLars, LassoLarsCV
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time


TRIAL_OUTPUT = 'winnerB'


def train(inData,inTestData):
  data = pd.DataFrame(inData)
  test_data = pd.DataFrame(inTestData)
  x = data [X_INPUTS]
  y = data [[TRIAL_OUTPUT]].values.ravel()
  x_season_test = test_data [X_INPUTS]
  y_season_test = test_data [[TRIAL_OUTPUT]].values.ravel()

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)
  base_learners = [
    # ('svc', SVC(probability=True)),
    ('lso', Lasso()),
    ('lsocv', LassoCV()),
    ('lrs', Lars()),
    ('lrscv', LarsCV()),
    ('ll', LassoLars()),
    ('llcv', LassoLarsCV()),
    ('dtc', DecisionTreeClassifier()),
    ('dtr', DecisionTreeRegressor()),
    ('rfc', RandomForestClassifier()),
    ('rfr', RandomForestRegressor()),
    ('knc', KNeighborsClassifier()),
    ('knr', KNeighborsRegressor()),
    # ('hgbc', HistGradientBoostingClassifier()),
    # ('hgbr', HistGradientBoostingRegressor()),
    ('gbc', GradientBoostingClassifier()),
    ('gbr', GradientBoostingRegressor()),
  ]
  final_estimator = LogisticRegression()
  for learner in base_learners:
    clf = StackingClassifier(estimators=[learner], final_estimator=final_estimator)
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    season_accuracy = clf.score(x_season_test, y_season_test)
    print(f"{TRIAL_OUTPUT} {learner[0]} Accuracy: {accuracy} Season Accuracy: {season_accuracy}")
  

  
  # dump(goalDifferential_stacked_clf, f'models/nhl_ai_v{FILE_VERSION}_stacked_{TRIAL_OUTPUT}.joblib')



# client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
# db = client["hockey"]
training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
test_data = load(f'training_data/test_data_v{FILE_VERSION}.joblib')

train(training_data,test_data)


# goalDifferential svc+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential dtc+LogisticRegression Accuracy:   0.4587719298245614
# goalDifferential dtr+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential rfc+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential rfr+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential knc+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential knr+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential gbc+LogisticRegression Accuracy:   0.4587719298245614
# goalDifferential gbr+LogisticRegression Accuracy:   0.4587719298245614
# goalDifferential lso+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential lsocv+LogisticRegression Accuracy: 0.4589181286549708
# goalDifferential lrs+LogisticRegression Accuracy:   0.4590643274853801
# goalDifferential lrscv+LogisticRegression Accuracy: 0.4589181286549708
# goalDifferential ll+LogisticRegression Accuracy:    0.4590643274853801
# goalDifferential llcv+LogisticRegression Accuracy:  0.4589181286549708
# goalDifferential dtc+LogisticRegression Accuracy:   0.4589181286549708
# goalDifferential dtr+LogisticRegression Accuracy:   0.4589181286549708

