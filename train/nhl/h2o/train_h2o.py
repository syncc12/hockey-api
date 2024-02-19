import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')

from constants.constants import VERSION, H2O_FILE_VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
import h2o
from h2o.automl import H2OAutoML
import pandas as pd
from joblib import load
# from sklearn.preprocessing import LabelEncoder

OUTPUT = 'goalDifferential'

def training_data_conversion():
  training_data = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
  data = pd.DataFrame(training_data)
  data.to_csv(f'training_data/training_data_v{H2O_FILE_VERSION}.csv', index=False)


def h2o_train():
  h2o.init()

  data = h2o.import_file(f'training_data/training_data_v{H2O_FILE_VERSION}.csv')
  data[OUTPUT] = data[OUTPUT].asfactor()
  train, valid, test = data.split_frame(ratios=[.7, .15], seed=12)
  x = X_INPUTS
  y = OUTPUT

  # aml = H2OAutoML(max_models=200, seed=12, max_runtime_secs=3600, stopping_metric='AUC')
  aml = H2OAutoML(max_models=50, nfolds=5, seed=12, max_runtime_secs_per_model=3600)
  aml.train(x=x, y=y, training_frame=train, validation_frame=valid, leaderboard_frame=test)

  lb = aml.leaderboard
  print(lb.head(rows=lb.nrows))  # Print all rows instead of default (10 rows)

  preds = aml.leader.predict(test)
  print(preds)

  # print('Model Accuracies:')
  # accuracy_column = lb['accuracy'] if 'accuracy' in lb.columns else None
  # if accuracy_column is not None:
  #     model_accuracies = accuracy_column.as_data_frame()
  #     print(model_accuracies)

  h2o.save_model(model=aml.leader, path=f'models/nhl_ai_v{H2O_FILE_VERSION}_h2o_{OUTPUT}', force=True)

  h2o.cluster().shutdown()

h2o_train()
# training_data_conversion()