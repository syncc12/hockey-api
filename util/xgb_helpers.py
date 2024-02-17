import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def aggregate_statistics():

  # Assuming df is your DataFrame and it contains the feature groups player1a...player20a and player1b...player20b
  features_group_a = [f'player{i}a' for i in range(1, 21)]
  features_group_b = [f'player{i}b' for i in range(1, 21)]

  # Compute aggregate statistics for each group
  for group, name in zip([features_group_a, features_group_b], ['a', 'b']):
    df[f'mean_{name}'] = df[group].mean(axis=1)
    df[f'std_{name}'] = df[group].std(axis=1)
    df[f'max_{name}'] = df[group].max(axis=1)
    df[f'min_{name}'] = df[group].min(axis=1)

  # Now df includes additional features that are permutation-invariant within each group

  # Prepare data for XGBoost
  X = df.drop(columns=['target'])  # Assuming 'target' is your target variable
  y = df['target']
