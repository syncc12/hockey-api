import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from joblib import load
import pandas as pd
from constants.inputConstants import X_INPUTS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, average_precision_score
from constants.constants import FILE_VERSION

model = load(f'models/nhl_ai_v{FILE_VERSION}_stacked_winnerB_3.joblib')
test_data = load(f'test_data/test_data_v{FILE_VERSION}.joblib')


data = pd.DataFrame(test_data)
x = data [X_INPUTS]
y = data [['winnerB']].values.ravel()

test_predictions = model.predict(x)
accuracy = accuracy_score(y, test_predictions)
balanced_accuracy = balanced_accuracy_score(y, test_predictions)
precision = precision_score(y, test_predictions, average='binary')
average_precision = average_precision_score(y, test_predictions)
recall = recall_score(y, test_predictions, average='binary')
f1 = f1_score(y, test_predictions, average='binary')

print("Accuracy:", accuracy)
print("Balanced Accuracy:", balanced_accuracy)
print("Precision:", precision)
print("Average Precision:", average_precision)
print("Recall:", recall)
print("F1:", f1)