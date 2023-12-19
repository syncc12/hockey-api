import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\constants')

from joblib import load, dump
from process2 import nhl_data
import pandas as pd
import requests


VERSION = 6

winner_model = load(f'models/nhl_ai_v{VERSION}_winner.joblib')

def predict():
  game = requests.get("https://api-web.nhle.com/v1/schedule/now").json()
  data = nhl_data(game=game['gameWeek'][0]['games'][0])
  predict_data = pd.DataFrame(data['data']['data'])
  winner_prediction = winner_model.predict(predict_data)
  print(winner_prediction)

predict()