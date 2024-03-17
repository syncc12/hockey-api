import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

import requests
from pages.mlb.process import ai_teams
import warnings

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

def predict_team_day(date, day, gamePick, wModels):
  res = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&sportId=21&startDate={date}&endDate={date}").json()
  game_data = res['dates'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  return ai_teams(game_data['games'], wModels)

def predict_team_day_simple(date, day, gamePick, wModels):
  res = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&sportId=21&startDate={date}&endDate={date}").json()
  game_data = res['dates'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  return ai_teams(game_data['games'], wModels, simple=True)

def predict_team_day_receipt(date, day, gamePick, wModels):
  res = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&sportId=21&startDate={date}&endDate={date}").json()
  game_data = res['dates'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  return ai_teams(game_data['games'], wModels, receipt=True)