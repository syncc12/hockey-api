import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')

from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process import nhl_ai
from process2 import nhl_data, nhl_test
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from util.training_data import save_training_data
from util.helpers import latestIDs, adjusted_winner
from inputs.inputs import master_inputs
from pages.nhl.nhl_helpers import ai, ai_return_dict
from constants.constants import VERSION

def debug():
  res = requests.get("https://api-web.nhle.com/v1/schedule/2023-11-22").json()
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]
  data = nhl_ai(game_data)
  return jsonify(data)


def test_model(db,startID,endID,show_data,**kwargs):
  Boxscores = db['dev_boxscores']

  if startID == -1 or endID == -1:
    md = metadata()
    if startID == -1:
      startID = md['saved']['training']+1
    if endID == -1:
      endID = min([md['saved']['boxscore'],md['saved']['game']])

    
  boxscore_list = list(Boxscores.find(
    {'id': {'$gte':startID,'$lt':endID+1}}
  ))

  test_results = {}

  all_winner_total = 0
  all_home_score_total = 0
  all_away_score_total = 0
  all_home_away_score_total = 0
  all_goal_total = 0
  all_goal_differential_total = 0
  all_list_total = 0
  all_ids = []
  for boxscore in boxscore_list:
    test_results[boxscore['gameDate']] = {
    'results': [],
    'winnerPercent': 0,
    'homeScorePercent': 0,
    'awayScorePercent': 0,
  }

  for boxscore in boxscore_list:
    all_ids.append(boxscore['id'])
    awayId = boxscore['awayTeam']['id']
    homeId = boxscore['homeTeam']['id']
    test_data = nhl_test(boxscore=boxscore)
    test_prediction_winner = kwargs['model_winner'].predict(test_data['data'])
    test_prediction_homeScore = kwargs['model_homeScore'].predict(test_data['data'])
    test_prediction_awayScore = kwargs['model_awayScore'].predict(test_data['data'])
    test_prediction_totalGoals = kwargs['model_totalGoals'].predict(test_data['data'])
    test_prediction_goalDifferential = kwargs['model_goalDifferential'].predict(test_data['data'])
    test_confidence_winner = kwargs['model_winner'].predict_proba(test_data['data'])
    test_confidence_homeScore = kwargs['model_homeScore'].predict_proba(test_data['data'])
    test_confidence_awayScore = kwargs['model_awayScore'].predict_proba(test_data['data'])
    test_confidence_totalGoals = kwargs['model_totalGoals'].predict_proba(test_data['data'])
    test_confidence_goalDifferential = kwargs['model_goalDifferential'].predict_proba(test_data['data'])
    predicted_winner = adjusted_winner(awayId, homeId, test_prediction_winner[0])
    predicted_homeScore = test_prediction_homeScore[0]
    predicted_awayScore = test_prediction_awayScore[0]
    predicted_totalGoals = test_prediction_totalGoals[0]
    predicted_goalDifferential = test_prediction_goalDifferential[0]
    test_winner = adjusted_winner(awayId, homeId, test_data['result'][0][2])
    test_homeScore = test_data['result'][0][0]
    test_awayScore = test_data['result'][0][1]
    test_totalGoals = test_data['result'][0][1]
    test_goalDifferential = test_data['result'][0][1]
    test_results[boxscore['gameDate']]['results'].append({
      'data': test_data['input_data'] if show_data != -1 else {},
      'winner': 1 if predicted_winner==test_winner else 0,
      'homeScore': 1 if predicted_homeScore==test_homeScore else 0,
      'awayScore': 1 if predicted_awayScore==test_awayScore else 0,
      'totalGoals': 1 if predicted_totalGoals==test_totalGoals else 0,
      'goalDifferential': 1 if predicted_goalDifferential==test_goalDifferential else 0,
      'winnerConfidence': int((np.max(test_confidence_winner, axis=1) * 100)[0]),
      'homeScoreConfidence': int((np.max(test_confidence_homeScore, axis=1) * 100)[0]),
      'awayScoreConfidence': int((np.max(test_confidence_awayScore, axis=1) * 100)[0]),
      'totalGoalsConfidence': int((np.max(test_confidence_totalGoals, axis=1) * 100)[0]),
      'goalDifferentialConfidence': int((np.max(test_confidence_goalDifferential, axis=1) * 100)[0]),
    })
    
  # for boxscore in boxscore_list:
    winner_total = 0
    home_score_total = 0
    away_score_total = 0
    home_away_score_total = 0
    goal_total = 0
    goal_differential_total = 0
    list_total = len(test_results[boxscore['gameDate']]['results'])
    all_list_total += list_total
    for r in test_results[boxscore['gameDate']]['results']:
      winner_total += r['winner']
      home_score_total += r['homeScore']
      away_score_total += r['awayScore']
      goal_total += r['totalGoals']
      goal_differential_total += r['goalDifferential']
      if home_score_total == 1 and away_score_total == 1:
        home_away_score_total += 1
    all_winner_total += winner_total
    all_home_score_total += home_score_total
    all_away_score_total += away_score_total
    all_home_away_score_total += home_away_score_total
    all_goal_total += goal_total
    all_goal_differential_total += goal_differential_total
    test_results[boxscore['gameDate']]['winnerPercent'] = (winner_total / list_total) * 100
    test_results[boxscore['gameDate']]['homeScorePercent'] = (home_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['awayScorePercent'] = (away_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['h2hScorePercent'] = (home_away_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['goalTotalPercent'] = (goal_total / list_total) * 100
    test_results[boxscore['gameDate']]['goalDifferentialPercent'] = (goal_differential_total / list_total) * 100
    test_results[boxscore['gameDate']]['totalGames'] = list_total
  
  test_results['allWinnerPercent'] = (all_winner_total / all_list_total) * 100
  test_results['allHomeScorePercent'] = (all_home_score_total / all_list_total) * 100
  test_results['allAwayScorePercent'] = (all_away_score_total / all_list_total) * 100
  test_results['allH2HScorePercent'] = (all_home_away_score_total / all_list_total) * 100
  test_results['allGoalTotalPercent'] = (all_goal_total / all_list_total) * 100
  test_results['allGoalDifferentialPercent'] = (all_goal_differential_total / all_list_total) * 100
  test_results['allIDs'] = all_ids
  test_results['allIDLength'] = len(all_ids)
  test_results['totalGames'] = all_list_total
  
  return test_results


def collect_boxscores(db,startID,endID,**kwargs):
  Boxscores = db['dev_boxscores']

  boxscores = []
  for id in range(startID, endID+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    boxscores.append(boxscore_data)
  Boxscores.insert_many(boxscores)
  return {'status':'done'}


def collect_training_data(db,startID,endID,id,**kwargs):
  training_data = []
  is_neutral_site = {}
  if startID != -1 and endID != -1 and id == -1:
    for id in range(startID, endID+1):
      loop_data = data_loop(id=id,is_neutral_site=is_neutral_site)
      is_neutral_site = loop_data['is_neutral_site']
      training_data.append(save_training_data(boxscores=loop_data['boxscore_data'],neutralSite=is_neutral_site[loop_data['boxscore_data']['id']]))
  else:
    if startID == -1 and endID == -1 and id != -1:
      loop_data = data_loop(id=id,is_neutral_site=is_neutral_site)
    elif startID != -1 and endID == -1 and id == -1:
      loop_data = data_loop(id=startID,is_neutral_site=is_neutral_site)
    elif startID == -1 and endID != -1 and id == -1:
      loop_data = data_loop(id=endID,is_neutral_site=is_neutral_site)
    training_data.append(save_training_data(boxscores=loop_data['boxscore_data'],neutralSite=is_neutral_site[loop_data['boxscore_data']['id']]))
  
  return training_data
  
def data_loop(id,is_neutral_site={}):
  boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()

  if not boxscore_data['id'] in is_neutral_site:
    game_week = requests.get(f"https://api-web.nhle.com/v1/schedule/{boxscore_data['gameDate']}").json()
    for games in game_week['gameWeek']:
      for game in games['games']:
        is_neutral_site[game['id']] = game['neutralSite']
  return {
    'boxscore_data':boxscore_data,
    'is_neutral_site':is_neutral_site,
  }


def predict(db,day,game,**kwargs):
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(game_data, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])


def predict_day(db,date,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)

  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(game, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])
    games.append(ai_data)
  return jsonify(games)


def predict_day_simple(db,date,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)

  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(game, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])
    if ai_data['message'] == 'using projected lineup':
      live_data = {}
    else:
      live_data = {
        'away': ai_data['live']['away'],
        'home': ai_data['live']['home'],
        'leader': ai_data['live']['leader'],
        'period': ai_data['live']['period'],
      }
    simple_data = {
      'awayTeam': f"{ai_data['awayTeam']} - {ai_data['awayScore']} - {ai_data['awayScoreConfidence']}%",
      'homeTeam': f"{ai_data['homeTeam']} - {ai_data['homeScore']} - {ai_data['homeScoreConfidence']}%",
      'live': live_data,
      'winningTeam': f"{ai_data['winningTeam']} - {ai_data['winnerConfidence']}%",
      'message': ai_data['message'],
      'offset': ai_data['offset'],
      'totalGoals': f"{ai_data['totalGoals']} - {ai_data['totalGoalsConfidence']}%",
      'goalDifferential': f"{ai_data['goalDifferential']} - {ai_data['goalDifferentialConfidence']}%",
    }
    games.append(simple_data)
  
  return jsonify(games)


def predict_week(db,**kwargs):
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

  games = {}
  for day in res['gameWeek']:
    games[day['dayAbbrev']] = []
    for game in day['games']:
      ai_data = ai(game, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])
      games[day['dayAbbrev']].append(ai_data)

  return jsonify(games)


def get_day_ids(db,date,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  ids = []
  for game in res['gameWeek'][0]['games']:
    ids.append(game['id'])
  
  return {'ids':ids}


def date_predict(db,date,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(game_data, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])


def now(db,**kwargs):
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()
  games = []
  for j, day in enumerate(res['gameWeek']):
    for i, game in enumerate(day['games']):
      games.append({
        'gameId': game['id'],
        'date': day['date'],
        'day': day['dayAbbrev'],
        'index': f'{j+1}-{i+1}',
        'query': f'?day={j+1}&game={i+1}',
        'homeTeam': game['homeTeam']['placeName']['default'],
        'awayTeam': game['awayTeam']['placeName']['default'],
      })
  return games


def game_date(db,date,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  games = []
  for j, day in enumerate(res['gameWeek']):
    for i, game in enumerate(day['games']):
      games.append({
        'gameId': game['id'],
        'date': day['date'],
        'day': day['dayAbbrev'],
        'index': f'{j+1}-{i+1}',
        'query': f'?day={j+1}&game={i+1}',
        'homeTeam': game['homeTeam']['placeName']['default'],
        'awayTeam': game['awayTeam']['placeName']['default'],
      })
  return games


def metadata(db,**kwargs):
  CURRENT_SEASON = db["dev_seasons"].find_one(sort=[("seasonId", -1)])['seasonId']
  used_training_data = load(f'training_data/v{VERSION}/training_data_v{VERSION}_{CURRENT_SEASON}.joblib')
  latest_ids = latestIDs(used_training_data)
  return latest_ids


def save_boxscores(db,date,**kwargs):
  Games = db['dev_games']
  latest_ids = latestIDs()
  
  Boxscores = db['dev_boxscores']
  boxscores = []
  for id in range(latest_ids['saved']['boxscore']+1,latest_ids['live']['boxscore']+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    Boxscores.insert_one(boxscore_data)
  
  schedule = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  for week in schedule['gameWeek']:
    for game in week['games']:
      if game['id'] <= latest_ids['live']['game']:
        try:
          game['date'] = week['date']
          Games.insert_one(game)
        except DuplicateKeyError:
          print('DUPLICATE', game['id'])
          pass
  
  print(latest_ids)

  return {'res':latest_ids}