import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process import nhl_ai, nhl_test
from process2 import nhl_data
from pymongo import MongoClient
import os
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from util.training_data import save_training_data
from util.helpers import latestIDs
import boto3
import io
from inputs.inputs import master_inputs

LATEST_DATE_TRAINED = '2023-11-11'
LATEST_DATE_COLLECTED = '2023-11-17'
LATEST_ID_COLLECTED = '2023020253'

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']

model = load('models/nhl_ai_v4.joblib')

def ai_return_dict(data, prediction):
  homeId = data['data']['home_team']['id']
  awayId = data['data']['away_team']['id']
  if len(data['data']['data'][0]) == 0 or len(prediction[0]) == 0:
    winnerId = -1
    homeScore = -1
    awayScore = -1
    state = 'OFF'
    homeTeam = data['data']['home_team']['city']
    awayTeam = data['data']['away_team']['city']
    live_away = -1
    live_home = -1
    live_period = -1
    live_clock = -1
    live_stopped = -1
    live_intermission = -1
    live_leaderId = -1
    live_leader = -1
    winningTeam = -1
    offset = -1
  else:
    winnerId = int(prediction[0][2])
    homeScore = int(prediction[0][0])
    awayScore = int(prediction[0][1])
    state = data['data']['state']
    homeTeam = f"{data['data']['home_team']['city']} {data['data']['home_team']['name']}"
    awayTeam = f"{data['data']['away_team']['city']} {data['data']['away_team']['name']}"
    live_away = data['data']['live']['away_score']
    live_home = data['data']['live']['home_score']
    live_period = data['data']['live']['period']
    live_clock = data['data']['live']['clock']
    live_stopped = data['data']['live']['stopped']
    live_intermission = data['data']['live']['intermission']
    if live_away > live_home:
      live_leaderId = awayId
      live_leader = awayTeam
    elif live_home > live_away:
      live_leaderId = homeId
      live_leader = homeTeam
    else:
      live_leaderId = -1
      live_leader = 'tied'
    if abs(winnerId - homeId) < abs(winnerId - awayId):
      winningTeam = homeTeam
      offset = abs(winnerId - homeId)
    elif abs(winnerId - homeId) > abs(winnerId - awayId):
      winningTeam = awayTeam
      offset = abs(winnerId - awayId)
    else:
      winningTeam = 'Inconclusive'
      offset = -1

  if data['message'] == 'using projected lineup':
    live_data = {}
  else:
    live_data = {
      'away': live_away,
      'home': live_home,
      'period': live_period,
      'clock': live_clock,
      'stopped': live_stopped,
      'intermission': live_intermission,
      'leader': live_leader,
      'leaderId': live_leaderId,
    }

  return {
    'gameId': data['data']['game_id'],
    'date': data['data']['date'],
    'state': state,
    'homeId': homeId,
    'awayId': awayId,
    'homeTeam': homeTeam,
    'awayTeam': awayTeam,
    'winnerId': winnerId,
    'winningTeam': winningTeam,
    'homeScore': homeScore,
    'awayScore': awayScore,
    'offset': offset,
    'live': live_data,
    # 'data': data['data']['data'],
    'message': data['message'],
  }

def ai(game_data):
  # data = nhl_ai(game_data)
  data = nhl_data(game_data)

  if len(data['data']['data'][0]) == 0:
    return ai_return_dict(data,[[]])

  prediction = model.predict(data['data']['data'])

  return ai_return_dict(data,prediction)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
  return jsonify({'data': 'root'})

@app.route('/debug', methods=['GET'])
def debug():
  res = requests.get("https://api-web.nhle.com/v1/schedule/2023-11-22").json()
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]
  data = nhl_ai(game_data)
  return jsonify(data)

@app.route('/test', methods=['GET'])
def test_model():
  Boxscores = db['dev_boxscores']
  startID = request.args.get('start', default=1, type=int)
  endID = request.args.get('end', default=1, type=int)
  # startID = int(request.json['startId'])
  # endID = int(request.json['endId'])
  boxscore_list = list(Boxscores.find(
    {'id': {'$gte':startID,'$lt':endID+1}}
  ))

  test_results = {}

  all_winner_total = 0
  all_home_score_total = 0
  all_away_score_total = 0
  all_home_away_score_total = 0
  all_score_total = 0
  all_list_total = 0
  for boxscore in boxscore_list:
    test_results[boxscore['gameDate']] = {
    'results': [],
    'winnerPercent': 0,
    'homeScorePercent': 0,
    'awayScorePercent': 0,
  }

  for boxscore in boxscore_list:
    test_data = nhl_test(boxscore=boxscore)
    test_prediction = model.predict(test_data['data'])
    test_confidence = model.predict_proba(test_data['data'])
    predicted_winner = test_prediction[0][2]
    predicted_homeScore = test_prediction[0][0]
    predicted_awayScore = test_prediction[0][1]
    test_winner = test_data['result'][0][2]
    test_homeScore = test_data['result'][0][0]
    test_awayScore = test_data['result'][0][1]
    test_results[boxscore['gameDate']]['results'].append({
      'winner': 1 if predicted_winner==test_winner else 0,
      'homeScore': 1 if predicted_homeScore==test_homeScore else 0,
      'awayScore': 1 if predicted_awayScore==test_awayScore else 0,
      'totalScore': 1 if (predicted_homeScore+predicted_awayScore)==(test_homeScore+test_awayScore)  else 0,
      'winnerConfidence': int((np.max(test_confidence[2], axis=1) * 100)[0]),
      'homeScoreConfidence': int((np.max(test_confidence[0], axis=1) * 100)[0]),
      'awayScoreConfidence': int((np.max(test_confidence[1], axis=1) * 100)[0]),
    })
    # print(int((np.max(test_confidence[2], axis=1) * 100)[0]))
    # print(predicted_winner==test_winner,predicted_homeScore==test_homeScore,predicted_awayScore==test_awayScore)
  for boxscore in boxscore_list:
    winner_total = 0
    home_score_total = 0
    away_score_total = 0
    home_away_score_total = 0
    score_total = 0
    list_total = len(test_results[boxscore['gameDate']]['results'])
    all_list_total += list_total
    for r in test_results[boxscore['gameDate']]['results']:
      winner_total += r['winner']
      home_score_total += r['homeScore']
      away_score_total += r['awayScore']
      score_total += r['totalScore']
      if home_score_total == 1 and away_score_total == 1:
        home_away_score_total += 1
    all_winner_total += winner_total
    all_home_score_total += home_score_total
    all_away_score_total += away_score_total
    all_home_away_score_total += home_away_score_total
    all_score_total += score_total
    test_results[boxscore['gameDate']]['winnerPercent'] = (winner_total / list_total) * 100
    test_results[boxscore['gameDate']]['homeScorePercent'] = (home_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['awayScorePercent'] = (away_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['h2hScorePercent'] = (home_away_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['totalScorePercent'] = (score_total / list_total) * 100
  
  test_results['allWinnerPercent'] = (all_winner_total / all_list_total) * 100
  test_results['allHomeScorePercent'] = (all_home_score_total / all_list_total) * 100
  test_results['allAwayScorePercent'] = (all_away_score_total / all_list_total) * 100
  test_results['allH2HScorePercent'] = (all_home_away_score_total / all_list_total) * 100
  test_results['allTotalScorePercent'] = (all_score_total / all_list_total) * 100
  
  return test_results

@app.route('/collect/boxscores', methods=['POST'])
def collect_boxscores():
  Boxscores = db['dev_boxscores']

  startID = int(request.json['startId'])
  endID = int(request.json['endId'])
  boxscores = []
  for id in range(startID, endID+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    boxscores.append(boxscore_data)
  Boxscores.insert_many(boxscores)
  return {'status':'done'}

# @app.route('/collect/training-data-v3', methods=['GET'])
# def collect_training_data_v3():
#   seasons = list(db["dev_seasons"].find(
#     {},
#     {'_id':0,'seasonId': 1}
#   ))
#   seasons = [season['seasonId'] for season in seasons]
#   Trainings_v3 = db['dev_trainings_v3']
#   for season in seasons:
#     print(f'Fired {season}')
#     try:
#       training_data = load(f'training_data/training_data_v3_{season}.joblib')
#       Trainings_v3.insert_many(training_data)
#     except Exception as error:
#       print(f'{season} - {error}')
#     print(f'DONE {season}')
#   return {'status':'done'}

@app.route('/collect/training-data', methods=['GET'])
def collect_training_data():
  startID = request.args.get('start', default=-1, type=int)
  endID = request.args.get('end', default=-1, type=int)
  id = request.args.get('id', default=-1, type=int)

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

@app.route('/nhl', methods=['GET'])
def predict():
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(game_data)

@app.route('/nhl/day', methods=['GET'])
def predict_day():
  date = request.args.get('date', default='now', type=str)
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)

  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(game)
    games.append(ai_data)
  print(games)
  return jsonify(games)

@app.route('/nhl/day/simple', methods=['GET'])
def predict_day_simple():
  date = request.args.get('date', default='now', type=str)
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)

  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(game)
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
      'awayTeam': f"{ai_data['awayTeam']} - {ai_data['awayScore']}",
      'homeTeam': F"{ai_data['homeTeam']} - {ai_data['homeScore']}",
      'live': live_data,
      'winningTeam': ai_data['winningTeam'],
      'message': ai_data['message'],
      'offset': ai_data['offset'],
    }
    games.append(simple_data)
  
  return jsonify(games)

@app.route('/nhl/week', methods=['GET'])
def predict_week():
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

  games = {}
  for day in res['gameWeek']:
    games[day['dayAbbrev']] = []
    for game in day['games']:
      ai_data = ai(game)
      games[day['dayAbbrev']].append(ai_data)

  return jsonify(games)

@app.route('/nhl/<date>', methods=['GET'])
def date_predict(date):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(game_data)

@app.route('/now', methods=['GET'])
def now():
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

@app.route('/date/<date>', methods=['GET'])
def game_date(date):
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

@app.route('/metadata', methods=['GET'])
def metadata():
  used_training_data = load('training_data/training_data.joblib')
  latest_ids = latestIDs(used_training_data)
  return latest_ids

@app.route('/db/update', methods=['GET'])
def save_boxscores():
  Boxscores = db['dev_boxscores']
  Games = db['dev_games']
  latest_ids = latestIDs()
  boxscores = []
  for id in range(latest_ids['saved']['boxscore']+1,latest_ids['live']['boxscore']+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    Boxscores.insert_one(boxscore_data)
  
  schedule_now = requests.get(f"https://api-web.nhle.com/v1/schedule/now").json()
  for week in schedule_now['gameWeek']:
    for game in week['games']:
      if game['id'] <= latest_ids['live']['game']:
        Games.insert_one(game)
  
  print(latest_ids)

  return {'res':latest_ids}

if __name__ == '__main__':
  app.run()
