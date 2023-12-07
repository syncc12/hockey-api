from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process import nhl_ai, nhl_test
from pymongo import MongoClient
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import boto3
import io

LATEST_DATE_TRAINED = '2023-11-11'
LATEST_DATE_COLLECTED = '2023-11-17'
LATEST_ID_COLLECTED = '2023020253'

model = load('nhl_ai.joblib')

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
  data = nhl_ai(game_data)

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
  db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  client = MongoClient(db_url)
  db = client['hockey']
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
    })
    # print(predicted_winner==test_winner,predicted_homeScore==test_homeScore,predicted_awayScore==test_awayScore)
  for boxscore in boxscore_list:
    winner_total = 0
    home_score_total = 0
    away_score_total = 0
    list_total = len(test_results[boxscore['gameDate']]['results'])
    all_list_total += list_total
    for r in test_results[boxscore['gameDate']]['results']:
      winner_total += r['winner']
      home_score_total += r['homeScore']
      away_score_total += r['awayScore']
    all_winner_total += winner_total
    all_home_score_total += home_score_total
    all_away_score_total += away_score_total
    test_results[boxscore['gameDate']]['winnerPercent'] = (winner_total / list_total) * 100
    test_results[boxscore['gameDate']]['homeScorePercent'] = (home_score_total / list_total) * 100
    test_results[boxscore['gameDate']]['awayScorePercent'] = (away_score_total / list_total) * 100
  
  test_results['allWinnerPercent'] = (all_winner_total / all_list_total) * 100
  test_results['allHomeScorePercent'] = (all_home_score_total / all_list_total) * 100
  test_results['allAwayScorePercent'] = (all_away_score_total / all_list_total) * 100
  
  return test_results

@app.route('/collect/boxscores', methods=['POST'])
def collect_boxscores():
  db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  client = MongoClient(db_url)
  db = client['hockey']
  Boxscores = db['dev_boxscores']

  startID = int(request.json['startId'])
  endID = int(request.json['endId'])
  boxscores = []
  for id in range(startID, endID+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    boxscores.append(boxscore_data)
  Boxscores.insert_many(boxscores)
  return {'status':'done'}

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

if __name__ == '__main__':
  app.run()
