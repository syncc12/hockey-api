from flask import Flask, request, jsonify, Response
from joblib import load
import requests
from process import nhl_ai
import boto3
import io
import os

LATEST_DATE_TRAINED = '2023-11-11'
LATEST_DATE_COLLECTED = '2023-11-17'
LATEST_ID_COLLECTED = '2023020253'

# def loadModelFromS3(bucket_name, model_key):
#   aws = boto3.Session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
#   s3 = aws.resource('s3')
#   with io.BytesIO() as data:
#     s3.Bucket(bucket_name).download_fileobj(model_key, data)
#     data.seek(0)    # move back to the beginning after writing
#     df = joblib.load(data)
#   return df
    

# def load_model_from_s3(bucket_name, model_key):
#   try:
#     aws = boto3.Session(aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
#     s3 = aws.client('s3')
#     response = s3.get_object(Bucket=bucket_name, Key=model_key)
#     model_res = response['Body']
#     model_str = model_res.read()

#     model = joblib.load(io.BytesIO(model_str))
#     return model
#   except Exception as error:
#     print('load_model_from_s3 - error', error)

# def stream_s3_file(bucket_name, object_name):
#   s3 = boto3.client('s3')
#   s3_response_object = s3.get_object(Bucket=bucket_name, Key=object_name)
#   return s3_response_object['Body']

# def load_joblib_object(bucket_name, object_name):
#   stream = stream_s3_file(bucket_name, object_name)
#   # Using a BytesIO object as a buffer
#   buffer = io.BytesIO(stream.read())
#   return joblib.load(buffer)


# try:
#   bucket_name = os.getenv('AWS_BUCKET_NAME')
#   model_key = os.getenv('AWS_MODEL_KEY')
#   model = load_joblib_object(bucket_name, model_key)
# except:
#   print('error')


# Load your trained model
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
    if abs(winnerId - homeId) < abs(winnerId - awayId):
      winningTeam = homeTeam
      offset = abs(winnerId - homeId)
    elif abs(winnerId - homeId) > abs(winnerId - awayId):
      winningTeam = awayTeam
      offset = abs(winnerId - awayId)
    else:
      winningTeam = 'Inconclusive'
      offset = -1


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
    'live': {
      'away': live_away,
      'home': live_home,
      'period': live_period,
      'clock': live_clock,
      'stopped': live_stopped,
      'intermission': live_intermission,
    },
    # 'data': data['data']['data'],
    'message': data['message'],
  }

def ai(game_data):
  data = nhl_ai(game_data)

  if len(data['data']['data'][0]) == 0:
    return ai_return_dict(data,[[]])
    # return {
    #   'gameId': data['data']['game_id'],
    #   'date': -1,
    #   'state': 'OFF',
    #   'homeId': data['data']['home_team']['id'],
    #   'awayId': data['data']['away_team']['id'],
    #   'homeTeam': data['data']['home_team']['city'],
    #   'awayTeam': data['data']['away_team']['city'],
    #   'winnerId': -1,
    #   'winningTeam': -1,
    #   'homeScore': -1,
    #   'awayScore': -1,
    #   'offset': -1,
    #   'live': {
    #     'away': -1,
    #     'home': -1,
    #     'period': 0,
    #     'clock': 0,
    #     'stopped': True,
    #     'intermission': False,
    #   },
    #   # 'data': data['data']['data'],
    #   'message': data['message'],
    # }

  prediction = model.predict(data['data']['data'])

  return ai_return_dict(data,prediction)

  # winnerId = int(prediction[0][2])
  # homeScore = int(prediction[0][0])
  # awayScore = int(prediction[0][1])
  # homeId = data['data']['home_team']['id']
  # awayId = data['data']['away_team']['id']
  # homeTeam = f"{data['data']['home_team']['city']} {data['data']['home_team']['name']}"
  # awayTeam = f"{data['data']['away_team']['city']} {data['data']['away_team']['name']}"

  # if abs(winnerId - homeId) < abs(winnerId - awayId):
  #   winningTeam = homeTeam
  #   offset = abs(winnerId - homeId)
  # elif abs(winnerId - homeId) > abs(winnerId - awayId):
  #   winningTeam = awayTeam
  #   offset = abs(winnerId - awayId)
  # else:
  #   winningTeam = 'Inconclusive'
  #   offset = -1

  # return {
  #   'gameId': data['data']['game_id'],
  #   'date': data['data']['date'],
  #   'state': data['data']['state'],
  #   'homeId': homeId,
  #   'awayId': awayId,
  #   'homeTeam': homeTeam,
  #   'awayTeam': awayTeam,
  #   'winnerId': winnerId,
  #   'winningTeam': winningTeam,
  #   'homeScore': homeScore,
  #   'awayScore': awayScore,
  #   'offset': offset,
  #   'live': {
  #     'away': data['data']['live']['away_score'],
  #     'home': data['data']['live']['home_score'],
  #     'period': data['data']['live']['period'],
  #     'clock': data['data']['live']['clock'],
  #     'stopped': data['data']['live']['stopped'],
  #     'intermission': data['data']['live']['intermission'],
  #   },
  #   # 'data': data['data']['data'],
  #   'message': data['message'],
  # }



# Initialize Flask
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
