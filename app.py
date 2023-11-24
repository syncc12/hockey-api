from flask import Flask, request, jsonify, Response
import joblib
import requests
from process import nhl_ai
import boto3
import io

def loadModelFromS3(bucket_name, model_key):
  s3 = boto3.resource('s3')
  with io.BytesIO() as data:
    s3.Bucket(bucket_name).download_fileobj(model_key, data)
    data.seek(0)    # move back to the beginning after writing
    df = joblib.load(data)
  return df
    

def load_model_from_s3(bucket_name, model_key):
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=model_key)
    model_str = response['Body'].read()

    model = joblib.load(io.BytesIO(model_str))
    return model

# Example usage
bucket_name = 'hockey-prl'
model_key = 'models/nhl_ai.joblib'
model = loadModelFromS3(bucket_name, model_key)

# Load your trained model
# model = joblib.load('nhl_ai.joblib')

def ai(game_data):
  data = nhl_ai(game_data)

  if len(data[0]) == 0:
    return {
    'gameId': data['game_id'],
    'homeId': homeId,
    'awayId': awayId,
    'homeTeam': homeTeam,
    'awayTeam': awayTeam,
    'winnerId': -1,
    'winningTeam': -1,
    'homeScore': -1,
    'awayScore': -1,
    'offset': -1,
  }

  prediction = model.predict(data['data'])

  winnerId = int(prediction[0][2])
  homeScore = int(prediction[0][0])
  awayScore = int(prediction[0][1])
  homeId = data['home_team_id']
  awayId = data['away_team_id']
  homeTeam = data['home_team']
  awayTeam = data['away_team']

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
    'gameId': data['game_id'],
    'homeId': homeId,
    'awayId': awayId,
    'homeTeam': homeTeam,
    'awayTeam': awayTeam,
    'winnerId': winnerId,
    'winningTeam': winningTeam,
    'homeScore': homeScore,
    'awayScore': awayScore,
    'offset': offset,
  }



# Initialize Flask
app = Flask(__name__)

@app.route('/', methods=['GET'])
def root():
  return jsonify({'data': 'root'})

# Define a route for predictions
@app.route('/nhl', methods=['GET'])
def predict():
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(game_data)

@app.route('/nhl/day', methods=['GET'])
def predict_day():
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

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

@app.route('/<date>', methods=['GET'])
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
