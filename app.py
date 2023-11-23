from flask import Flask, request, jsonify, Response
import joblib
import requests
from process import nhl_ai

# Load your trained model
model = joblib.load('nhl_ai.joblib')

def ai(game_data):
  data = nhl_ai(game_data)

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
  elif abs(winnerId - homeId) > abs(winnerId - awayId):
    winningTeam = awayTeam
  else:
    winningTeam = 'Inconclusive'

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
        'index': f'{i+1}-{j+1}',
        'query': f'?day={i+1}&game={j+1}',
        'homeTeam': game['homeTeam']['placeName']['default'],
        'awayTeam': game['awayTeam']['placeName']['default'],
      })
  return games

if __name__ == '__main__':
  app.run()
