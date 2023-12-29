import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\pages\nhl')

from flask import Flask, request, jsonify, Response
from joblib import load
from pymongo import MongoClient
from pages.nhl.service import debug, test_model, collect_boxscores, collect_training_data, predict, predict_day, predict_day_simple, predict_week, get_day_ids, date_predict, now, game_date, metadata, save_boxscores
from constants.constants import VERSION

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']

CURRENT_SEASON = db["dev_seasons"].find_one(sort=[("seasonId", -1)])['seasonId']

# model = load(f'models/nhl_ai_v{VERSION}.joblib')
model_winner = load(f'models/nhl_ai_v{VERSION}_winner.joblib')
model_homeScore = load(f'models/nhl_ai_v{VERSION}_homeScore.joblib')
model_awayScore = load(f'models/nhl_ai_v{VERSION}_awayScore.joblib')
model_totalGoals = load(f'models/nhl_ai_v{VERSION}_totalGoals.joblib')
model_goalDifferential = load(f'models/nhl_ai_v{VERSION}_goalDifferential.joblib')

app = Flask(__name__)

# app.run(host="0.0.0.0")

@app.route('/', methods=['GET'])
def root():
  return jsonify({'data': 'root'})

@app.route('/debug', methods=['GET'])
def nhl_debug():
  return debug()

@app.route('/test', methods=['GET'])
def nhl_test_model():
  startID = request.args.get('start', default=-1, type=int)
  endID = request.args.get('end', default=-1, type=int)
  show_data = request.args.get('data', default=-1, type=int)
  wager = request.args.get('wager', default=10, type=int)
  return test_model(db, startID,endID,show_data,wager, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/collect/boxscores', methods=['POST'])
def nhl_collect_boxscores():
  startID = int(request.json['startId'])
  endID = int(request.json['endId'])
  return collect_boxscores(db, startID,endID, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/collect/training-data', methods=['GET'])
def nhl_collect_training_data():
  startID = request.args.get('start', default=-1, type=int)
  endID = request.args.get('end', default=-1, type=int)
  id = request.args.get('id', default=-1, type=int)

  return collect_training_data(db, startID,endID,id, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/nhl', methods=['GET'])
def nhl_predict():
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  return predict(db, day,game, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/nhl/day', methods=['GET'])
def nhl_predict_day():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  return predict_day(db, date, day, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/nhl/day/simple', methods=['GET'])
def nhl_predict_day_simple():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  return predict_day_simple(db, date, day, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/nhl/week', methods=['GET'])
def nhl_predict_week():
  return predict_week(db, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/nhl/ids', methods=['GET'])
def nhl_get_day_ids():
  date = request.args.get('date', default='now', type=str)
  return get_day_ids(db, date, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/nhl/<date>', methods=['GET'])
def nhl_date_predict(date):
  return date_predict(db, date, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/now', methods=['GET'])
def nhl_now():
  return now(db, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/date/<date>', methods=['GET'])
def nhl_game_date(date):
  return game_date(db, date, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/metadata', methods=['GET'])
def nhl_metadata():
  return metadata(db, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

@app.route('/db/update', methods=['GET'])
def nhl_save_boxscores():
  date = request.args.get('date', default='now', type=str)
  return save_boxscores(db, date, model_winner=model_winner,model_homeScore=model_homeScore,model_awayScore=model_awayScore,model_totalGoals=model_totalGoals,model_goalDifferential=model_goalDifferential)

if __name__ == '__main__':
  app.run()
