import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages\nhl')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from flask import Flask, request, jsonify, Response
from joblib import load
from pymongo import MongoClient
from pages.nhl.service import debug, test_model, collect_boxscores, predict, predict_day, predict_day_simple, predict_week, get_day_ids, date_predict, now, game_date, metadata, save_boxscores
from constants.constants import FILE_VERSION
from util.helpers import recommended_wagers
from util.models import MODELS

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']

CURRENT_SEASON = db["dev_seasons"].find_one(sort=[("seasonId", -1)])['seasonId']

# model = load(f'models/nhl_ai_v{FILE_VERSION}.joblib')
# model_winner = load(f'models/nhl_ai_v{FILE_VERSION}_winner.joblib')
# model_homeScore = load(f'models/nhl_ai_v{FILE_VERSION}_homeScore.joblib')
# model_awayScore = load(f'models/nhl_ai_v{FILE_VERSION}_awayScore.joblib')
# model_totalGoals = load(f'models/nhl_ai_v{FILE_VERSION}_totalGoals.joblib')
# model_goalDifferential = load(f'models/nhl_ai_v{FILE_VERSION}_goalDifferential.joblib')
# model_finalPeriod = load(f'models/nhl_ai_v{FILE_VERSION}_finalPeriod.joblib')
# model_pastRegulation = load(f'models/nhl_ai_v{FILE_VERSION}_pastRegulation.joblib')
# model_awayShots = load(f'models/nhl_ai_v{FILE_VERSION}_awayShots.joblib')
# model_homeShots = load(f'models/nhl_ai_v{FILE_VERSION}_homeShots.joblib')
# model_awayShotsPeriod1 = load(f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod1.joblib')
# model_homeShotsPeriod1 = load(f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod1.joblib')
# model_awayShotsPeriod2 = load(f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod2.joblib')
# model_homeShotsPeriod2 = load(f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod2.joblib')
# model_awayShotsPeriod3 = load(f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod3.joblib')
# model_homeShotsPeriod3 = load(f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod3.joblib')
# model_awayShotsPeriod4 = load(f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod4.joblib')
# model_homeShotsPeriod4 = load(f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod4.joblib')
# model_awayShotsPeriod5 = load(f'models/nhl_ai_v{FILE_VERSION}_awayShotsPeriod5.joblib')
# model_homeShotsPeriod5 = load(f'models/nhl_ai_v{FILE_VERSION}_homeShotsPeriod5.joblib')
# model_awayScorePeriod1 = load(f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod1.joblib')
# model_homeScorePeriod1 = load(f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod1.joblib')
# model_awayScorePeriod2 = load(f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod2.joblib')
# model_homeScorePeriod2 = load(f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod2.joblib')
# model_awayScorePeriod3 = load(f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod3.joblib')
# model_homeScorePeriod3 = load(f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod3.joblib')
# model_awayScorePeriod4 = load(f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod4.joblib')
# model_homeScorePeriod4 = load(f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod4.joblib')
# model_awayScorePeriod5 = load(f'models/nhl_ai_v{FILE_VERSION}_awayScorePeriod5.joblib')
# model_homeScorePeriod5 = load(f'models/nhl_ai_v{FILE_VERSION}_homeScorePeriod5.joblib')
# model_period1PuckLine = load(f'models/nhl_ai_v{FILE_VERSION}_period1PuckLine.joblib')
# model_period2PuckLine = load(f'models/nhl_ai_v{FILE_VERSION}_period2PuckLine.joblib')
# model_period3PuckLine = load(f'models/nhl_ai_v{FILE_VERSION}_period3PuckLine.joblib')
models = MODELS

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
  return test_model(db, startID,endID,show_data,wager, models)

@app.route('/collect/boxscores', methods=['POST'])
def nhl_collect_boxscores():
  startID = int(request.json['startId'])
  endID = int(request.json['endId'])
  return collect_boxscores(db, startID,endID, models)

# @app.route('/collect/training-data', methods=['GET'])
# def nhl_collect_training_data():
#   startID = request.args.get('start', default=-1, type=int)
#   endID = request.args.get('end', default=-1, type=int)
#   id = request.args.get('id', default=-1, type=int)

#   return collect_training_data(db, startID,endID,id, models)

@app.route('/nhl', methods=['GET'])
def nhl_predict():
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  date = request.args.get('date', default="now", type=str)
  return predict(db,day,game,date, models)

@app.route('/nhl/day', methods=['GET'])
def nhl_predict_day():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  prediction = predict_day(db, date, day, models)
  
  recommended_wagers(100,prediction,False)
  return jsonify(prediction)

@app.route('/nhl/day/simple', methods=['GET'])
def nhl_predict_day_simple():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  return predict_day_simple(db, date, day, models)

@app.route('/nhl/week', methods=['GET'])
def nhl_predict_week():
  return predict_week(db, models)

@app.route('/nhl/ids', methods=['GET'])
def nhl_get_day_ids():
  date = request.args.get('date', default='now', type=str)
  return get_day_ids(db, date, models)

@app.route('/nhl/<date>', methods=['GET'])
def nhl_date_predict(date):
  return date_predict(db, date, models)

@app.route('/now', methods=['GET'])
def nhl_now():
  return now(db, models)

@app.route('/date/<date>', methods=['GET'])
def nhl_game_date(date):
  return game_date(db, date, models)

@app.route('/metadata', methods=['GET'])
def nhl_metadata():
  return metadata(db, models)

@app.route('/db/update', methods=['GET'])
def nhl_save_boxscores():
  date = request.args.get('date', default='now', type=str)
  return save_boxscores(db, date, models)

if __name__ == '__main__':
  app.run()
