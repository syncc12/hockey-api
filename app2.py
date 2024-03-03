import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages\nhl')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

# import h2o
# h2o.init()

from flask import Flask, request, jsonify, Response
from joblib import load
from pymongo import MongoClient
from pages.nhl.service import predict_team_day, predict_team_day_simple, predict_team_day_receipt
from constants.constants import FILE_VERSION
from util.helpers import recommended_wagers
from util.team_models import W_MODELS, L_MODELS

db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
# db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
client = MongoClient(db_url)
db = client['hockey']

CURRENT_SEASON = db["dev_seasons"].find_one(sort=[("seasonId", -1)])['seasonId']

wModels = W_MODELS
lModels = L_MODELS

app = Flask(__name__)


# app.run(host="0.0.0.0")

@app.route('/', methods=['GET'])
def root():
  return jsonify({'data': 'root'})

@app.route('/nhl/team/day', methods=['GET'])
def nhl_predict_team_day():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=-1, type=int)
  projectedLineup = request.args.get('projectedLineup', default=False, type=bool)
  return predict_team_day(db, date, day, game, projectedLineup, wModels, lModels)

@app.route('/nhl/team/day/simple', methods=['GET'])
def nhl_predict_team_day_simple():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=-1, type=int)
  projectedLineup = request.args.get('projectedLineup', default=False, type=bool)
  return predict_team_day_simple(db, date, day, game, projectedLineup, wModels, lModels)

@app.route('/nhl/team/day/receipt', methods=['GET'])
def nhl_predict_team_day_receipt():
  date = request.args.get('date', default='now', type=str)
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=-1, type=int)
  projectedLineup = request.args.get('projectedLineup', default=False, type=bool)
  return predict_team_day_receipt(db, date, day, game, projectedLineup, wModels, lModels)

if __name__ == '__main__':
  app.run()
