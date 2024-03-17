import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from flask import Flask, request, jsonify
from pages.mlb.service import predict_team_day, predict_team_day_simple, predict_team_day_receipt
from pages.mlb.models import W_MODELS
from pages.mlb.helpers import now

wModels = W_MODELS

app = Flask(__name__)


# app.run(host="0.0.0.0")

@app.route('/', methods=['GET'])
def root():
  return jsonify({'data': 'root'})

@app.route('/mlb/day', methods=['GET'])
def mlb_predict_team_day():
  default_date = now()
  date = request.args.get('date', default=default_date, type=str)
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=-1, type=int)
  return predict_team_day(date, day, game, wModels)

@app.route('/mlb/day/simple', methods=['GET'])
def mlb_predict_team_day_simple():
  default_date = now()
  date = request.args.get('date', default=default_date, type=str)
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=-1, type=int)
  return predict_team_day_simple(date, day, game, wModels)

@app.route('/mlb/day/receipt', methods=['GET'])
def mlb_predict_team_day_receipt():
  default_date = now()
  date = request.args.get('date', default=default_date, type=str)
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=-1, type=int)
  return predict_team_day_receipt(date, day, game, wModels)

if __name__ == '__main__':
  app.run()
