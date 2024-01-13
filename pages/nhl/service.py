import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
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
# from util.training_data import save_training_data
from util.helpers import false_chain, latestIDs, adjusted_winner, test_recommended_wagers
from inputs.inputs import master_inputs
from pages.nhl.nhl_helpers import ai, ai_return_dict
from constants.constants import VERSION, FILE_VERSION

def debug():
  res = requests.get("https://api-web.nhle.com/v1/schedule/2023-11-22").json()
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]
  data = nhl_ai(game_data)
  return jsonify(data)


def test_model(db,startID,endID,show_data,wager,**kwargs):
  Boxscores = db['dev_boxscores']
  Odds = db['dev_odds']

  if startID == -1 or endID == -1:
    md = metadata(db)
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
  all_finalPeriod_total = 0
  all_pastRegulation_total = 0
  all_awayShots_total = 0
  all_homeShots_total = 0
  all_awayShotsPeriod1_total = 0
  all_homeShotsPeriod1_total = 0
  all_awayShotsPeriod2_total = 0
  all_homeShotsPeriod2_total = 0
  all_awayShotsPeriod3_total = 0
  all_homeShotsPeriod3_total = 0
  all_awayShotsPeriod4_total = 0
  all_homeShotsPeriod4_total = 0
  all_awayShotsPeriod5_total = 0
  all_homeShotsPeriod5_total = 0
  all_awayScorePeriod1_total = 0
  all_homeScorePeriod1_total = 0
  all_awayScorePeriod2_total = 0
  all_homeScorePeriod2_total = 0
  all_awayScorePeriod3_total = 0
  all_homeScorePeriod3_total = 0
  all_awayScorePeriod4_total = 0
  all_homeScorePeriod4_total = 0
  all_awayScorePeriod5_total = 0
  all_homeScorePeriod5_total = 0
  all_period1PuckLine_total = 0
  all_period2PuckLine_total = 0
  all_period3PuckLine_total = 0
  all_winnings_total = 0
  all_winnings10_total = 0
  all_winnings100_total = 0
  all_returns_total = 0
  all_returns10_total = 0
  all_returns100_total = 0
  all_total_wagered = 0
  all_total_wagered10 = 0
  all_total_wagered100 = 0
  all_list_total = 0
  all_ids = []
  odds_dict = {}
  confidence_dict = {}
  winners_dict = {}

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
    test_data = nhl_test(db=db,boxscore=boxscore)
    test_prediction_winner = kwargs['model_winner'].predict(test_data['data'])
    test_prediction_homeScore = kwargs['model_homeScore'].predict(test_data['data'])
    test_prediction_awayScore = kwargs['model_awayScore'].predict(test_data['data'])
    test_prediction_totalGoals = kwargs['model_totalGoals'].predict(test_data['data'])
    test_prediction_pastRegulation = kwargs['model_pastRegulation'].predict(test_data['data'])
    test_prediction_goalDifferential = kwargs['model_goalDifferential'].predict(test_data['data'])
    test_prediction_finalPeriod = kwargs['model_finalPeriod'].predict(test_data['data'])
    test_prediction_awayShots = kwargs['model_awayShots'].predict(test_data['data'])
    test_prediction_homeShots = kwargs['model_homeShots'].predict(test_data['data'])
    test_prediction_awayShotsPeriod1 = kwargs['model_awayShotsPeriod1'].predict(test_data['data'])
    test_prediction_homeShotsPeriod1 = kwargs['model_homeShotsPeriod1'].predict(test_data['data'])
    test_prediction_awayShotsPeriod2 = kwargs['model_awayShotsPeriod2'].predict(test_data['data'])
    test_prediction_homeShotsPeriod2 = kwargs['model_homeShotsPeriod2'].predict(test_data['data'])
    test_prediction_awayShotsPeriod3 = kwargs['model_awayShotsPeriod3'].predict(test_data['data'])
    test_prediction_homeShotsPeriod3 = kwargs['model_homeShotsPeriod3'].predict(test_data['data'])
    test_prediction_awayShotsPeriod4 = kwargs['model_awayShotsPeriod4'].predict(test_data['data'])
    test_prediction_homeShotsPeriod4 = kwargs['model_homeShotsPeriod4'].predict(test_data['data'])
    test_prediction_awayShotsPeriod5 = kwargs['model_awayShotsPeriod5'].predict(test_data['data'])
    test_prediction_homeShotsPeriod5 = kwargs['model_homeShotsPeriod5'].predict(test_data['data'])
    test_prediction_awayScorePeriod1 = kwargs['model_awayScorePeriod1'].predict(test_data['data'])
    test_prediction_homeScorePeriod1 = kwargs['model_homeScorePeriod1'].predict(test_data['data'])
    test_prediction_awayScorePeriod2 = kwargs['model_awayScorePeriod2'].predict(test_data['data'])
    test_prediction_homeScorePeriod2 = kwargs['model_homeScorePeriod2'].predict(test_data['data'])
    test_prediction_awayScorePeriod3 = kwargs['model_awayScorePeriod3'].predict(test_data['data'])
    test_prediction_homeScorePeriod3 = kwargs['model_homeScorePeriod3'].predict(test_data['data'])
    test_prediction_awayScorePeriod4 = kwargs['model_awayScorePeriod4'].predict(test_data['data'])
    test_prediction_homeScorePeriod4 = kwargs['model_homeScorePeriod4'].predict(test_data['data'])
    test_prediction_awayScorePeriod5 = kwargs['model_awayScorePeriod5'].predict(test_data['data'])
    test_prediction_homeScorePeriod5 = kwargs['model_homeScorePeriod5'].predict(test_data['data'])
    test_prediction_period1PuckLine = kwargs['model_period1PuckLine'].predict(test_data['data'])
    test_prediction_period2PuckLine = kwargs['model_period2PuckLine'].predict(test_data['data'])
    test_prediction_period3PuckLine = kwargs['model_period3PuckLine'].predict(test_data['data'])
    
    test_confidence_winner = kwargs['model_winner'].predict_proba(test_data['data'])
    test_confidence_homeScore = kwargs['model_homeScore'].predict_proba(test_data['data'])
    test_confidence_awayScore = kwargs['model_awayScore'].predict_proba(test_data['data'])
    test_confidence_totalGoals = kwargs['model_totalGoals'].predict_proba(test_data['data'])
    test_confidence_goalDifferential = kwargs['model_goalDifferential'].predict_proba(test_data['data'])
    test_confidence_finalPeriod = kwargs['model_finalPeriod'].predict_proba(test_data['data'])
    test_confidence_pastRegulation = kwargs['model_pastRegulation'].predict_proba(test_data['data'])
    test_confidence_awayShots = kwargs['model_awayShots'].predict_proba(test_data['data'])
    test_confidence_homeShots = kwargs['model_homeShots'].predict_proba(test_data['data'])
    test_confidence_awayShotsPeriod1 = kwargs['model_awayShotsPeriod1'].predict_proba(test_data['data'])
    test_confidence_homeShotsPeriod1 = kwargs['model_homeShotsPeriod1'].predict_proba(test_data['data'])
    test_confidence_awayShotsPeriod2 = kwargs['model_awayShotsPeriod2'].predict_proba(test_data['data'])
    test_confidence_homeShotsPeriod2 = kwargs['model_homeShotsPeriod2'].predict_proba(test_data['data'])
    test_confidence_awayShotsPeriod3 = kwargs['model_awayShotsPeriod3'].predict_proba(test_data['data'])
    test_confidence_homeShotsPeriod3 = kwargs['model_homeShotsPeriod3'].predict_proba(test_data['data'])
    test_confidence_awayShotsPeriod4 = kwargs['model_awayShotsPeriod4'].predict_proba(test_data['data'])
    test_confidence_homeShotsPeriod4 = kwargs['model_homeShotsPeriod4'].predict_proba(test_data['data'])
    test_confidence_awayShotsPeriod5 = kwargs['model_awayShotsPeriod5'].predict_proba(test_data['data'])
    test_confidence_homeShotsPeriod5 = kwargs['model_homeShotsPeriod5'].predict_proba(test_data['data'])
    test_confidence_awayScorePeriod1 = kwargs['model_awayScorePeriod1'].predict_proba(test_data['data'])
    test_confidence_homeScorePeriod1 = kwargs['model_homeScorePeriod1'].predict_proba(test_data['data'])
    test_confidence_awayScorePeriod2 = kwargs['model_awayScorePeriod2'].predict_proba(test_data['data'])
    test_confidence_homeScorePeriod2 = kwargs['model_homeScorePeriod2'].predict_proba(test_data['data'])
    test_confidence_awayScorePeriod3 = kwargs['model_awayScorePeriod3'].predict_proba(test_data['data'])
    test_confidence_homeScorePeriod3 = kwargs['model_homeScorePeriod3'].predict_proba(test_data['data'])
    test_confidence_awayScorePeriod4 = kwargs['model_awayScorePeriod4'].predict_proba(test_data['data'])
    test_confidence_homeScorePeriod4 = kwargs['model_homeScorePeriod4'].predict_proba(test_data['data'])
    test_confidence_awayScorePeriod5 = kwargs['model_awayScorePeriod5'].predict_proba(test_data['data'])
    test_confidence_homeScorePeriod5 = kwargs['model_homeScorePeriod5'].predict_proba(test_data['data'])
    test_confidence_period1PuckLine = kwargs['model_period1PuckLine'].predict_proba(test_data['data'])
    test_confidence_period2PuckLine = kwargs['model_period2PuckLine'].predict_proba(test_data['data'])
    test_confidence_period3PuckLine = kwargs['model_period3PuckLine'].predict_proba(test_data['data'])
    
    predicted_winner = adjusted_winner(awayId, homeId, test_prediction_winner[0])
    predicted_homeScore = test_prediction_homeScore[0]
    predicted_awayScore = test_prediction_awayScore[0]
    predicted_totalGoals = test_prediction_totalGoals[0]
    predicted_goalDifferential = test_prediction_goalDifferential[0]
    predicted_finalPeriod = test_prediction_finalPeriod[0]
    predicted_pastRegulation = test_prediction_pastRegulation[0]
    predicted_awayShots = test_prediction_awayShots[0]
    predicted_homeShots = test_prediction_homeShots[0]
    predicted_awayShotsPeriod1 = test_prediction_awayShotsPeriod1[0]
    predicted_homeShotsPeriod1 = test_prediction_homeShotsPeriod1[0]
    predicted_awayShotsPeriod2 = test_prediction_awayShotsPeriod2[0]
    predicted_homeShotsPeriod2 = test_prediction_homeShotsPeriod2[0]
    predicted_awayShotsPeriod3 = test_prediction_awayShotsPeriod3[0]
    predicted_homeShotsPeriod3 = test_prediction_homeShotsPeriod3[0]
    predicted_awayShotsPeriod4 = test_prediction_awayShotsPeriod4[0]
    predicted_homeShotsPeriod4 = test_prediction_homeShotsPeriod4[0]
    predicted_awayShotsPeriod5 = test_prediction_awayShotsPeriod5[0]
    predicted_homeShotsPeriod5 = test_prediction_homeShotsPeriod5[0]
    predicted_awayScorePeriod1 = test_prediction_awayScorePeriod1[0]
    predicted_homeScorePeriod1 = test_prediction_homeScorePeriod1[0]
    predicted_awayScorePeriod2 = test_prediction_awayScorePeriod2[0]
    predicted_homeScorePeriod2 = test_prediction_homeScorePeriod2[0]
    predicted_awayScorePeriod3 = test_prediction_awayScorePeriod3[0]
    predicted_homeScorePeriod3 = test_prediction_homeScorePeriod3[0]
    predicted_awayScorePeriod4 = test_prediction_awayScorePeriod4[0]
    predicted_homeScorePeriod4 = test_prediction_homeScorePeriod4[0]
    predicted_awayScorePeriod5 = test_prediction_awayScorePeriod5[0]
    predicted_homeScorePeriod5 = test_prediction_homeScorePeriod5[0]
    predicted_period1PuckLine = test_prediction_period1PuckLine[0]
    predicted_period2PuckLine = test_prediction_period2PuckLine[0]
    predicted_period3PuckLine = test_prediction_period3PuckLine[0]
    test_winner = adjusted_winner(awayId, homeId, test_data['result']['winner'])
    # print("test_data['result']",test_data['result'])
    test_homeScore = test_data['result']['homeScore']
    test_awayScore = test_data['result']['awayScore']
    test_totalGoals = test_data['result']['totalGoals']
    test_goalDifferential = test_data['result']['goalDifferential']
    test_finalPeriod = test_data['result']['finalPeriod']
    test_pastRegulation = test_data['result']['pastRegulation']
    test_awayShots = test_data['result']['awayShots']
    test_homeShots = test_data['result']['homeShots']
    test_awayShotsPeriod1 = test_data['result']['awayShotsPeriod1']
    test_homeShotsPeriod1 = test_data['result']['homeShotsPeriod1']
    test_awayShotsPeriod2 = test_data['result']['awayShotsPeriod2']
    test_homeShotsPeriod2 = test_data['result']['homeShotsPeriod2']
    test_awayShotsPeriod3 = test_data['result']['awayShotsPeriod3']
    test_homeShotsPeriod3 = test_data['result']['homeShotsPeriod3']
    test_awayShotsPeriod4 = test_data['result']['awayShotsPeriod4']
    test_homeShotsPeriod4 = test_data['result']['homeShotsPeriod4']
    test_awayShotsPeriod5 = test_data['result']['awayShotsPeriod5']
    test_homeShotsPeriod5 = test_data['result']['homeShotsPeriod5']
    test_awayScorePeriod1 = test_data['result']['awayScorePeriod1']
    test_homeScorePeriod1 = test_data['result']['homeScorePeriod1']
    test_awayScorePeriod2 = test_data['result']['awayScorePeriod2']
    test_homeScorePeriod2 = test_data['result']['homeScorePeriod2']
    test_awayScorePeriod3 = test_data['result']['awayScorePeriod3']
    test_homeScorePeriod3 = test_data['result']['homeScorePeriod3']
    test_awayScorePeriod4 = test_data['result']['awayScorePeriod4']
    test_homeScorePeriod4 = test_data['result']['homeScorePeriod4']
    test_awayScorePeriod5 = test_data['result']['awayScorePeriod5']
    test_homeScorePeriod5 = test_data['result']['homeScorePeriod5']
    test_period1PuckLine = test_data['result']['period1PuckLine']
    test_period2PuckLine = test_data['result']['period2PuckLine']
    test_period3PuckLine = test_data['result']['period3PuckLine']
    test_results[boxscore['gameDate']]['results'].append({
      'id': boxscore['id'],
      'winner': 1 if predicted_winner==test_winner else 0,
      'homeScore': 1 if predicted_homeScore==test_homeScore else 0,
      'awayScore': 1 if predicted_awayScore==test_awayScore else 0,
      'totalGoals': 1 if predicted_totalGoals==test_totalGoals else 0,
      'goalDifferential': 1 if predicted_goalDifferential==test_goalDifferential else 0,
      'finalPeriod': 1 if predicted_finalPeriod==test_finalPeriod else 0,
      'pastRegulation': 1 if predicted_pastRegulation==test_pastRegulation else 0,
      'awayShots': 1 if predicted_awayShots==test_awayShots else 0,
      'homeShots': 1 if predicted_homeShots==test_homeShots else 0,
      'awayShotsPeriod1': 1 if predicted_awayShotsPeriod1==test_awayShotsPeriod1 else 0,
      'homeShotsPeriod1': 1 if predicted_homeShotsPeriod1==test_homeShotsPeriod1 else 0,
      'awayShotsPeriod2': 1 if predicted_awayShotsPeriod2==test_awayShotsPeriod2 else 0,
      'homeShotsPeriod2': 1 if predicted_homeShotsPeriod2==test_homeShotsPeriod2 else 0,
      'awayShotsPeriod3': 1 if predicted_awayShotsPeriod3==test_awayShotsPeriod3 else 0,
      'homeShotsPeriod3': 1 if predicted_homeShotsPeriod3==test_homeShotsPeriod3 else 0,
      'awayShotsPeriod4': 1 if predicted_awayShotsPeriod4==test_awayShotsPeriod4 else 0,
      'homeShotsPeriod4': 1 if predicted_homeShotsPeriod4==test_homeShotsPeriod4 else 0,
      'awayShotsPeriod5': 1 if predicted_awayShotsPeriod5==test_awayShotsPeriod5 else 0,
      'homeShotsPeriod5': 1 if predicted_homeShotsPeriod5==test_homeShotsPeriod5 else 0,
      'awayScorePeriod1': 1 if predicted_awayScorePeriod1==test_awayScorePeriod1 else 0,
      'homeScorePeriod1': 1 if predicted_homeScorePeriod1==test_homeScorePeriod1 else 0,
      'awayScorePeriod2': 1 if predicted_awayScorePeriod2==test_awayScorePeriod2 else 0,
      'homeScorePeriod2': 1 if predicted_homeScorePeriod2==test_homeScorePeriod2 else 0,
      'awayScorePeriod3': 1 if predicted_awayScorePeriod3==test_awayScorePeriod3 else 0,
      'homeScorePeriod3': 1 if predicted_homeScorePeriod3==test_homeScorePeriod3 else 0,
      'awayScorePeriod4': 1 if predicted_awayScorePeriod4==test_awayScorePeriod4 else 0,
      'homeScorePeriod4': 1 if predicted_homeScorePeriod4==test_homeScorePeriod4 else 0,
      'awayScorePeriod5': 1 if predicted_awayScorePeriod5==test_awayScorePeriod5 else 0,
      'homeScorePeriod5': 1 if predicted_homeScorePeriod5==test_homeScorePeriod5 else 0,
      'period1PuckLine': 1 if predicted_period1PuckLine==test_period1PuckLine else 0,
      'period2PuckLine': 1 if predicted_period2PuckLine==test_period2PuckLine else 0,
      'period3PuckLine': 1 if predicted_period3PuckLine==test_period3PuckLine else 0,
      'winnerConfidence': int((np.max(test_confidence_winner, axis=1) * 100)[0]),
      'homeScoreConfidence': int((np.max(test_confidence_homeScore, axis=1) * 100)[0]),
      'awayScoreConfidence': int((np.max(test_confidence_awayScore, axis=1) * 100)[0]),
      'totalGoalsConfidence': int((np.max(test_confidence_totalGoals, axis=1) * 100)[0]),
      'goalDifferentialConfidence': int((np.max(test_confidence_goalDifferential, axis=1) * 100)[0]),
      'finalPeriodConfidence': int((np.max(test_confidence_finalPeriod, axis=1) * 100)[0]),
      'pastRegulationConfidence': int((np.max(test_confidence_pastRegulation, axis=1) * 100)[0]),
      'awayShotsConfidence': int((np.max(test_confidence_awayShots, axis=1) * 100)[0]),
      'homeShotsConfidence': int((np.max(test_confidence_homeShots, axis=1) * 100)[0]),
      'awayShotsPeriod1Confidence': int((np.max(test_confidence_awayShotsPeriod1, axis=1) * 100)[0]),
      'homeShotsPeriod1Confidence': int((np.max(test_confidence_homeShotsPeriod1, axis=1) * 100)[0]),
      'awayShotsPeriod2Confidence': int((np.max(test_confidence_awayShotsPeriod2, axis=1) * 100)[0]),
      'homeShotsPeriod2Confidence': int((np.max(test_confidence_homeShotsPeriod2, axis=1) * 100)[0]),
      'awayShotsPeriod3Confidence': int((np.max(test_confidence_awayShotsPeriod3, axis=1) * 100)[0]),
      'homeShotsPeriod3Confidence': int((np.max(test_confidence_homeShotsPeriod3, axis=1) * 100)[0]),
      'awayShotsPeriod4Confidence': int((np.max(test_confidence_awayShotsPeriod4, axis=1) * 100)[0]),
      'homeShotsPeriod4Confidence': int((np.max(test_confidence_homeShotsPeriod4, axis=1) * 100)[0]),
      'awayShotsPeriod5Confidence': int((np.max(test_confidence_awayShotsPeriod5, axis=1) * 100)[0]),
      'homeShotsPeriod5Confidence': int((np.max(test_confidence_homeShotsPeriod5, axis=1) * 100)[0]),
      'awayScorePeriod1Confidence': int((np.max(test_confidence_awayScorePeriod1, axis=1) * 100)[0]),
      'homeScorePeriod1Confidence': int((np.max(test_confidence_homeScorePeriod1, axis=1) * 100)[0]),
      'awayScorePeriod2Confidence': int((np.max(test_confidence_awayScorePeriod2, axis=1) * 100)[0]),
      'homeScorePeriod2Confidence': int((np.max(test_confidence_homeScorePeriod2, axis=1) * 100)[0]),
      'awayScorePeriod3Confidence': int((np.max(test_confidence_awayScorePeriod3, axis=1) * 100)[0]),
      'homeScorePeriod3Confidence': int((np.max(test_confidence_homeScorePeriod3, axis=1) * 100)[0]),
      'awayScorePeriod4Confidence': int((np.max(test_confidence_awayScorePeriod4, axis=1) * 100)[0]),
      'homeScorePeriod4Confidence': int((np.max(test_confidence_homeScorePeriod4, axis=1) * 100)[0]),
      'awayScorePeriod5Confidence': int((np.max(test_confidence_awayScorePeriod5, axis=1) * 100)[0]),
      'homeScorePeriod5Confidence': int((np.max(test_confidence_homeScorePeriod5, axis=1) * 100)[0]),
      'period1PuckLineConfidence': int((np.max(test_confidence_period1PuckLine, axis=1) * 100)[0]),
      'period2PuckLineConfidence': int((np.max(test_confidence_period2PuckLine, axis=1) * 100)[0]),
      'period3PuckLineConfidence': int((np.max(test_confidence_period3PuckLine, axis=1) * 100)[0]),
    })
    test_results_len = len(test_results[boxscore['gameDate']]['results']) - 1
    if show_data != -1:
      test_results[boxscore['gameDate']]['results'][test_results_len]['data'] = test_data['input_data']
    game_odds = Odds.find_one({'id':boxscore['id']})
    winnings = 0
    winnings10 = 0
    winnings100 = 0
    returns = 0
    returns10 = 0
    returns100 = 0
    if game_odds:
      test_results[boxscore['gameDate']]['results'][test_results_len]['awayOdds'] = float(game_odds['odds']['awayTeam'])
      test_results[boxscore['gameDate']]['results'][test_results_len]['homeOdds'] = float(game_odds['odds']['homeTeam'])
      winning_odds = float(game_odds['odds']['awayTeam']) if test_winner == awayId else float(game_odds['odds']['homeTeam'])
      if not boxscore['gameDate'] in odds_dict:
        odds_dict[boxscore['gameDate']] = []
      odds_dict[boxscore['gameDate']].append(winning_odds)
      if not boxscore['gameDate'] in confidence_dict:
        confidence_dict[boxscore['gameDate']] = []
      confidence_dict[boxscore['gameDate']].append(int((np.max(test_confidence_winner, axis=1) * 100)[0]))
      
      if not boxscore['gameDate'] in winners_dict:
        winners_dict[boxscore['gameDate']] = []

      if test_results[boxscore['gameDate']]['results'][test_results_len]['winner']:
        winnings = abs(((100/winning_odds)*wager) if winning_odds < 0 else ((winning_odds/100)*wager))
        winnings10 = abs(((100/winning_odds)*10) if winning_odds < 0 else ((winning_odds/100)*10))
        winnings100 = abs(((100/winning_odds)*100) if winning_odds < 0 else ((winning_odds/100)*100))
        returns = winnings + wager
        returns10 = winnings10 + 10
        returns100 = winnings100 + 100
        winners_dict[boxscore['gameDate']].append(1)
      else:
        winners_dict[boxscore['gameDate']].append(0)
      
      test_results[boxscore['gameDate']]['results'][test_results_len]['betting'] = {
        'winnings': {
          'wager': winnings,
          '10': winnings10,
          '100': winnings100,
        },
        'returns': {
          'wager': returns,
          '10': returns10,
          '100': returns100,
        },
      }
      

    
    ## All Totals
    winner_total = 0
    home_score_total = 0
    away_score_total = 0
    home_away_score_total = 0
    goal_total = 0
    goal_differential_total = 0
    finalPeriod_total = 0
    pastRegulation_total = 0
    awayShots_total = 0
    homeShots_total = 0
    awayShotsPeriod1_total = 0
    homeShotsPeriod1_total = 0
    awayShotsPeriod2_total = 0
    homeShotsPeriod2_total = 0
    awayShotsPeriod3_total = 0
    homeShotsPeriod3_total = 0
    awayShotsPeriod4_total = 0
    homeShotsPeriod4_total = 0
    awayShotsPeriod5_total = 0
    homeShotsPeriod5_total = 0
    awayScorePeriod1_total = 0
    homeScorePeriod1_total = 0
    awayScorePeriod2_total = 0
    homeScorePeriod2_total = 0
    awayScorePeriod3_total = 0
    homeScorePeriod3_total = 0
    awayScorePeriod4_total = 0
    homeScorePeriod4_total = 0
    awayScorePeriod5_total = 0
    homeScorePeriod5_total = 0
    period1PuckLine_total = 0
    period2PuckLine_total = 0
    period3PuckLine_total = 0
    winnings_total = 0
    winnings10_total = 0
    winnings100_total = 0
    returns_total = 0
    returns10_total = 0
    returns100_total = 0
    list_total = len(test_results[boxscore['gameDate']]['results'])
    all_list_total += list_total
    for r in test_results[boxscore['gameDate']]['results']:
      winner_total += r['winner']
      home_score_total += r['homeScore']
      away_score_total += r['awayScore']
      goal_total += r['totalGoals']
      goal_differential_total += r['goalDifferential']
      finalPeriod_total += r['finalPeriod']
      pastRegulation_total += r['pastRegulation']
      awayShots_total += r['awayShots']
      homeShots_total += r['homeShots']
      awayShotsPeriod1_total += r['awayShotsPeriod1']
      homeShotsPeriod1_total += r['homeShotsPeriod1']
      awayShotsPeriod2_total += r['awayShotsPeriod2']
      homeShotsPeriod2_total += r['homeShotsPeriod2']
      awayShotsPeriod3_total += r['awayShotsPeriod3']
      homeShotsPeriod3_total += r['homeShotsPeriod3']
      awayShotsPeriod4_total += r['awayShotsPeriod4']
      homeShotsPeriod4_total += r['homeShotsPeriod4']
      awayShotsPeriod5_total += r['awayShotsPeriod5']
      homeShotsPeriod5_total += r['homeShotsPeriod5']
      awayScorePeriod1_total += r['awayScorePeriod1']
      homeScorePeriod1_total += r['homeScorePeriod1']
      awayScorePeriod2_total += r['awayScorePeriod2']
      homeScorePeriod2_total += r['homeScorePeriod2']
      awayScorePeriod3_total += r['awayScorePeriod3']
      homeScorePeriod3_total += r['homeScorePeriod3']
      awayScorePeriod4_total += r['awayScorePeriod4']
      homeScorePeriod4_total += r['homeScorePeriod4']
      awayScorePeriod5_total += r['awayScorePeriod5']
      homeScorePeriod5_total += r['homeScorePeriod5']
      period1PuckLine_total += r['period1PuckLine']
      period2PuckLine_total += r['period2PuckLine']
      period3PuckLine_total += r['period3PuckLine']
      # if game_odds:
      #   winnings_total += r['betting']['winnings']['wager']
      #   winnings10_total += r['betting']['winnings']['10']
      #   winnings100_total += r['betting']['winnings']['100']
      #   returns_total += r['betting']['returns']['wager']
      #   returns10_total += r['betting']['returns']['10']
      #   returns100_total += r['betting']['returns']['100']
      if home_score_total == 1 and away_score_total == 1:
        home_away_score_total += 1
    all_winner_total += winner_total
    all_home_score_total += home_score_total
    all_away_score_total += away_score_total
    all_home_away_score_total += home_away_score_total
    all_goal_total += goal_total
    all_goal_differential_total += goal_differential_total
    all_finalPeriod_total += finalPeriod_total
    all_pastRegulation_total += pastRegulation_total
    all_awayShots_total += awayShots_total
    all_homeShots_total += homeShots_total
    all_awayShotsPeriod1_total += awayShotsPeriod1_total
    all_homeShotsPeriod1_total += homeShotsPeriod1_total
    all_awayShotsPeriod2_total += awayShotsPeriod2_total
    all_homeShotsPeriod2_total += homeShotsPeriod2_total
    all_awayShotsPeriod3_total += awayShotsPeriod3_total
    all_homeShotsPeriod3_total += homeShotsPeriod3_total
    all_awayShotsPeriod4_total += awayShotsPeriod4_total
    all_homeShotsPeriod4_total += homeShotsPeriod4_total
    all_awayShotsPeriod5_total += awayShotsPeriod5_total
    all_homeShotsPeriod5_total += homeShotsPeriod5_total
    all_awayScorePeriod1_total += awayScorePeriod1_total
    all_homeScorePeriod1_total += homeScorePeriod1_total
    all_awayScorePeriod2_total += awayScorePeriod2_total
    all_homeScorePeriod2_total += homeScorePeriod2_total
    all_awayScorePeriod3_total += awayScorePeriod3_total
    all_homeScorePeriod3_total += homeScorePeriod3_total
    all_awayScorePeriod4_total += awayScorePeriod4_total
    all_homeScorePeriod4_total += homeScorePeriod4_total
    all_awayScorePeriod5_total += awayScorePeriod5_total
    all_homeScorePeriod5_total += homeScorePeriod5_total
    all_period1PuckLine_total += period1PuckLine_total
    all_period2PuckLine_total += period2PuckLine_total
    all_period3PuckLine_total += period3PuckLine_total
    if game_odds:
      all_winnings_total += winnings_total 
      all_winnings10_total += winnings10_total 
      all_winnings100_total += winnings100_total 
      all_returns_total += returns_total 
      all_returns10_total += returns10_total 
      all_returns100_total += returns100_total
      all_total_wagered += (list_total * wager)
      all_total_wagered10 += (list_total * 10)
      all_total_wagered100 += (list_total * 100)
      test_results[boxscore['gameDate']]['winnerPercent'] = (winner_total / list_total) * 100
      test_results[boxscore['gameDate']]['homeScorePercent'] = (home_score_total / list_total) * 100
      test_results[boxscore['gameDate']]['awayScorePercent'] = (away_score_total / list_total) * 100
      test_results[boxscore['gameDate']]['h2hScorePercent'] = (home_away_score_total / list_total) * 100
      test_results[boxscore['gameDate']]['goalTotalPercent'] = (goal_total / list_total) * 100
      test_results[boxscore['gameDate']]['goalDifferentialPercent'] = (goal_differential_total / list_total) * 100
      test_results[boxscore['gameDate']]['betting'] = {
        'totalWinnings': {
          'wager': f'${winnings_total}',
          '10': f'${winnings10_total}',
          '100': f'${winnings100_total}',
        },
        'totalWagered': {
          'wager': f'${list_total * wager}',
          '10': f'${list_total * 10}',
          '100': f'${list_total * 100}',
        },
        'totalReturned': {
          'wager': f'${returns_total}',
          '10': f'${returns10_total}',
          '100': f'${returns100_total}',
        },
        'totalProfit': {
          'wager': f'${returns_total - (list_total * wager)}',
          '10': f'${returns10_total - (list_total * 10)}',
          '100': f'${returns100_total - (list_total * 100)}',
        },
      }
      test_results[boxscore['gameDate']]['totalGames'] = list_total
  
  for day in list(odds_dict.keys()):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(day)
    test_recommended_wagers(100,odds=odds_dict[day],confidence=confidence_dict[day],winners=winners_dict[day])

  test_results['allWinnerPercent'] = (all_winner_total / all_list_total) * 100
  test_results['allHomeScorePercent'] = (all_home_score_total / all_list_total) * 100
  test_results['allAwayScorePercent'] = (all_away_score_total / all_list_total) * 100
  test_results['allH2HScorePercent'] = (all_home_away_score_total / all_list_total) * 100
  test_results['allGoalTotalPercent'] = (all_goal_total / all_list_total) * 100
  test_results['allGoalDifferentialPercent'] = (all_goal_differential_total / all_list_total) * 100
  test_results['allFinalPeriod'] = (all_finalPeriod_total / all_list_total) * 100
  test_results['allPastRegulation'] = (all_pastRegulation_total / all_list_total) * 100
  test_results['allAwayShots'] = (all_awayShots_total / all_list_total) * 100
  test_results['allHomeShots'] = (all_homeShots_total / all_list_total) * 100
  test_results['allAwayShotsPeriod1'] = (all_awayShotsPeriod1_total / all_list_total) * 100
  test_results['allHomeShotsPeriod1'] = (all_homeShotsPeriod1_total / all_list_total) * 100
  test_results['allAwayShotsPeriod2'] = (all_awayShotsPeriod2_total / all_list_total) * 100
  test_results['allHomeShotsPeriod2'] = (all_homeShotsPeriod2_total / all_list_total) * 100
  test_results['allAwayShotsPeriod3'] = (all_awayShotsPeriod3_total / all_list_total) * 100
  test_results['allHomeShotsPeriod3'] = (all_homeShotsPeriod3_total / all_list_total) * 100
  test_results['allAwayShotsPeriod4'] = (all_awayShotsPeriod4_total / all_list_total) * 100
  test_results['allHomeShotsPeriod4'] = (all_homeShotsPeriod4_total / all_list_total) * 100
  test_results['allAwayShotsPeriod5'] = (all_awayShotsPeriod5_total / all_list_total) * 100
  test_results['allHomeShotsPeriod5'] = (all_homeShotsPeriod5_total / all_list_total) * 100
  test_results['allAwayScorePeriod1'] = (all_awayScorePeriod1_total / all_list_total) * 100
  test_results['allHomeScorePeriod1'] = (all_homeScorePeriod1_total / all_list_total) * 100
  test_results['allAwayScorePeriod2'] = (all_awayScorePeriod2_total / all_list_total) * 100
  test_results['allHomeScorePeriod2'] = (all_homeScorePeriod2_total / all_list_total) * 100
  test_results['allAwayScorePeriod3'] = (all_awayScorePeriod3_total / all_list_total) * 100
  test_results['allHomeScorePeriod3'] = (all_homeScorePeriod3_total / all_list_total) * 100
  test_results['allAwayScorePeriod4'] = (all_awayScorePeriod4_total / all_list_total) * 100
  test_results['allHomeScorePeriod4'] = (all_homeScorePeriod4_total / all_list_total) * 100
  test_results['allAwayScorePeriod5'] = (all_awayScorePeriod5_total / all_list_total) * 100
  test_results['allHomeScorePeriod5'] = (all_homeScorePeriod5_total / all_list_total) * 100
  test_results['allPeriod1PuckLine'] = (all_period1PuckLine_total / all_list_total) * 100
  test_results['allPeriod2PuckLine'] = (all_period2PuckLine_total / all_list_total) * 100
  test_results['allPeriod3PuckLine'] = (all_period3PuckLine_total / all_list_total) * 100
  test_results['allIDs'] = all_ids
  test_results['allIDLength'] = len(all_ids)
  test_results['totalGames'] = all_list_total
  if game_odds:
    test_results['betting'] = {
      'wager': f'${wager}',
      'allTotalWinnings': {
        'wager': f'${all_winnings_total}',
        '10': f'${all_winnings10_total}',
        '100': f'${all_winnings100_total}',
      },
      'allTotalWagered': {
        'wager': f'${list_total * wager}',
        '10': f'${list_total * 10}',
        '100': f'${list_total * 100}',
      },
      'allTotalReturned': {
        'wager': f'${all_returns_total}',
        '10': f'${all_returns10_total}',
        '100': f'${all_returns100_total}',
      },
      'allTotalProfit': {
        'wager': f'${returns_total - (list_total * wager)}',
        '10': f'${returns10_total - (list_total * 10)}',
        '100': f'${returns100_total - (list_total * 100)}',
      },
    }
  
  return test_results


def collect_boxscores(db,startID,endID,**kwargs):
  Boxscores = db['dev_boxscores']

  boxscores = []
  for id in range(startID, endID+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    boxscores.append(boxscore_data)
  Boxscores.insert_many(boxscores)
  return {'status':'done'}


# def collect_training_data(db,startID,endID,id,**kwargs):
#   training_data = []
#   is_neutral_site = {}
#   if startID != -1 and endID != -1 and id == -1:
#     for id in range(startID, endID+1):
#       loop_data = data_loop(id=id,is_neutral_site=is_neutral_site)
#       is_neutral_site = loop_data['is_neutral_site']
#       training_data.append(save_training_data(boxscores=loop_data['boxscore_data'],neutralSite=is_neutral_site[loop_data['boxscore_data']['id']]))
#   else:
#     if startID == -1 and endID == -1 and id != -1:
#       loop_data = data_loop(id=id,is_neutral_site=is_neutral_site)
#     elif startID != -1 and endID == -1 and id == -1:
#       loop_data = data_loop(id=startID,is_neutral_site=is_neutral_site)
#     elif startID == -1 and endID != -1 and id == -1:
#       loop_data = data_loop(id=endID,is_neutral_site=is_neutral_site)
#     training_data.append(save_training_data(boxscores=loop_data['boxscore_data'],neutralSite=is_neutral_site[loop_data['boxscore_data']['id']]))
  
#   return training_data
  
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


def predict(db,day,game,date,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(db, game_data, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'],model_finalPeriod=kwargs['model_finalPeriod'],model_pastRegulation=kwargs['model_pastRegulation'],model_awayShots=kwargs['model_awayShots'],model_homeShots=kwargs['model_homeShots'],model_awayShotsPeriod1=kwargs['model_awayShotsPeriod1'],model_homeShotsPeriod1=kwargs['model_homeShotsPeriod1'],model_awayShotsPeriod2=kwargs['model_awayShotsPeriod2'],model_homeShotsPeriod2=kwargs['model_homeShotsPeriod2'],model_awayShotsPeriod3=kwargs['model_awayShotsPeriod3'],model_homeShotsPeriod3=kwargs['model_homeShotsPeriod3'],model_awayShotsPeriod4=kwargs['model_awayShotsPeriod4'],model_homeShotsPeriod4=kwargs['model_homeShotsPeriod4'],model_awayShotsPeriod5=kwargs['model_awayShotsPeriod5'],model_homeShotsPeriod5=kwargs['model_homeShotsPeriod5'],model_awayScorePeriod1=kwargs['model_awayScorePeriod1'],model_homeScorePeriod1=kwargs['model_homeScorePeriod1'],model_awayScorePeriod2=kwargs['model_awayScorePeriod2'],model_homeScorePeriod2=kwargs['model_homeScorePeriod2'],model_awayScorePeriod3=kwargs['model_awayScorePeriod3'],model_homeScorePeriod3=kwargs['model_homeScorePeriod3'],model_awayScorePeriod4=kwargs['model_awayScorePeriod4'],model_homeScorePeriod4=kwargs['model_homeScorePeriod4'],model_awayScorePeriod5=kwargs['model_awayScorePeriod5'],model_homeScorePeriod5=kwargs['model_homeScorePeriod5'],model_period1PuckLine=kwargs['model_period1PuckLine'],model_period2PuckLine=kwargs['model_period2PuckLine'],model_period3PuckLine=kwargs['model_period3PuckLine'])


def predict_day(db,date,day,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(db, game, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'],model_finalPeriod=kwargs['model_finalPeriod'],model_pastRegulation=kwargs['model_pastRegulation'],model_awayShots=kwargs['model_awayShots'],model_homeShots=kwargs['model_homeShots'],model_awayShotsPeriod1=kwargs['model_awayShotsPeriod1'],model_homeShotsPeriod1=kwargs['model_homeShotsPeriod1'],model_awayShotsPeriod2=kwargs['model_awayShotsPeriod2'],model_homeShotsPeriod2=kwargs['model_homeShotsPeriod2'],model_awayShotsPeriod3=kwargs['model_awayShotsPeriod3'],model_homeShotsPeriod3=kwargs['model_homeShotsPeriod3'],model_awayShotsPeriod4=kwargs['model_awayShotsPeriod4'],model_homeShotsPeriod4=kwargs['model_homeShotsPeriod4'],model_awayShotsPeriod5=kwargs['model_awayShotsPeriod5'],model_homeShotsPeriod5=kwargs['model_homeShotsPeriod5'],model_awayScorePeriod1=kwargs['model_awayScorePeriod1'],model_homeScorePeriod1=kwargs['model_homeScorePeriod1'],model_awayScorePeriod2=kwargs['model_awayScorePeriod2'],model_homeScorePeriod2=kwargs['model_homeScorePeriod2'],model_awayScorePeriod3=kwargs['model_awayScorePeriod3'],model_homeScorePeriod3=kwargs['model_homeScorePeriod3'],model_awayScorePeriod4=kwargs['model_awayScorePeriod4'],model_homeScorePeriod4=kwargs['model_homeScorePeriod4'],model_awayScorePeriod5=kwargs['model_awayScorePeriod5'],model_homeScorePeriod5=kwargs['model_homeScorePeriod5'],model_period1PuckLine=kwargs['model_period1PuckLine'],model_period2PuckLine=kwargs['model_period2PuckLine'],model_period3PuckLine=kwargs['model_period3PuckLine'])
    games.append(ai_data)
  return games


def predict_day_simple(db,date,day,**kwargs):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(db, game, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'],model_finalPeriod=kwargs['model_finalPeriod'],model_pastRegulation=kwargs['model_pastRegulation'],model_awayShots=kwargs['model_awayShots'],model_homeShots=kwargs['model_homeShots'],model_awayShotsPeriod1=kwargs['model_awayShotsPeriod1'],model_homeShotsPeriod1=kwargs['model_homeShotsPeriod1'],model_awayShotsPeriod2=kwargs['model_awayShotsPeriod2'],model_homeShotsPeriod2=kwargs['model_homeShotsPeriod2'],model_awayShotsPeriod3=kwargs['model_awayShotsPeriod3'],model_homeShotsPeriod3=kwargs['model_homeShotsPeriod3'],model_awayShotsPeriod4=kwargs['model_awayShotsPeriod4'],model_homeShotsPeriod4=kwargs['model_homeShotsPeriod4'],model_awayShotsPeriod5=kwargs['model_awayShotsPeriod5'],model_homeShotsPeriod5=kwargs['model_homeShotsPeriod5'],model_awayScorePeriod1=kwargs['model_awayScorePeriod1'],model_homeScorePeriod1=kwargs['model_homeScorePeriod1'],model_awayScorePeriod2=kwargs['model_awayScorePeriod2'],model_homeScorePeriod2=kwargs['model_homeScorePeriod2'],model_awayScorePeriod3=kwargs['model_awayScorePeriod3'],model_homeScorePeriod3=kwargs['model_homeScorePeriod3'],model_awayScorePeriod4=kwargs['model_awayScorePeriod4'],model_homeScorePeriod4=kwargs['model_homeScorePeriod4'],model_awayScorePeriod5=kwargs['model_awayScorePeriod5'],model_homeScorePeriod5=kwargs['model_homeScorePeriod5'],model_period1PuckLine=kwargs['model_period1PuckLine'],model_period2PuckLine=kwargs['model_period2PuckLine'],model_period3PuckLine=kwargs['model_period3PuckLine'])
    if ai_data['message'] == 'using projected lineup':
      goalie_combos = list(ai_data['prediction'].keys())
      
      simple_data = {
        'prediction': {
          f'{goalie_combos[0]}': {
            'awayTeam': f"{ai_data['awayTeam']} - {ai_data['prediction'][goalie_combos[0]]['awayScore']} - {ai_data['prediction'][goalie_combos[0]]['awayScoreConfidence']}%",
            'homeTeam': f"{ai_data['homeTeam']} - {ai_data['prediction'][goalie_combos[0]]['homeScore']} - {ai_data['prediction'][goalie_combos[0]]['homeScoreConfidence']}%",
            'winningTeam': f"{ai_data['prediction'][goalie_combos[0]]['winningTeam']} - {ai_data['prediction'][goalie_combos[0]]['winnerConfidence']}%",
            'offset': ai_data['prediction'][goalie_combos[0]]['offset'],
            'totalGoals': f"{ai_data['prediction'][goalie_combos[0]]['totalGoals']} - {ai_data['prediction'][goalie_combos[0]]['totalGoalsConfidence']}%",
            'goalDifferential': f"{ai_data['prediction'][goalie_combos[0]]['goalDifferential']} - {ai_data['prediction'][goalie_combos[0]]['goalDifferentialConfidence']}%",
            # 'finalPeriod': f"{ai_data['prediction'][goalie_combos[0]]['finalPeriod']} - {ai_data['prediction'][goalie_combos[0]]['finalPeriodConfidence']}%",
            # 'pastRegulation': f"{ai_data['prediction'][goalie_combos[0]]['pastRegulation']} - {ai_data['prediction'][goalie_combos[0]]['pastRegulationConfidence']}%",
          },
          f'{goalie_combos[1]}': {
            'awayTeam': f"{ai_data['awayTeam']} - {ai_data['prediction'][goalie_combos[1]]['awayScore']} - {ai_data['prediction'][goalie_combos[1]]['awayScoreConfidence']}%",
            'homeTeam': f"{ai_data['homeTeam']} - {ai_data['prediction'][goalie_combos[1]]['homeScore']} - {ai_data['prediction'][goalie_combos[1]]['homeScoreConfidence']}%",
            'winningTeam': f"{ai_data['prediction'][goalie_combos[1]]['winningTeam']} - {ai_data['prediction'][goalie_combos[1]]['winnerConfidence']}%",
            'offset': ai_data['prediction'][goalie_combos[1]]['offset'],
            'totalGoals': f"{ai_data['prediction'][goalie_combos[1]]['totalGoals']} - {ai_data['prediction'][goalie_combos[1]]['totalGoalsConfidence']}%",
            'goalDifferential': f"{ai_data['prediction'][goalie_combos[1]]['goalDifferential']} - {ai_data['prediction'][goalie_combos[1]]['goalDifferentialConfidence']}%",
            # 'finalPeriod': f"{ai_data['prediction'][goalie_combos[1]]['finalPeriod']} - {ai_data['prediction'][goalie_combos[1]]['finalPeriodConfidence']}%",
            # 'pastRegulation': f"{ai_data['prediction'][goalie_combos[1]]['pastRegulation']} - {ai_data['prediction'][goalie_combos[1]]['pastRegulationConfidence']}%",
          },
          f'{goalie_combos[2]}': {
            'awayTeam': f"{ai_data['awayTeam']} - {ai_data['prediction'][goalie_combos[2]]['awayScore']} - {ai_data['prediction'][goalie_combos[2]]['awayScoreConfidence']}%",
            'homeTeam': f"{ai_data['homeTeam']} - {ai_data['prediction'][goalie_combos[2]]['homeScore']} - {ai_data['prediction'][goalie_combos[2]]['homeScoreConfidence']}%",
            'winningTeam': f"{ai_data['prediction'][goalie_combos[2]]['winningTeam']} - {ai_data['prediction'][goalie_combos[2]]['winnerConfidence']}%",
            'offset': ai_data['prediction'][goalie_combos[2]]['offset'],
            'totalGoals': f"{ai_data['prediction'][goalie_combos[2]]['totalGoals']} - {ai_data['prediction'][goalie_combos[2]]['totalGoalsConfidence']}%",
            'goalDifferential': f"{ai_data['prediction'][goalie_combos[2]]['goalDifferential']} - {ai_data['prediction'][goalie_combos[2]]['goalDifferentialConfidence']}%",
            # 'finalPeriod': f"{ai_data['prediction'][goalie_combos[2]]['finalPeriod']} - {ai_data['prediction'][goalie_combos[2]]['finalPeriodConfidence']}%",
            # 'pastRegulation': f"{ai_data['prediction'][goalie_combos[2]]['pastRegulation']} - {ai_data['prediction'][goalie_combos[2]]['pastRegulationConfidence']}%",
          },
          f'{goalie_combos[3]}': {
            'awayTeam': f"{ai_data['awayTeam']} - {ai_data['prediction'][goalie_combos[3]]['awayScore']} - {ai_data['prediction'][goalie_combos[3]]['awayScoreConfidence']}%",
            'homeTeam': f"{ai_data['homeTeam']} - {ai_data['prediction'][goalie_combos[3]]['homeScore']} - {ai_data['prediction'][goalie_combos[3]]['homeScoreConfidence']}%",
            'winningTeam': f"{ai_data['prediction'][goalie_combos[3]]['winningTeam']} - {ai_data['prediction'][goalie_combos[3]]['winnerConfidence']}%",
            'offset': ai_data['prediction'][goalie_combos[3]]['offset'],
            'totalGoals': f"{ai_data['prediction'][goalie_combos[3]]['totalGoals']} - {ai_data['prediction'][goalie_combos[3]]['totalGoalsConfidence']}%",
            'goalDifferential': f"{ai_data['prediction'][goalie_combos[3]]['goalDifferential']} - {ai_data['prediction'][goalie_combos[3]]['goalDifferentialConfidence']}%",
            # 'finalPeriod': f"{ai_data['prediction'][goalie_combos[3]]['finalPeriod']} - {ai_data['prediction'][goalie_combos[3]]['finalPeriodConfidence']}%",
            # 'pastRegulation': f"{ai_data['prediction'][goalie_combos[3]]['pastRegulation']} - {ai_data['prediction'][goalie_combos[3]]['pastRegulationConfidence']}%",
          },
        },
        'message': ai_data['message'],
      }
    else:
      live_data = {
        'away': ai_data['live']['away'],
        'home': ai_data['live']['home'],
        'leader': ai_data['live']['leader'],
        'period': ai_data['live']['period'],
      }
      simple_data = {
        'awayTeam': f"{ai_data['awayTeam']} - {ai_data['prediction']['awayScore']} - {ai_data['prediction']['awayScoreConfidence']}%",
        'homeTeam': f"{ai_data['homeTeam']} - {ai_data['prediction']['homeScore']} - {ai_data['prediction']['homeScoreConfidence']}%",
        'live': live_data,
        'winningTeam': f"{ai_data['prediction']['winningTeam']} - {ai_data['prediction']['winnerConfidence']}%",
        'message': ai_data['message'],
        'offset': ai_data['prediction']['offset'],
        'totalGoals': f"{ai_data['prediction']['totalGoals']} - {ai_data['prediction']['totalGoalsConfidence']}%",
        'goalDifferential': f"{ai_data['prediction']['goalDifferential']} - {ai_data['prediction']['goalDifferentialConfidence']}%",
        'finalPeriod': f"{ai_data['prediction']['finalPeriod']} - {ai_data['prediction']['finalPeriodConfidence']}%",
        'pastRegulation': f"{ai_data['prediction']['pastRegulation']} - {ai_data['prediction']['pastRegulationConfidence']}%",
      }
    games.append(simple_data)
  
  return jsonify(games)


def predict_week(db,**kwargs):
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

  games = {}
  for day in res['gameWeek']:
    games[day['dayAbbrev']] = []
    for game in day['games']:
      ai_data = ai(db, game, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])
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

  return ai(db, game_data, model_winner=kwargs['model_winner'],model_homeScore=kwargs['model_homeScore'],model_awayScore=kwargs['model_awayScore'],model_totalGoals=kwargs['model_totalGoals'],model_goalDifferential=kwargs['model_goalDifferential'])


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
  # CURRENT_SEASON = db["dev_seasons"].find_one(sort=[("seasonId", -1)])['seasonId']
  FINAL_SEASON = db["dev_training_records"].find_one({'version': VERSION})['finalSeason']
  used_training_data = load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{FINAL_SEASON}.joblib')
  latest_ids = latestIDs(used_training_data)
  return latest_ids


def save_boxscores(db,date,**kwargs):
  Games = db['dev_games']
  latest_ids = latestIDs()
  Odds = db['dev_odds']
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
          print('DUPLICATE - Games', game['id'])
          pass
      else:
        try:
          if false_chain(game,'homeTeam','odds') and false_chain(game,'awayTeam','odds'):
            for i in range(0, len(schedule['oddsPartners'])):
              if schedule['oddsPartners'][i]['country'].lower() == 'us':
                usPartnerId = schedule['oddsPartners'][i]['partnerId']
                usPartnerIndex = i
                for provider in game['homeTeam']['odds']:
                  if provider['providerId'] == usPartnerId:
                    usHomeOdds = provider['value']
                    break
                for provider in game['awayTeam']['odds']:
                  if provider['providerId'] == usPartnerId:
                    usAwayOdds = provider['value']
                    break
                break
            Odds.insert_one({
              'id': game['id'],
              'date': week['date'],
              'odds': {
                'homeTeam': usHomeOdds,
                'awayTeam': usAwayOdds,
                },
              'oddsPartner': schedule['oddsPartners'][usPartnerIndex]['name']
            })
        except DuplicateKeyError:
          print('DUPLICATE - Odds', game['id'])
          pass
  updated_ids = latestIDs()
  print(latest_ids)
  print(updated_ids)

  return {'res':{
    'previous': latest_ids,
    'updated': updated_ids
  }}