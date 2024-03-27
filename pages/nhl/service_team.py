import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\inputs')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api')
sys.path.append(r'C:\Users\patricklyden\Projects\Hockey\hockey-api\constants')

import requests
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from util.helpers import false_chain, latestIDs, adjusted_winner, test_recommended_wagers, safe_chain
from inputs.inputs import master_inputs
from pages.nhl.nhl_helpers_team import ai_teams
from util.team_models import PREDICT_SCORE_H2H, PREDICT_H2H, PREDICT_COVERS, PREDICT_SCORE_COVERS, PREDICT_LGBM_H2H, PREDICT_LGBM_SCORE_H2H
import warnings
import xgboost as xgb

# Suppress specific UserWarning from sklearn
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names")

def test_model_team(db,startID,endID,models,useModel, projectedLineup=False, projectedRoster=False):
  if useModel == 'xgb':
    wModels = models['wModels']
    lModels = models['lModels']
  elif useModel == 'lgbm':
    wModels = models['wModelsLGBM']
    lModels = None
  cModels = models['cModels']

  Boxscores = db['dev_boxscores']

  if startID == -1 or endID == -1:
    md = metadata(db)
    if startID == -1:
      startID = md['saved']['training']+1
    if endID == -1:
      endID = min([md['saved']['boxscore'],md['saved']['game']])

  boxscore_list = list(Boxscores.find(
    {'id': {'$gte':startID,'$lt':endID+1}}
  ))

  winnerB_results = []
  winner_covers_agreement = []
  winner_covers_disagreement = []
  winner_covers_daily_agreement = []
  winner_covers_daily_disagreement = []
  winnerB_correct_confidences = []
  winnerB_incorrect_confidences = []
  winnerB_daily_percents = []
  covers_results = []
  covers_correct_confidences = []
  covers_incorrect_confidences = []
  covers_daily_percents = []

  test_results = {}

  for boxscore in boxscore_list:
    test_results[boxscore['gameDate']] = {}
    test_results[boxscore['gameDate']] = {
      'games': [],
      'winnerBPercent': 0,
      'winnerB_line_results': [],
      'agrees': [],
      'disagrees': [],
      'agrees_false': [],
      'coversPercent': 0,
      'covers_line_results': [],
    }

  for boxscore in boxscore_list:

    gameId = boxscore['id']
    inputs = master_inputs(db,boxscore,isProjectedRoster=projectedRoster,isProjectedLineup=projectedLineup)
    inputs = inputs['data']

    if useModel == 'xgb':
      winnerB_prediction, winnerB_probability = PREDICT_SCORE_H2H([inputs],wModels,lModels,simple_return=True)
    elif useModel == 'lgbm':
      winnerB_prediction, winnerB_probability = PREDICT_LGBM_SCORE_H2H([inputs],wModels,simple_return=True)
    
    covers_prediction, covers_probability = PREDICT_SCORE_COVERS([inputs],cModels,simple_return=True)

    winnerB_true = inputs['winnerB']
    covers_true = inputs['covers']
    winnerB_calculation = 1 if winnerB_prediction[0] == winnerB_true else 0
    covers_calculation = 1 if covers_prediction[0] == covers_true else 0
    w_c_agree_true = 1 if winnerB_prediction[0] == covers_prediction[0] and winnerB_prediction[0] == winnerB_true and covers_prediction[0] == covers_true else 0
    w_c_agree_false = 1 if winnerB_prediction[0] != covers_prediction[0] and winnerB_prediction[0] != winnerB_true and covers_prediction[0] != covers_true else 0
    w_c_disagree = 1 if winnerB_prediction[0] != covers_prediction[0] else 0
    agreement = w_c_agree_true
    disagreement = w_c_disagree
    winnerB_results.append(winnerB_calculation)
    if winnerB_calculation == 1:
      winnerB_correct_confidences.append(round(winnerB_probability[0] * 100))
    else:
      winnerB_incorrect_confidences.append(round(winnerB_probability[0] * 100))
    covers_results.append(covers_calculation)
    if covers_calculation == 1:
      covers_correct_confidences.append(round(covers_probability[0] * 100))
    else:
      covers_incorrect_confidences.append(round(covers_probability[0] * 100))

    winner_covers_agreement.append(agreement)
    winner_covers_disagreement.append(disagreement)
    test_results[boxscore['gameDate']]['winnerB_line_results'].append(winnerB_calculation)
    test_results[boxscore['gameDate']]['covers_line_results'].append(covers_calculation)
    test_results[boxscore['gameDate']]['agrees'].append(agreement)
    test_results[boxscore['gameDate']]['disagrees'].append(disagreement)
    test_results[boxscore['gameDate']]['agrees_false'].append(w_c_agree_false)
    test_results[boxscore['gameDate']]['games'].append({
      'id': gameId,
      'home': inputs['homeTeam'],
      'away': inputs['awayTeam'],
      'homeScore': inputs['homeScore'],
      'awayScore': inputs['awayScore'],
      'covers': {
        'prediction': covers_prediction[0],
        'actual': covers_true,
        'calculation': 1 if covers_prediction[0] == covers_true else 0,
        'confidence': round(covers_probability[0] * 100),
      },
      'winnerB': {
        'prediction': winnerB_prediction[0],
        'actual': winnerB_true,
        'calculation': winnerB_calculation,
        'confidence': round(winnerB_probability[0] * 100),
      },
    })


  for date in test_results:
    winnerBPercent = (sum(test_results[date]['winnerB_line_results']) / len(test_results[date]['winnerB_line_results'])) * 100
    coversPercent = (sum(test_results[date]['covers_line_results']) / len(test_results[date]['covers_line_results'])) * 100
    test_results[date]['winnerBPercent'] = winnerBPercent
    test_results[date]['coversPercent'] = coversPercent
    winnerB_daily_percents.append((winnerBPercent,len(test_results[date]['winnerB_line_results'])))
    covers_daily_percents.append((coversPercent,len(test_results[date]['covers_line_results'])))
    agreement_percent = (sum(test_results[date]['agrees']) / len(test_results[date]['agrees'])) * 100
    disagreement_percent = (sum(test_results[date]['disagrees']) / len(test_results[date]['disagrees'])) * 100
    false_agreement_percent = (sum(test_results[date]['agrees_false']) / len(test_results[date]['agrees_false'])) * 100
    test_results[date]['agreement'] = agreement_percent
    test_results[date]['disagreement'] = disagreement_percent
    test_results[date]['agrees_false'] = false_agreement_percent
    winner_covers_daily_agreement.append((agreement_percent,len(test_results[date]['agrees'])))

  
  winnerB_accuracy_bins = {}
  for p in winnerB_daily_percents:
    if str(p[1]) in winnerB_accuracy_bins:
      winnerB_accuracy_bins[str(p[1])].append(p[0])
    else:
      winnerB_accuracy_bins[str(p[1])] = [p[0]]
  winnerB_accuracy_bin_averages = {}
  for b in winnerB_accuracy_bins:
    winnerB_accuracy_bin_averages[b] = sum(winnerB_accuracy_bins[b]) / len(winnerB_accuracy_bins[b])

  covers_accuracy_bins = {}
  for p in covers_daily_percents:
    if str(p[1]) in covers_accuracy_bins:
      covers_accuracy_bins[str(p[1])].append(p[0])
    else:
      covers_accuracy_bins[str(p[1])] = [p[0]]
  covers_accuracy_bin_averages = {}
  for b in covers_accuracy_bins:
    covers_accuracy_bin_averages[b] = sum(covers_accuracy_bins[b]) / len(covers_accuracy_bins[b])

  agreement_bins = {}
  for p in winner_covers_daily_agreement:
    if str(p[1]) in agreement_bins:
      agreement_bins[str(p[1])].append(p[0])
    else:
      agreement_bins[str(p[1])] = [p[0]]
  agreement_bin_averages = {}
  for b in agreement_bins:
    agreement_bin_averages[b] = sum(agreement_bins[b]) / len(agreement_bins[b])
  
  disagreement_bins = {}
  for p in winner_covers_daily_disagreement:
    if str(p[1]) in disagreement_bins:
      disagreement_bins[str(p[1])].append(p[0])
    else:
      disagreement_bins[str(p[1])] = [p[0]]
  disagreement_bin_averages = {}
  for b in disagreement_bins:
    disagreement_bin_averages[b] = sum(disagreement_bins[b]) / len(disagreement_bins[b])

  # test_results['WinnerB Results'] = winnerB_results
  # test_results['WinnerB Daily Percents'] = winnerB_daily_percents
  test_results['Daily WinnerB Accuracy Bins'] = winnerB_accuracy_bins
  test_results['Daily WinnerB Accuracy Bin Averages'] = winnerB_accuracy_bin_averages
  test_results['Daily Covers Accuracy Bins'] = covers_accuracy_bins
  test_results['Daily Covers Accuracy Bin Averages'] = covers_accuracy_bin_averages
  test_results['Daily Agreement Bins'] = agreement_bins
  test_results['Daily Agreement Bin Averages'] = agreement_bin_averages
  test_results['Daily Disagreement Bins'] = disagreement_bins
  test_results['Daily Disagreement Bin Averages'] = disagreement_bin_averages
  # test_results['WinnerB Correct Confidences'] = winnerB_correct_confidences
  # test_results['WinnerB Incorrect Confidences'] = winnerB_incorrect_confidences
  test_results['allWinnerBPercent'] = (sum(winnerB_results) / len(winnerB_results)) * 100
  test_results['allCoversPercent'] = (sum(covers_results) / len(covers_results)) * 100
  test_results['allAgreementPercent'] = (sum(winner_covers_agreement) / len(winner_covers_agreement)) * 100
  test_results['allDisagreementPercent'] = (sum(winner_covers_disagreement) / len(winner_covers_disagreement)) * 100
  return test_results

def predict_team_day(db, date, day, gamePick, projectedLineup, models, useModel, projectedRoster, vote):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  projectedLineups = [projectedLineup]*len(game_data['games'])
  projectedRosters = [projectedRoster]*len(game_data['games'])
  return ai_teams(db, game_data['games'], projectedLineups, models, useModel, projectedRosters, vote)

def predict_team_day_simple(db, date, day, gamePick, projectedLineup, models, useModel, projectedRoster, vote):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  projectedLineups = [projectedLineup]*len(game_data['games'])
  projectedRosters = [projectedRoster]*len(game_data['games'])
  return ai_teams(db, game_data['games'], projectedLineups, models, useModel, projectedRosters, simple=True)

def predict_team_day_receipt(db, date, day, gamePick, projectedLineup, models, useModel, projectedRoster, vote):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  projectedLineups = [projectedLineup]*len(game_data['games'])
  projectedRosters = [projectedRoster]*len(game_data['games'])
  return ai_teams(db, game_data['games'], projectedLineups, models, useModel, projectedRosters, receipt=True)


def save_boxscores(db,date,models=False):
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

def clean_boxscores(db):
  Boxscores = db['dev_boxscores']
  query = {"gameState": "FUT"}
  result = Boxscores.delete_many(query)
  return {"Documents deleted": result.deleted_count}