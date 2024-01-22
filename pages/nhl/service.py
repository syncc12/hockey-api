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
from util.helpers import false_chain, latestIDs, adjusted_winner, test_recommended_wagers, safe_chain
from inputs.inputs import master_inputs
from pages.nhl.nhl_helpers import ai, ai_return_dict
from constants.constants import VERSION, FILE_VERSION
from util.models import TEST_ALL_INIT, TEST_LINE_INIT, TEST_ALL_UPDATE, TEST_LINE_UPDATE, TEST_PREDICTION, TEST_CONFIDENCE, TEST_COMPARE, TEST_DATA, TEST_RESULTS, TEST_CONFIDENCE_RESULTS

def debug():
  res = requests.get("https://api-web.nhle.com/v1/schedule/2023-11-22").json()
  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]
  data = nhl_ai(game_data)
  return jsonify(data)


def test_model(db,startID,endID,show_data,wager,useProjectedLineup,models):
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
  all_total = TEST_ALL_INIT
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
    'game': [],
    'winnerPercent': 0,
    'winnerBPercent': 0,
    'homeScorePercent': 0,
    'awayScorePercent': 0,
    }

  for boxscore in boxscore_list:
    all_ids.append(boxscore['id'])
    awayId = boxscore['awayTeam']['id']
    homeId = boxscore['homeTeam']['id']
    test_data = nhl_test(db=db,boxscore=boxscore,useProjectedLineup=useProjectedLineup)
    test_prediction = TEST_PREDICTION(models,test_data)
    
    test_confidence = TEST_CONFIDENCE(models,test_data)
    
    predicted = TEST_COMPARE(test_prediction,awayId,homeId)

    test_data_result = TEST_DATA(test_data,awayId,homeId)

    test_results[boxscore['gameDate']]['game'].append({
      'id': boxscore['id'],
      'results': TEST_RESULTS(predicted,test_data_result),
      'confidence': TEST_CONFIDENCE_RESULTS(test_confidence),
    })
    test_results_len = len(test_results[boxscore['gameDate']]['game']) - 1
    if show_data != -1:
      test_results[boxscore['gameDate']]['game'][test_results_len]['data'] = test_data['input_data']
    game_odds = Odds.find_one({'id':boxscore['id']})
    winnings = 0
    winnings10 = 0
    winnings100 = 0
    returns = 0
    returns10 = 0
    returns100 = 0
    if game_odds:
      test_results[boxscore['gameDate']]['game'][test_results_len]['awayOdds'] = float(game_odds['odds']['awayTeam'])
      test_results[boxscore['gameDate']]['game'][test_results_len]['homeOdds'] = float(game_odds['odds']['homeTeam'])
      winning_odds = float(game_odds['odds']['awayTeam']) if test_data_result['test_winner'] == awayId else float(game_odds['odds']['homeTeam'])
      # winning_odds = float(game_odds['odds']['awayTeam']) if test_winner == awayId else float(game_odds['odds']['homeTeam'])
      if not boxscore['gameDate'] in odds_dict:
        odds_dict[boxscore['gameDate']] = []
      odds_dict[boxscore['gameDate']].append(winning_odds)
      if not boxscore['gameDate'] in confidence_dict:
        confidence_dict[boxscore['gameDate']] = []
      confidence_dict[boxscore['gameDate']].append(int((np.max(test_confidence['test_confidence_winner'], axis=1) * 100)[0]))
      
      if not boxscore['gameDate'] in winners_dict:
        winners_dict[boxscore['gameDate']] = []

      if false_chain(test_results,safe_chain(boxscore,'gameDate'),'results',test_results_len,'winner'):
        winnings = abs(((100/winning_odds)*wager) if winning_odds < 0 else ((winning_odds/100)*wager))
        winnings10 = abs(((100/winning_odds)*10) if winning_odds < 0 else ((winning_odds/100)*10))
        winnings100 = abs(((100/winning_odds)*100) if winning_odds < 0 else ((winning_odds/100)*100))
        returns = winnings + wager
        returns10 = winnings10 + 10
        returns100 = winnings100 + 100
        winners_dict[boxscore['gameDate']].append(1)
      else:
        winners_dict[boxscore['gameDate']].append(0)
      
      test_results[boxscore['gameDate']]['game'][test_results_len]['betting'] = {
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
    line_total = TEST_LINE_INIT
    winnings_total = 0
    winnings10_total = 0
    winnings100_total = 0
    returns_total = 0
    returns10_total = 0
    returns100_total = 0
    list_total = len(test_results[boxscore['gameDate']]['game'])
    all_list_total += list_total
    for r in test_results[boxscore['gameDate']]['game']:
      line_total = TEST_LINE_UPDATE(line_total,r)
      # if game_odds:
      #   winnings_total += r['betting']['winnings']['wager']
      #   winnings10_total += r['betting']['winnings']['10']
      #   winnings100_total += r['betting']['winnings']['100']
      #   returns_total += r['betting']['returns']['wager']
      #   returns10_total += r['betting']['returns']['10']
      #   returns100_total += r['betting']['returns']['100']
      # if home_score_total == 1 and away_score_total == 1:
      #   home_away_score_total += 1

    all_total = TEST_ALL_UPDATE(all_total,line_total)
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
    test_results[boxscore['gameDate']]['winnerPercent'] = (line_total['winner_total'] / list_total) * 100
    test_results[boxscore['gameDate']]['winnerBPercent'] = (line_total['winnerB_total'] / list_total) * 100
    test_results[boxscore['gameDate']]['homeScorePercent'] = (line_total['homeScore_total'] / list_total) * 100
    test_results[boxscore['gameDate']]['awayScorePercent'] = (line_total['awayScore_total'] / list_total) * 100
    # test_results[boxscore['gameDate']]['h2hScorePercent'] = (line_total['home_away_score_total'] / list_total) * 100
    test_results[boxscore['gameDate']]['goalTotalPercent'] = (line_total['totalGoals_total'] / list_total) * 100
    test_results[boxscore['gameDate']]['goalDifferentialPercent'] = (line_total['goalDifferential_total'] / list_total) * 100
    test_results[boxscore['gameDate']]['totalGames'] = list_total
  
  for day in list(odds_dict.keys()):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(day)
    test_recommended_wagers(100,odds=odds_dict[day],confidence=confidence_dict[day],winners=winners_dict[day])

  test_results['allWinnerPercent'] = (all_total['all_winner_total'] / all_list_total) * 100
  test_results['allWinnerBPercent'] = (all_total['all_winnerB_total'] / all_list_total) * 100
  test_results['allHomeScorePercent'] = (all_total['all_homeScore_total'] / all_list_total) * 100
  test_results['allAwayScorePercent'] = (all_total['all_awayScore_total'] / all_list_total) * 100
  # test_results['allH2HScorePercent'] = (all_total['all_home_away_score_total'] / all_list_total) * 100
  test_results['allGoalTotalPercent'] = (all_total['all_totalGoals_total'] / all_list_total) * 100
  test_results['allGoalDifferentialPercent'] = (all_total['all_goalDifferential_total'] / all_list_total) * 100
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


def collect_boxscores(db,startID,endID,models):
  Boxscores = db['dev_boxscores']

  boxscores = []
  for id in range(startID, endID+1):
    boxscore_data = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{id}/boxscore").json()
    boxscores.append(boxscore_data)
  Boxscores.insert_many(boxscores)
  return {'status':'done'}

  
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


def predict(db,day,game,date,models):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(db, game_data, model_winner=models['model_winner'],model_homeScore=models['model_homeScore'],model_awayScore=models['model_awayScore'],model_totalGoals=models['model_totalGoals'],model_goalDifferential=models['model_goalDifferential'],model_finalPeriod=models['model_finalPeriod'],model_pastRegulation=models['model_pastRegulation'],model_awayShots=models['model_awayShots'],model_homeShots=models['model_homeShots'],model_awayShotsPeriod1=models['model_awayShotsPeriod1'],model_homeShotsPeriod1=models['model_homeShotsPeriod1'],model_awayShotsPeriod2=models['model_awayShotsPeriod2'],model_homeShotsPeriod2=models['model_homeShotsPeriod2'],model_awayShotsPeriod3=models['model_awayShotsPeriod3'],model_homeShotsPeriod3=models['model_homeShotsPeriod3'],model_awayShotsPeriod4=models['model_awayShotsPeriod4'],model_homeShotsPeriod4=models['model_homeShotsPeriod4'],model_awayShotsPeriod5=models['model_awayShotsPeriod5'],model_homeShotsPeriod5=models['model_homeShotsPeriod5'],model_awayScorePeriod1=models['model_awayScorePeriod1'],model_homeScorePeriod1=models['model_homeScorePeriod1'],model_awayScorePeriod2=models['model_awayScorePeriod2'],model_homeScorePeriod2=models['model_homeScorePeriod2'],model_awayScorePeriod3=models['model_awayScorePeriod3'],model_homeScorePeriod3=models['model_homeScorePeriod3'],model_awayScorePeriod4=models['model_awayScorePeriod4'],model_homeScorePeriod4=models['model_homeScorePeriod4'],model_awayScorePeriod5=models['model_awayScorePeriod5'],model_homeScorePeriod5=models['model_homeScorePeriod5'],model_period1PuckLine=models['model_period1PuckLine'],model_period2PuckLine=models['model_period2PuckLine'],model_period3PuckLine=models['model_period3PuckLine'])


def predict_day(db,date,day,projectedLineup,models):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  game_data = res['gameWeek'][day-1]
  games = []
  for game in game_data['games']:
    ai_data = ai(db, game, projectedLineup, models)
    games.append(ai_data)
  return games


def predict_day_simple(db,date,day,gamePick,projectedLineup,models):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  game_data = res['gameWeek'][day-1]
  if gamePick > 0:
    game_data['games'] = [game_data['games'][gamePick-1]]
  games = []
  for game in game_data['games']:
    ai_data = ai(db, game, projectedLineup, models)
    if ai_data['message'] == 'using projected lineup':
      goalie_combos = list(ai_data['prediction'].keys())
      simple_data = {
        'prediction': {},
        'message': ai_data['message'],
      }
      for combo in goalie_combos:
        simple_data['prediction'][combo] = {}
        simple_data['prediction'][combo]['awayTeam'] = f"{ai_data['awayTeam']} - {ai_data['prediction'][combo]['prediction_awayScore']} - {ai_data['confidence'][combo]['confidence_awayScore']}%"
        simple_data['prediction'][combo]['homeTeam'] = f"{ai_data['homeTeam']} - {ai_data['prediction'][combo]['prediction_homeScore']} - {ai_data['confidence'][combo]['confidence_homeScore']}%"
        simple_data['prediction'][combo]['winningTeam'] = f"{ai_data['prediction'][combo]['winner']} - {ai_data['confidence'][combo]['confidence_winner']}%"
        simple_data['prediction'][combo]['winningTeamB'] = f"{ai_data['prediction'][combo]['winnerB']} - {ai_data['confidence'][combo]['confidence_winnerB']}%"
        simple_data['prediction'][combo]['offset'] = ai_data['prediction'][combo]['offset']
        simple_data['prediction'][combo]['totalGoals'] = f"{ai_data['prediction'][combo]['prediction_totalGoals']} - {ai_data['confidence'][combo]['confidence_totalGoals']}%"
        simple_data['prediction'][combo]['goalDifferential'] = f"{ai_data['prediction'][combo]['prediction_goalDifferential']} - {ai_data['confidence'][combo]['confidence_goalDifferential']}%"
    else:
      live_data = {
        'away': ai_data['live']['away'],
        'home': ai_data['live']['home'],
        'leader': ai_data['live']['leader'],
        'period': ai_data['live']['period'],
      }
      simple_data = {
        'awayTeam': f"{ai_data['awayTeam']} - {ai_data['prediction']['prediction_awayScore'][0]} - {ai_data['confidence']['confidence_awayScore']}%",
        'homeTeam': f"{ai_data['homeTeam']} - {ai_data['prediction']['prediction_homeScore'][0]} - {ai_data['confidence']['confidence_homeScore']}%",
        'live': live_data,
        'winningTeam': f"{ai_data['prediction']['winner']} - {ai_data['confidence']['confidence_winner']}%",
        'winningTeamB': f"{ai_data['prediction']['winnerB']} - {ai_data['confidence']['confidence_winnerB']}%",
        'message': ai_data['message'],
        'offset': ai_data['prediction']['offset'],
        'totalGoals': f"{ai_data['prediction']['prediction_totalGoals'][0]} - {ai_data['confidence']['confidence_totalGoals']}%",
        'goalDifferential': f"{ai_data['prediction']['prediction_goalDifferential'][0]} - {ai_data['confidence']['confidence_goalDifferential']}%",
      }
    games.append(simple_data)
  
  return jsonify(games)


def predict_week(db,models):
  res = requests.get("https://api-web.nhle.com/v1/schedule/now").json()

  games = {}
  for day in res['gameWeek']:
    games[day['dayAbbrev']] = []
    for game in day['games']:
      ai_data = ai(db, game, model_winner=models['model_winner'],model_homeScore=models['model_homeScore'],model_awayScore=models['model_awayScore'],model_totalGoals=models['model_totalGoals'],model_goalDifferential=models['model_goalDifferential'])
      games[day['dayAbbrev']].append(ai_data)

  return jsonify(games)


def get_day_ids(db,date,models):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()
  ids = []
  for game in res['gameWeek'][0]['games']:
    ids.append(game['id'])
  
  return {'ids':ids}


def date_predict(db,date,models):
  res = requests.get(f"https://api-web.nhle.com/v1/schedule/{date}").json()

  day = request.args.get('day', default=1, type=int)
  game = request.args.get('game', default=1, type=int)
  game_data = res['gameWeek'][int(day)-1]['games'][int(game)-1]

  return ai(db, game_data, model_winner=models['model_winner'],model_homeScore=models['model_homeScore'],model_awayScore=models['model_awayScore'],model_totalGoals=models['model_totalGoals'],model_goalDifferential=models['model_goalDifferential'])


def now(db,models):
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


def game_date(db,date,models):
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


def metadata(db):
  # CURRENT_SEASON = db["dev_seasons"].find_one(sort=[("seasonId", -1)])['seasonId']
  FINAL_SEASON = db["dev_training_records"].find_one({'version': VERSION})['finalSeason']
  used_training_data = load(f'training_data/v{VERSION}/training_data_v{FILE_VERSION}_{FINAL_SEASON}.joblib')
  latest_ids = latestIDs(used_training_data)
  return latest_ids


def save_boxscores(db,date,models):
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