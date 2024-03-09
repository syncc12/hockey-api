import math
import requests
from pymongo import MongoClient
from datetime import datetime
import numpy as np
from itertools import combinations

REPLACE_VALUE = -1


def nhl_api():
  data = requests.get('localhost:8001/nhl/boxscores').json()
  print(data)

def nhl_db():
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
  Trainings = db["dev_trainings"]
  training_data = Trainings.find({})
  for data in training_data:
    print(data)

def n2n(in_name):
  try:
    if in_name and not isNaN(in_name):
      char_to_num = {'r':1,'t':2,'n':3,'s':4,'l':5,'c':6,'d':7,'p':8,'m':9,
                     'h':10,'g':11,'b':12,'f':13,'y':14,'w':15,'k':16,'j':17}
      # char_to_num = {'R':1,'T':1,'N':2,'S':2,'L':3,'C':3,'D':4,'P':4,'M':5,
      #                'H':5,'G':6,'B':6,'F':7,'Y':7,'W':8,'K':8,'J':9}
      nums = []
      for char in str(in_name).lower().replace(' ', ''):
        if char in char_to_num.keys():
          nums.append(char_to_num[char])
      # return int(''.join(str(char_to_num[char]) for char in in_name.lower().replace(' ', '')))
      return int(''.join(str(char) for char in nums))
    else:
      return REPLACE_VALUE
  except Exception as error:
    return REPLACE_VALUE

def b2n(in_bool):
  if type(in_bool) == bool:
    if in_bool:
      return 1
    else:
      return 0
  else:
    return in_bool

def isNaN(inCheck):
  try:
    float_value = float(inCheck)
    return math.isnan(float_value)
  except (ValueError, TypeError):
    return False
  
def formatTime(inTime):
  if (type(inTime) == str):
    st = inTime.split(':')
    return float(int(st[0]) * (int(st[1])/60))
  else:
    return inTime

def formatDatetime(inDatetime):
  is_negative = False
  if inDatetime[0] == '-':
    is_negative = True
    inDatetime = inDatetime[1:]
  dt = datetime.strptime(inDatetime, '%Y-%m-%dT%H:%M:%SZ')
  outTime = float(int(dt.hour) * ((int(dt.minute) * (int(dt.second)/60))/60))
  if is_negative:
    outTime = -1 * outTime
  return outTime

def formatDate(inDate):
  if (type(inDate) == str):
    splitDate = inDate.split('-')
    return int(f'{splitDate[0]}{splitDate[1]}{splitDate[2]}')
  else:
    return inDate

def getPlayerData(players, player_id):
  if not isNaN(players) and not isNaN(player_id):
    return next((player for player in players if player['playerId'] == player_id), REPLACE_VALUE)
  else:
    return REPLACE_VALUE

def safe_chain(obj, *keys, default=REPLACE_VALUE):
  for key in keys:
    if key == default:
      return default
    else:
      if type(key) == int:
        if len(obj) > key:
          obj = obj[key]
        else:
          return default
      else:
        try:
          obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
        except (KeyError, TypeError, AttributeError):
          return default
  return obj

def safe_none(inValue,replaceValue=-1):
  if inValue is None:
    return replaceValue
  else:
    return inValue

def false_chain(obj, *keys, default=REPLACE_VALUE):
  for key in keys:
    if key == default:
      return False
    else:
      if type(key) == int:
        if len(obj) > key:
          obj = obj[key]
        else:
          return False
      else:
        try:
          obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
        except (KeyError, TypeError, AttributeError):
          return False
  return True

def getPlayer(allPlayers,playerId):
  if not isNaN(allPlayers) and not isNaN(playerId):
    return [p for p in allPlayers if p['playerId'] == playerId]
  else:
    return REPLACE_VALUE

def lastGameId(team,gameId):
  TEAM_SEASON_SCHEDULE = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/now'
  data = requests.get(TEAM_SEASON_SCHEDULE).json()
  gameIDs = [game['id'] for game in data['games'] if game['id'] < gameId]
  return min(gameIDs, key=lambda x:abs(x-gameId))

def lastGame(team,gameId):
  TEAM_SEASON_SCHEDULE = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/now'
  data = requests.get(TEAM_SEASON_SCHEDULE).json()
  gameIDs = [game['id'] for game in data['games'] if game['id'] < gameId]
  last_game = min(gameIDs, key=lambda x:abs(x-gameId))
  LAST_GAME_BOXSCORE = f'https://api-web.nhle.com/v1/gamecenter/{last_game}/boxscore'
  return requests.get(LAST_GAME_BOXSCORE).json()

def projectedLineup(team,gameId,last_boxscore=False):
  TEAM_SEASON_SCHEDULE = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/now'
  data = requests.get(TEAM_SEASON_SCHEDULE).json()
  gameIDs = [game['id'] for game in data['games'] if game['id'] < gameId]
  last_game = min(gameIDs, key=lambda x:abs(x-gameId))
  LAST_GAME_BOXSCORE = f'https://api-web.nhle.com/v1/gamecenter/{last_game}/boxscore'
  boxscore = requests.get(LAST_GAME_BOXSCORE).json()
  home_or_away = ''
  if boxscore['awayTeam']['abbrev'] == team:
    home_or_away = 'awayTeam'
  elif boxscore['homeTeam']['abbrev'] == team:
    home_or_away = 'homeTeam'
  if last_boxscore:
    return boxscore, home_or_away
  else:
    return last_game, home_or_away

def latestIDs(training_data=-1):
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client["hockey"]
  # Trainings = db["dev_trainings"]
  Boxscores = db["dev_boxscores"]
  Games = db["dev_games"]
  Seasons = db["dev_seasons"]

  season_list = list(Seasons.find(
    {},
    {'_id': 0, 'seasonId': 1}
  ))

  current_season = max([int(e['seasonId']) for e in season_list])
  # trainings_list = list(Trainings.find(
  #   {'season':current_season},
  #   {'_id': 0, 'id': 1}
  # ))
  boxscores_list = list(Boxscores.find(
    {'season':current_season},
    {'_id': 0, 'id': 1}
  ))
  games_list = list(Games.find(
    {'season':current_season},
    {'_id': 0, 'id': 1}
  ))
  
  schedule_now = requests.get(f"https://api-web.nhle.com/v1/schedule/now").json()
  latest_final_game = ''
  for week in schedule_now['gameWeek']:
    for game in week['games']:
      if latest_final_game == '':
        if 'gameOutcome' in game:
          latest_final_game = game['id']
        else:
          latest_final_game = game['id'] - 1
      else:
        if 'gameOutcome' in game:
          if game['id'] > latest_final_game:
            latest_final_game = game['id']

  # last_training_id = max([int(e['id']) for e in trainings_list])
  if training_data != -1:
    last_training_id = max(training_data, key=lambda x:x['id'])['id']
  else:
    last_training_id = -1
  last_boxscore_id = max([int(e['id']) for e in boxscores_list])
  last_game_id = max([int(e['id']) for e in games_list])

  return {
    'saved': {
      'boxscore': last_boxscore_id,
      'game': last_game_id,
      'training': last_training_id,
    },
    'live': {
      'game': latest_final_game,
      'boxscore': latest_final_game,
    },
  }

def adjusted_winner(awayId,homeId,winnerId):
  away_offset = abs(awayId - winnerId)
  home_offset = abs(homeId - winnerId)
  if away_offset < home_offset:
    return awayId
  elif home_offset < away_offset:
    return homeId
  else:
    return winnerId

def pad_list(lst, length, value):
  return lst + [value] * (length - len(lst))

def add_players(db,playerIds):
  Players = db['dev_players']
  player_list = []
  for id in playerIds:
    res = requests.get(f"https://api-web.nhle.com/v1/player/{id}/landing").json()
    player_list.append({
      'playerId': res['playerId'],
      'firstName': res['firstName']['default'],
      'lastName': res['lastName']['default'],
      'sweaterNumber': res['sweaterNumber'],
      'heightInInches': res['heightInInches'],
      'weightInPounds': res['weightInPounds'],
      'birthDate': res['birthDate'],
      'shootsCatches': res['shootsCatches'],
      'playerSlug': res['playerSlug'],
    })
  Players.insert_many(player_list)
  return player_list

def add_player(db,playerId):
  Players = db['dev_players']
  res = requests.get(f"https://api-web.nhle.com/v1/player/{playerId}/landing").json()
  player_data = {
    'playerId': res['playerId'],
    'firstName': res['firstName']['default'],
    'lastName': res['lastName']['default'],
    'sweaterNumber': safe_chain(res,'sweaterNumber'),
    'heightInInches': res['heightInInches'],
    'weightInPounds': res['weightInPounds'],
    'birthDate': res['birthDate'],
    'shootsCatches': res['shootsCatches'],
    'playerSlug': res['playerSlug'],
  }
  Players.insert_one(player_data)
  return player_data

def collect_players(db,allPlayerIds):
  if len(allPlayerIds) > 0:
    Players = db['dev_players']
    allPlayers = list(Players.find(
      {'$or': allPlayerIds},
      {'_id': 0, 'playerId': 1, 'birthDate': 1, 'shootsCatches': 1, 'weightInPounds': 1, 'heightInInches': 1}
    ))
    playerIds = [p['playerId'] for p in allPlayers]
    for playerId in allPlayerIds:
      if not playerId['playerId'] in playerIds:
        player_data = add_player(db,playerId['playerId'])
        allPlayers.append({
          'playerId': player_data['playerId'],
          'birthDate': player_data['birthDate'],
          'shootsCatches': player_data['shootsCatches'],
          'weightInPounds': player_data['weightInPounds'],
          'heightInInches': player_data['heightInInches'],
        })
    return allPlayers
  else:
    return []

def recommended_wagers(totalWager,gameData,isProjectedLineup=False):
  if not isinstance(gameData,list):
    gameData = [gameData]
  
  if not isProjectedLineup:
    winner_odds = []
    for g in gameData:
      winning_team = 'home' if g['prediction']['winnerId'] == g['homeId'] else 'away'
      win_odds = g[f'{winning_team}Odds']
      win_odds = (100/(win_odds+100)) if win_odds >= 0 else (abs(win_odds)/(abs(win_odds)+100))
      winner_odds.append(win_odds)
    winner_confidence = [(g['prediction']['winnerConfidence'])/100 for g in gameData]
    # print('winner_odds',winner_odds)
    # print('winner_confidence',winner_confidence)

def test_recommended_wagers(totalWager,odds,confidence,winners=[]):
  confidence_boost = (100 - np.max(confidence)) / 100
  # confidence_boost = 0
  odds_weight = 1
  confidence_weight = 1
  odds = [(100/(o+100)) if o >= 0 else (abs(o)/(abs(o)+100)) for o in odds]
  confidence = [(c/100)+confidence_boost for c in confidence]
  wager_per_game = totalWager / len(odds)
  wagers = []
  winnings = []
  wins = []
  for i in range(0, len(odds)):
    game_wager = wager_per_game * (odds[i] * odds_weight) * (confidence[i] * confidence_weight)
    # game_wager = wager_per_game * (confidence[i] * confidence_weight)
    wagers.append(game_wager)
  
  remainingWagers = totalWager - sum(wagers)
  third = math.floor(len(odds) / 3)

  sorted_confidence = sorted(list(enumerate(confidence)), key=lambda i: i[1], reverse=True)
  top_third = [i for i, e in sorted_confidence[:third]]

  third_remainingWagers = remainingWagers / 3
  spread_remainingWagers = remainingWagers / len(wagers)
  # for i in top_third:
  for i in range(0, len(wagers)):
    # wagers[i] + third_remainingWagers
    wagers[i] + spread_remainingWagers

  for i in range(0, len(odds)):
    game_winnings = game_wager + (game_wager * odds[i])
    winnings.append(game_winnings)
    if winners[i] == 1:
      wins.append(game_winnings)


  totalWinnings = sum(wins) - sum(wagers)
  print('confidence_boost',confidence_boost)
  print('odds',odds)
  print('confidence',confidence)
  print('wagers',wagers)
  print('winnings',winnings)
  print('totalWinnings',totalWinnings)

def getGamesPlayed(db,playerId,gameId=-1,position=-1):
  Boxscores = db['dev_boxscores']
  if position == -1:
    query = {
      '$or':[
        {'boxscore.playerByGameStats.awayTeam.forwards.playerId': playerId},
        {'boxscore.playerByGameStats.awayTeam.defense.playerId': playerId},
        {'boxscore.playerByGameStats.awayTeam.goalies.playerId': playerId},
        {'boxscore.playerByGameStats.homeTeam.forwards.playerId': playerId},
        {'boxscore.playerByGameStats.homeTeam.defense.playerId': playerId},
        {'boxscore.playerByGameStats.homeTeam.goalies.playerId': playerId},
      ]
    }
  else:
    query = {
      '$or':[
        {f'boxscore.playerByGameStats.awayTeam.{position}.playerId': playerId},
        {f'boxscore.playerByGameStats.homeTeam.{position}.playerId': playerId},
      ]
    }

  if gameId != -1:
    query['id'] = {'$lt': gameId, '$gte': int(f'{int(str(gameId)[0:4])-30}000000')}
  return Boxscores.count_documents(query)

def getAllGamesPlayed(db,playerIds=[],gameId=-1,position=-1):
  if len(playerIds) > 0:
    Boxscores = db['dev_boxscores']
    player_counts = {}

    
    if position == -1:
      for playerId in playerIds:
        query = {
          '$or':[
            {'boxscore.playerByGameStats.awayTeam.forwards.playerId': playerId},
            {'boxscore.playerByGameStats.awayTeam.defense.playerId': playerId},
            {'boxscore.playerByGameStats.awayTeam.goalies.playerId': playerId},
            {'boxscore.playerByGameStats.homeTeam.forwards.playerId': playerId},
            {'boxscore.playerByGameStats.homeTeam.defense.playerId': playerId},
            {'boxscore.playerByGameStats.homeTeam.goalies.playerId': playerId},
          ]
        }
        if gameId != -1:
          query['id'] = {'$lt': gameId}
        count = Boxscores.count_documents(query)
        player_counts[playerId] = count
        
    else:
      for playerId in playerIds:
        # query = {
        #   '$or':[
        #     {f'boxscore.playerByGameStats.awayTeam.{position}': {'$elemMatch': {'playerId': playerId}}},
        #     {f'boxscore.playerByGameStats.homeTeam.{position}': {'$elemMatch': {'playerId': playerId}}},
        #   ]
        # }
        query = {
          '$or':[
            {f'boxscore.playerByGameStats.awayTeam.{position}.playerId': playerId},
            {f'boxscore.playerByGameStats.homeTeam.{position}.playerId': playerId},
          ]
        }
        if gameId != -1:
          query['id'] = {'$lt': gameId, '$gte': int(f'{int(str(gameId)[0:4])-30}000000')}

        # explaination = Boxscores.find(query).explain()
        # print(explaination)
        count = Boxscores.count_documents(query)
        player_counts[playerId] = count

    return player_counts
  else:
    return -1

def getTotalGamesPlayed(db,playerIds=[],gameId=-1):
  if len(playerIds) > 0:
    Boxscores = db['dev_boxscores']
    player_query = []
    for playerId in playerIds:
      player_query.append({'boxscore.playerByGameStats.awayTeam.forwards': {'$elemMatch': {'playerId': playerId}}})
      player_query.append({'boxscore.playerByGameStats.awayTeam.defense': {'$elemMatch': {'playerId': playerId}}})
      player_query.append({'boxscore.playerByGameStats.awayTeam.goalies': {'$elemMatch': {'playerId': playerId}}})
      player_query.append({'boxscore.playerByGameStats.homeTeam.forwards': {'$elemMatch': {'playerId': playerId}}})
      player_query.append({'boxscore.playerByGameStats.homeTeam.defense': {'$elemMatch': {'playerId': playerId}}})
      player_query.append({'boxscore.playerByGameStats.homeTeam.goalies': {'$elemMatch': {'playerId': playerId}}})

    if gameId == -1:
      query = {
        '$or': player_query
      }
    else:
      query = {
        '$or': player_query,
        'id': {'$lt': gameId},
      }
    return Boxscores.count_documents(query)
  else:
    return -1

def getAge(player, game_date):
  try:
    if isNaN(player) or isNaN(game_date):
      return REPLACE_VALUE
    birthday = player[0]['birthDate']
    b_year, b_month, b_day = map(int, birthday.split('-'))
    g_year, g_month, g_day = map(int, game_date.split('-'))
    age = g_year - b_year
    if (b_month > g_month) or (b_month == g_month and b_day > g_day):
      age -= 1

    return age

  except Exception as error:
    return REPLACE_VALUE

def getAllAges(db,playerIds=[],gameDate=-1):
  if len(playerIds) > 0:
    Players = db['dev_players']
    player_ages = {}
    ids = []
    allPlayers = Players.find({
      'playerId': {'$in': playerIds}
    })
    for player in allPlayers:
      age = getAge(player=player,game_date=gameDate)
      player_ages[player['playerId']] = age
    return player_ages
  else:
    return REPLACE_VALUE
  
def getPlayerStats(db,playerId,season,gameId,position):
  Boxscores = db['dev_boxscores']

  position_dict = {
    'forward': 'forwards',
    'forwards': 'forwards',
    'defenseman': 'defense',
    'defense': 'defense',
    'goalie': 'goalies',
    'goalies': 'goalies',
  }

  stats = {
    'goals': 0,
    'assists': 0,
    'points': 0,
    'plusMinus': 0,
    'pim': 0,
    'hits': 0,
    'blockedShots': 0,
    'powerPlayGoals': 0,
    'powerPlayPoints': 0,
    'shorthandedGoals': 0,
    'shorthandedPoints': 0,
    'shots': 0,
    'savePercentage': 0.0,
    'evenStrengthGoalsAgainst': 0,
    'powerPlayGoalsAgainst': 0,
    'goalsAgainst': 0,
  }

  query = {
    '$or':[
      {f'boxscore.playerByGameStats.awayTeam.{position_dict[position]}.playerId': playerId},
      {f'boxscore.playerByGameStats.homeTeam.{position_dict[position]}.playerId': playerId},
    ],
    'id': {'$lt': gameId},
    'season': season,
  }
  
  statPipeline = [
    {
      '$match': query
    },
    {
      '$project': {
        '_id': 0,
        f'away{position_dict[position].capitalize()}': {'$filter': {'input': f'$boxscore.playerByGameStats.awayTeam.{position_dict[position]}','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
        f'home{position_dict[position].capitalize()}': {'$filter': {'input': f'$boxscore.playerByGameStats.homeTeam.{position_dict[position]}','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      }
    }
  ]
  games = list(Boxscores.aggregate(statPipeline))
  for game in games:
    if len(game[f'away{position_dict[position].capitalize()}']) > 0:
      data = game[f'away{position_dict[position].capitalize()}'][0]
    elif len(game[f'home{position_dict[position].capitalize()}']) > 0:
      data = game[f'home{position_dict[position].capitalize()}'][0]

    if position_dict[position] == 'goalies':
      if data['savePercentage']:
        savePercentage = float(data['savePercentage'])
      else:
        savePercentage = 0.0
      stats['savePercentage'] = stats['savePercentage'] + savePercentage
      stats['evenStrengthGoalsAgainst'] = stats['evenStrengthGoalsAgainst'] + data['evenStrengthGoalsAgainst']
      stats['powerPlayGoalsAgainst'] = stats['powerPlayGoalsAgainst'] + data['powerPlayGoalsAgainst']
      stats['goalsAgainst'] = stats['goalsAgainst'] + data['goalsAgainst']
      stats['pim'] = stats['pim'] + data['pim']
    else:
      stats['goals'] = stats['goals'] + data['goals']
      stats['assists'] = stats['assists'] + data['assists']
      stats['points'] = stats['points'] + data['points']
      stats['plusMinus'] = stats['plusMinus'] + data['plusMinus']
      stats['pim'] = stats['pim'] + data['pim']
      stats['hits'] = stats['hits'] + data['hits']
      stats['blockedShots'] = stats['blockedShots'] + data['blockedShots']
      stats['powerPlayGoals'] = stats['powerPlayGoals'] + data['powerPlayGoals']
      stats['powerPlayPoints'] = stats['powerPlayPoints'] + data['powerPlayPoints']
      stats['shorthandedGoals'] = stats['shorthandedGoals'] + data['shorthandedGoals']
      stats['shorthandedPoints'] = stats['shorthandedPoints'] + data['shPoints']
      stats['shots'] = stats['shots'] + data['shots']
  
  return stats

def projected_roster(gameId):
  landing_url = f'https://api-web.nhle.com/v1_1/gamecenter/{gameId}/landing'
  landing = requests.get(landing_url).json()
  if false_chain(landing,'matchup'):
    # print(landing)
    awayId = landing['awayTeam']['id']
    homeId = landing['homeTeam']['id']
    af = []
    ad = []
    ag = []
    hf = []
    hd = []
    hg = []

    for skater in landing['matchup']['skaterSeasonStats']:
      skater_data = {'playerId':skater['playerId'], 'gamesPlayed':safe_chain(skater,'gamesPlayed',default=0)}
      if skater['teamId'] == awayId:
        if skater['position'].lower() == 'd':
          ad.append(skater_data)
        else:
          af.append(skater_data)
      elif skater['teamId'] == homeId:
        if skater['position'].lower() == 'd':
          hd.append(skater_data)
        else:
          hf.append(skater_data)
    
    for goalie in landing['matchup']['goalieSeasonStats']:
      goalie_data = {'playerId':goalie['playerId'], 'gamesPlayed':safe_chain(goalie,'gamesPlayed',default=0)}
      if goalie['teamId'] == awayId:
        ag.append(goalie_data)
      elif goalie['teamId'] == homeId:
        hg.append(goalie_data)
    
    af = sorted(af, key=lambda d: d['gamesPlayed'], reverse=True)
    ad = sorted(ad, key=lambda d: d['gamesPlayed'], reverse=True)
    ag = sorted(ag, key=lambda d: d['gamesPlayed'], reverse=True)
    hf = sorted(hf, key=lambda d: d['gamesPlayed'], reverse=True)
    hd = sorted(hd, key=lambda d: d['gamesPlayed'], reverse=True)
    hg = sorted(hg, key=lambda d: d['gamesPlayed'], reverse=True)

    af_final = []
    ad_final = []
    ag_final = []
    hf_final = []
    hd_final = []
    hg_final = []
    af_len = 13 if len(af) >= 13 else len(af)
    ad_len = 6 if len(ad) >= 6 else len(ad)
    ag_len = 2 if len(ag) >= 2 else len(ag)
    hf_len = 13 if len(hf) >= 13 else len(hf)
    hd_len = 6 if len(hd) >= 6 else len(hd)
    hg_len = 2 if len(hf) >= 2 else len(hf)
    for i in range(0,af_len):
      af_final.append(af[i]['playerId'])
    for i in range(0,hf_len):
      hf_final.append(hf[i]['playerId'])
    for i in range(0,ad_len):
      ad_final.append(ad[i]['playerId'])
    for i in range(0,hd_len):
      hd_final.append(hd[i]['playerId'])
    for i in range(0,ag_len):
      ag_final.append(ag[i]['playerId'])
    for i in range(0,hg_len):
      hg_final.append(hg[i]['playerId'])
    awayRoster = {
      'forwards': af_final,
      'defense': ad_final,
      'goalies': ag_final,
    }
    homeRoster = {
      'forwards': hf_final,
      'defense': hd_final,
      'goalies': hg_final,
    }
    return awayRoster, homeRoster
  else:
    return False, False

def getLastTeamLineups(db,gameId):
  try:
    Boxscores = db['dev_boxscores']
    boxscore = Boxscores.find_one({'id':gameId})
    awayId = boxscore['awayTeam']['id']
    homeId = boxscore['homeTeam']['id']
    awayQuery = {
      'id': {'$lt': gameId},
      'awayTeam.id': awayId,
    }
    homeQuery = {
      'id': {'$lt': gameId},
      'homeTeam.id': homeId,
    }
    projection = {'id': 1, 'season': 1, 'gameType': 1, 'gameDate': 1, 'venue': 1, 'neutralSite': 1, 'homeTeam': 1, 'awayTeam': 1, 'boxscore': 1, 'period': 1}
    awayTeamLastGame = list(Boxscores.find(awayQuery,projection).sort('id',-1).limit(1))
    homeTeamLastGame = list(Boxscores.find(homeQuery,projection).sort('id',-1).limit(1))
    if len(awayTeamLastGame) == 0 or len(homeTeamLastGame) == 0:
      return {}
    else:
      awayTeamLastGame = awayTeamLastGame[0]
      homeTeamLastGame = homeTeamLastGame[0]
      away_home_away = 'awayTeam' if awayTeamLastGame['awayTeam']['id'] == awayId else 'homeTeam'
      home_home_away = 'awayTeam' if homeTeamLastGame['awayTeam']['id'] == homeId else 'homeTeam'

      awayTeamLastLineup = awayTeamLastGame['boxscore']['playerByGameStats'][away_home_away]
      homeTeamLastLineup = homeTeamLastGame['boxscore']['playerByGameStats'][home_home_away]

      return {
        'playerByGameStats': {
          'awayTeam': awayTeamLastLineup, 
          'homeTeam': homeTeamLastLineup,
        }
      }
  except:
    return {}

def all_combinations(lst):
  result = []
  for r in range(1, len(lst) + 1):
    for combo in combinations(lst, r):
      result.append(list(combo))
  return result

def parse_utc_offset(inUTCOffset):
  if inUTCOffset and inUTCOffset != -1:
    return abs(int(inUTCOffset.split(':')[0]))
  else:
    return -1

def parse_start_time(inStartTime):
  if inStartTime and inStartTime != -1:
    dt = datetime.strptime(str(inStartTime), "%Y-%m-%dT%H:%M:%SZ")
    start_date = int(f'{dt.year}{dt.month}{dt.day}')
    start_time = float(int(dt.hour) * ((int(dt.minute) * (int(dt.second)/60))/60))
    return start_date, start_time
  else:
    return -1, -1

def team_lookup(db):
  Teams = db['dev_teams']
  teams = list(Teams.find({},{'_id': 0}))
  team_lookup = {}
  for team in teams:
    team_lookup[team['id']] = team
  return team_lookup

def projectedRoster(db, gameId):
  landing_url = f'https://api-web.nhle.com/v1_1/gamecenter/{gameId}/landing'
  landing = requests.get(landing_url).json()
  awayId = landing['awayTeam']['id']
  homeId = landing['homeTeam']['id']
  
  af, ad, ag, hf, hd, hg = [], [], [], [], [], []
  afIds, adIds, agIds, hfIds, hdIds, hgIds = [], [], [], [], [], []
  afAges, adAges, agAges, hfAges, hdAges, hgAges = [], [], [], [], [], []
  
  if false_chain(landing,'matchup'):

    for skater in landing['matchup']['skaterSeasonStats']:
      skater_data = {'playerId':skater['playerId'], 'gamesPlayed':safe_chain(skater,'gamesPlayed',default=0)}
      if skater['teamId'] == awayId:
        if skater['position'].lower() == 'd':
          ad.append(skater_data)
          adIds.append(skater['playerId'])
        else:
          af.append(skater_data)
          afIds.append(skater['playerId'])
      elif skater['teamId'] == homeId:
        if skater['position'].lower() == 'd':
          hd.append(skater_data)
          hdIds.append(skater['playerId'])
        else:
          hf.append(skater_data)
          hfIds.append(skater['playerId'])
    
    for goalie in landing['matchup']['goalieSeasonStats']:
      goalie_data = {'playerId':goalie['playerId'], 'gamesPlayed':safe_chain(goalie,'gamesPlayed',default=0)}
      if goalie['teamId'] == awayId:
        ag.append(goalie_data)
        agIds.append(goalie['playerId'])
      elif goalie['teamId'] == homeId:
        hg.append(goalie_data)
        hgIds.append(goalie['playerId'])

  else:
    teamLookup = team_lookup(db)
    season = landing['season']
    away_abbr = teamLookup[awayId]['abbrev']
    home_abbr = teamLookup[homeId]['abbrev']
    away_roster_url = f'https://api-web.nhle.com/v1/roster/{away_abbr}/{season}'
    home_roster_url = f'https://api-web.nhle.com/v1/roster/{home_abbr}/{season}'
    home_roster = requests.get(home_roster_url).json()
    away_roster = requests.get(away_roster_url).json()

    for player in away_roster['forwards']:
      af.append({'playerId':player['id']})
      afIds.append(player['id'])
    for player in away_roster['defensemen']:
      ad.append({'playerId':player['id']})
      adIds.append(player['id'])
    for player in away_roster['goalies']:
      ag.append({'playerId':player['id']})
      agIds.append(player['id'])
    for player in home_roster['forwards']:
      hf.append({'playerId':player['id']})
      hfIds.append(player['id'])
    for player in home_roster['defensemen']:
      hd.append({'playerId':player['id']})
      hdIds.append(player['id'])
    for player in home_roster['goalies']:
      hg.append({'playerId':player['id']})
      hgIds.append(player['id'])

  afAges = getAllAges(db,afIds,landing['gameDate'])
  adAges = getAllAges(db,adIds,landing['gameDate'])
  agAges = getAllAges(db,agIds,landing['gameDate'])
  hfAges = getAllAges(db,hfIds,landing['gameDate'])
  hdAges = getAllAges(db,hdIds,landing['gameDate'])
  hgAges = getAllAges(db,hgIds,landing['gameDate'])

  afAges = [v for k,v in afAges.items()]
  adAges = [v for k,v in adAges.items()]
  agAges = [v for k,v in agAges.items()]
  hfAges = [v for k,v in hfAges.items()]
  hdAges = [v for k,v in hdAges.items()]
  hgAges = [v for k,v in hgAges.items()]
  
  if false_chain(landing,'matchup'):
    af = sorted(af, key=lambda d: d['gamesPlayed'], reverse=True)
    ad = sorted(ad, key=lambda d: d['gamesPlayed'], reverse=True)
    ag = sorted(ag, key=lambda d: d['gamesPlayed'], reverse=True)
    hf = sorted(hf, key=lambda d: d['gamesPlayed'], reverse=True)
    hd = sorted(hd, key=lambda d: d['gamesPlayed'], reverse=True)
    hg = sorted(hg, key=lambda d: d['gamesPlayed'], reverse=True)

  af_final, ad_final, ag_final, hf_final, hd_final, hg_final = [], [], [], [], [], []
  
  af_len = 13 if len(af) >= 13 else len(af)
  ad_len = 6 if len(ad) >= 6 else len(ad)
  ag_len = 2 if len(ag) >= 2 else len(ag)
  hf_len = 13 if len(hf) >= 13 else len(hf)
  hd_len = 6 if len(hd) >= 6 else len(hd)
  hg_len = 2 if len(hf) >= 2 else len(hf)
  
  # af_len, ad_len, ag_len, hf_len, hd_len, hg_len = len(af), len(ad), len(ag), len(hf), len(hd), len(hf)

  for i in range(0,af_len):
    af_final.append({'playerId': af[i]['playerId']})
  for i in range(0,hf_len):
    hf_final.append({'playerId': hf[i]['playerId']})
  for i in range(0,ad_len):
    ad_final.append({'playerId': ad[i]['playerId']})
  for i in range(0,hd_len):
    hd_final.append({'playerId': hd[i]['playerId']})
  for i in range(0,ag_len):
    ag_final.append({'playerId': ag[i]['playerId']})
  for i in range(0,hg_len):
    hg_final.append({'playerId': hg[i]['playerId']})
  awayRoster = {
    'forwards': af_final,
    'defense': ad_final,
    'goalies': ag_final,
    'forwardAverage': [(sum(afIds)/len(afIds)) if len(afIds) > 0 else -1],
    'defenseAverage': [(sum(adIds)/len(adIds)) if len(adIds) > 0 else -1],
    'goalieAverage': [(sum(agIds)/len(agIds)) if len(agIds) > 0 else -1],
    'forwardAverageAge': [(sum(afAges)/len(afAges)) if len(afAges) > 0 else -1],
    'defenseAverageAge': [(sum(adAges)/len(adAges)) if len(adAges) > 0 else -1],
    'goalieAverageAge': [(sum(agAges)/len(agAges)) if len(agAges) > 0 else -1],
  }
  homeRoster = {
    'forwards': hf_final,
    'defense': hd_final,
    'goalies': hg_final,
    'forwardAverage': [(sum(hfIds)/len(hfIds)) if len(hfIds) > 0 else -1],
    'defenseAverage': [(sum(hdIds)/len(hdIds)) if len(hdIds) > 0 else -1],
    'goalieAverage': [(sum(hgIds)/len(hgIds)) if len(hgIds) > 0 else -1],
    'forwardAverageAge': [(sum(hfAges)/len(hfAges)) if len(hfAges) > 0 else -1],
    'defenseAverageAge': [(sum(hdAges)/len(hdAges)) if len(hdAges) > 0 else -1],
    'goalieAverageAge': [(sum(hgAges)/len(hgAges)) if len(hgAges) > 0 else -1],
  }
  return awayRoster, homeRoster, landing