import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from pymongo import MongoClient
from util.helpers import safe_chain, n2n, formatTime, formatDatetime, formatDate

GAMES_BACK = 5

def playerFind(playerId):
  return {'$or':[
    {'boxscore.playerByGameStats.awayTeam.forwards': {'$elemMatch': {'playerId': playerId}}},
    {'boxscore.playerByGameStats.awayTeam.defense': {'$elemMatch': {'playerId': playerId}}},
    {'boxscore.playerByGameStats.awayTeam.goalies': {'$elemMatch': {'playerId': playerId}}},
    {'boxscore.playerByGameStats.homeTeam.forwards': {'$elemMatch': {'playerId': playerId}}},
    {'boxscore.playerByGameStats.homeTeam.defense': {'$elemMatch': {'playerId': playerId}}},
    {'boxscore.playerByGameStats.homeTeam.goalies': {'$elemMatch': {'playerId': playerId}}},
  ]}

def teamFind(teamId):
  return {'$or':[
    {'homeTeam.id': teamId},
    {'awayTeam.id': teamId},
  ]}

def last_player_game_stats(db,gameId,playerIDs,previousGames=GAMES_BACK):
  Boxscores = db['dev_boxscores']

  playerGames = {}
  for playerId0 in playerIDs:
    playerId = playerId0['playerId']
    playerGames[playerId] = []
    query = {
      '$or': [
        {'boxscore.playerByGameStats.awayTeam.forwards': {'$elemMatch': {'playerId': playerId}}},
        {'boxscore.playerByGameStats.awayTeam.defense': {'$elemMatch': {'playerId': playerId}}},
        {'boxscore.playerByGameStats.awayTeam.goalies': {'$elemMatch': {'playerId': playerId}}},
        {'boxscore.playerByGameStats.homeTeam.forwards': {'$elemMatch': {'playerId': playerId}}},
        {'boxscore.playerByGameStats.homeTeam.defense': {'$elemMatch': {'playerId': playerId}}},
        {'boxscore.playerByGameStats.homeTeam.goalies': {'$elemMatch': {'playerId': playerId}}},
      ],
      'id': {'$lt': gameId}
    }
    last_games = list(Boxscores.find(
      query,
      {
        '_id':0,
        'id':1,
        'gameDate': 1,
        'awayTeam.id': 1,
        'homeTeam.id': 1,
        'awayForwards': {'$filter': {'input': '$boxscore.playerByGameStats.awayTeam.forwards','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
        'awayDefense': {'$filter': {'input': '$boxscore.playerByGameStats.awayTeam.defense','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
        'awayGoalies': {'$filter': {'input': '$boxscore.playerByGameStats.awayTeam.goalies','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
        'homeForwards': {'$filter': {'input': '$boxscore.playerByGameStats.homeTeam.forwards','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
        'homeDefense': {'$filter': {'input': '$boxscore.playerByGameStats.homeTeam.defense','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
        'homeGoalies': {'$filter': {'input': '$boxscore.playerByGameStats.homeTeam.goalies','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      }
    ).sort('id',-1).limit(previousGames))
    
    for i in last_games:
      statLine = {'id': i['id'],'gameDate':i['gameDate']}
      if len(i['awayForwards']) > 0:
        statLine['stats'] = i['awayForwards']
        statLine['team'] = i['awayTeam']['id']
        statLine['position'] = 'Forward'
        statLine['homeAway'] = 'away'
      elif len(i['awayDefense']) > 0:
        statLine['stats'] = i['awayDefense']
        statLine['team'] = i['awayTeam']['id']
        statLine['position'] = 'Defenseman'
        statLine['homeAway'] = 'away'
      elif len(i['awayGoalies']) > 0:
        statLine['stats'] = i['awayGoalies']
        statLine['team'] = i['awayTeam']['id']
        statLine['position'] = 'Goalie'
        statLine['homeAway'] = 'away'
      elif len(i['homeForwards']) > 0:
        statLine['stats'] = i['homeForwards']
        statLine['team'] = i['homeTeam']['id']
        statLine['position'] = 'Forward'
        statLine['homeAway'] = 'home'
      elif len(i['homeDefense']) > 0:
        statLine['stats'] = i['homeDefense']
        statLine['team'] = i['homeTeam']['id']
        statLine['position'] = 'Defenseman'
        statLine['homeAway'] = 'home'
      elif len(i['homeGoalies']) > 0:
        statLine['stats'] = i['homeGoalies']
        statLine['team'] = i['homeTeam']['id']
        statLine['position'] = 'Goalie'
        statLine['homeAway'] = 'home'

      playerGames[playerId].append(statLine)
  return playerGames

def list_players_in_game(db,gameId):
  Boxscores = db['dev_boxscores']
  game = list(Boxscores.find(
    {'id':gameId},
    {'_id':0,'id':1,'boxscore.playerByGameStats':1}
  ))
  playerIDs = {'home':[],'away':[]}
  for player in game[0]['boxscore']['playerByGameStats']['homeTeam']['forwards']:
    playerIDs['home'].append(player['playerId'])
  for player in game[0]['boxscore']['playerByGameStats']['homeTeam']['defense']:
    playerIDs['home'].append(player['playerId'])
  for player in game[0]['boxscore']['playerByGameStats']['homeTeam']['goalies']:
    playerIDs['home'].append(player['playerId'])
  for player in game[0]['boxscore']['playerByGameStats']['awayTeam']['forwards']:
    playerIDs['away'].append(player['playerId'])
  for player in game[0]['boxscore']['playerByGameStats']['awayTeam']['defense']:
    playerIDs['away'].append(player['playerId'])
  for player in game[0]['boxscore']['playerByGameStats']['awayTeam']['goalies']:
    playerIDs['away'].append(player['playerId'])
  return playerIDs

def list_player_games(db,playerId,sort=-1):
  Boxscores = db['dev_boxscores']
  playerQuery = playerFind(playerId=playerId)
  gameList = list(Boxscores.find(
    playerQuery,
    {'_id': 0, 'id': 1}
  ).sort('id', sort))
  gameList = [game['id'] for game in gameList]
  return gameList

def list_team_games(db,teamId,sort=-1):
  Boxscores = db['dev_boxscores']
  teamQuery = teamFind(teamId=teamId)
  gameList = list(Boxscores.find(
    teamQuery,
    {'_id': 0, 'id': 1}
  ).sort('id', sort))
  gameList = [game['id'] for game in gameList]
  return gameList


def get_last_game_player_stats(playerData,playerId,playerTitle='',previousGames=GAMES_BACK,isGoalie=False):
  try:
    if playerId != -1:
      # gameList = list_player_games(db=db,playerId=playerId)
      # lastGameIDs = gameList[0:previousGames]
      # playerQuery = playerFind(playerId=playerId)
      # Boxscores = db['dev_boxscores']
      # statPipeline = [
      #   {
      #     '$match': {
      #       '$and': [
      #         {'id': {'$in': lastGameIDs}},
      #         playerQuery
      #       ]
      #     }
      #   },
      #   {
      #     '$project': {
      #       '_id': 0,
      #       'id': 1,
      #       'gameDate': 1,
      #       'awayTeam.id': 1,
      #       'homeTeam.id': 1,
      #       'awayForwards': {'$filter': {'input': '$boxscore.playerByGameStats.awayTeam.forwards','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      #       'awayDefense': {'$filter': {'input': '$boxscore.playerByGameStats.awayTeam.defense','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      #       'awayGoalies': {'$filter': {'input': '$boxscore.playerByGameStats.awayTeam.goalies','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      #       'homeForwards': {'$filter': {'input': '$boxscore.playerByGameStats.homeTeam.forwards','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      #       'homeDefense': {'$filter': {'input': '$boxscore.playerByGameStats.homeTeam.defense','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      #       'homeGoalies': {'$filter': {'input': '$boxscore.playerByGameStats.homeTeam.goalies','as': 'item','cond': {'$eq': ['$$item.playerId', playerId]}}},
      #     }
      #   }
      # ]
      # fullStatList = list(Boxscores.aggregate(statPipeline))
      # fullStatList = playerData[playerId]
      statList = playerData[playerId]
      # for stat in fullStatList:
      #   statLine = {'id': stat['id'],'gameDate':stat['gameDate']}
      #   if len(stat['awayForwards']) > 0 or len(stat['awayDefense']) > 0 or len(stat['awayGoalies']) > 0:
      #     statLine['team'] = stat['awayTeam']['id']
      #     statLine['homeAway'] = 'away'
      #   elif len(stat['homeForwards']) > 0 or len(stat['homeDefense']) > 0 or len(stat['homeGoalies']) > 0:
      #     statLine['team'] = stat['homeTeam']['id']
      #     statLine['homeAway'] = 'home'

      #   if len(stat['awayForwards']) > 0:
      #     statLine['stats'] = stat['awayForwards']
      #   elif len(stat['awayDefense']) > 0:
      #     statLine['stats'] = stat['awayDefense']
      #   elif len(stat['awayGoalies']) > 0:
      #     statLine['stats'] = stat['awayGoalies']
      #   elif len(stat['homeForwards']) > 0:
      #     statLine['stats'] = stat['homeForwards']
      #   elif len(stat['homeDefense']) > 0:
      #     statLine['stats'] = stat['homeDefense']
      #   elif len(stat['homeGoalies']) > 0:
      #     statLine['stats'] = stat['homeGoalies']

      #   statList.append(statLine)

      lastGames = {}
      for i in range(0,len(statList)):
        lastGames[f'{playerTitle}Back{i+1}GameId'] = safe_chain(statList,i,'id')
        lastGames[f'{playerTitle}Back{i+1}GameDate'] = formatDate(safe_chain(statList,i,'gameDate'))
        lastGames[f'{playerTitle}Back{i+1}GameTeam'] = safe_chain(statList,i,'team')
        lastGames[f'{playerTitle}Back{i+1}GameHomeAway'] = n2n(safe_chain(statList,i,'homeAway'))
        lastGames[f'{playerTitle}Back{i+1}GamePlayer'] = safe_chain(statList,i,'stats','playerId')
        lastGames[f'{playerTitle}Back{i+1}GamePosition'] = n2n(safe_chain(statList,i,'stats','position'))
        lastGames[f'{playerTitle}Back{i+1}GamePIM'] = safe_chain(statList,i,'stats','pim')
        lastGames[f'{playerTitle}Back{i+1}GameTOI'] = formatTime(safe_chain(statList,i,'stats','toi'))
        if isGoalie:
          if type(safe_chain(statList,i,'stats','evenStrengthShotsAgainst')) == str:
            essa = int(safe_chain(statList,i,'stats','evenStrengthShotsAgainst').split('/')[1])
          else:
            essa = safe_chain(statList,i,'stats','evenStrengthShotsAgainst')
          if type(safe_chain(statList,i,'stats','powerPlayShotsAgainst')) == str:
            ppsa = int(safe_chain(statList,i,'stats','powerPlayShotsAgainst').split('/')[1])
          else:
            ppsa = safe_chain(statList,i,'stats','powerPlayShotsAgainst')
          if type(safe_chain(statList,i,'stats','shorthandedShotsAgainst')) == str:
            ssa = int(safe_chain(statList,i,'stats','shorthandedShotsAgainst').split('/')[1])
          else:
            ssa = safe_chain(statList,i,'stats','shorthandedShotsAgainst')
          if type(safe_chain(statList,i,'stats','saveShotsAgainst')) == str:
            saves = int(safe_chain(statList,i,'stats','saveShotsAgainst').split('/')[1])
          else:
            saves = safe_chain(statList,i,'stats','saveShotsAgainst')
          lastGames[f'{playerTitle}Back{i+1}GameEvenStrengthShotsAgainst'] = essa
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayShotsAgainst'] = ppsa
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedShotsAgainst'] = ssa
          lastGames[f'{playerTitle}Back{i+1}GameSaves'] = saves
          lastGames[f'{playerTitle}Back{i+1}GameSavePercentage'] = float(safe_chain(statList,i,'stats','savePctg'))
          lastGames[f'{playerTitle}Back{i+1}GameEvenStrengthGoalsAgainst'] = safe_chain(statList,i,'stats','evenStrengthGoalsAgainst')
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayGoalsAgainst'] = safe_chain(statList,i,'stats','powerPlayGoalsAgainst')
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedGoalsAgainst'] = safe_chain(statList,i,'stats','shorthandedGoalsAgainst')
          lastGames[f'{playerTitle}Back{i+1}GameGoalsAgainst'] = safe_chain(statList,i,'stats','goalsAgainst')
        else:
          if type(safe_chain(statList,i,'stats','faceoffs')) == str:
            faceoffs = int(safe_chain(statList,i,'stats','faceoffs').split('/')[1])
          else:
            faceoffs = safe_chain(statList,i,'stats','faceoffs')
          lastGames[f'{playerTitle}Back{i+1}GameGoals'] = safe_chain(statList,i,'stats','goals')
          lastGames[f'{playerTitle}Back{i+1}GameAssists'] = safe_chain(statList,i,'stats','assists')
          lastGames[f'{playerTitle}Back{i+1}GamePoints'] = safe_chain(statList,i,'stats','points')
          lastGames[f'{playerTitle}Back{i+1}GamePlusMinus'] = safe_chain(statList,i,'stats','plusMinus')
          lastGames[f'{playerTitle}Back{i+1}GameHits'] = safe_chain(statList,i,'stats','hits')
          lastGames[f'{playerTitle}Back{i+1}GameBlockedShots'] = safe_chain(statList,i,'stats','blockedShots')
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayGoals'] = safe_chain(statList,i,'stats','powerPlayGoals')
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayPoints'] = safe_chain(statList,i,'stats','powerPlayPoints')
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedGoals'] = safe_chain(statList,i,'stats','shorthandedGoals')
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedPoints'] = safe_chain(statList,i,'stats','shPoints')
          lastGames[f'{playerTitle}Back{i+1}GameShots'] = safe_chain(statList,i,'stats','shots')
          lastGames[f'{playerTitle}Back{i+1}GameFaceoffs'] = faceoffs
          lastGames[f'{playerTitle}Back{i+1}GameFaceoffWinPercentage'] = safe_chain(statList,i,'stats','faceoffWinningPctg')
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayTOI'] = formatTime(safe_chain(statList,i,'stats','powerPlayToi'))
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedTOI'] = formatTime(safe_chain(statList,i,'stats','shorthandedToi'))
      return lastGames
    else:
      lastGames = {}
      for i in range(0,previousGames):
        lastGames[f'{playerTitle}Back{i+1}GameId'] = -1
        lastGames[f'{playerTitle}Back{i+1}GameDate'] = -1
        lastGames[f'{playerTitle}Back{i+1}GameTeam'] = -1
        lastGames[f'{playerTitle}Back{i+1}GameHomeAway'] = -1
        lastGames[f'{playerTitle}Back{i+1}GamePlayer'] = -1
        lastGames[f'{playerTitle}Back{i+1}GamePosition'] = -1
        lastGames[f'{playerTitle}Back{i+1}GamePIM'] = -1
        lastGames[f'{playerTitle}Back{i+1}GameTOI'] = -1
        if isGoalie:
          lastGames[f'{playerTitle}Back{i+1}GameEvenStrengthShotsAgainst'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayShotsAgainst'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedShotsAgainst'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameSaves'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameSavePercentage'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameEvenStrengthGoalsAgainst'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayGoalsAgainst'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedGoalsAgainst'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameGoalsAgainst'] = -1
        else:
          lastGames[f'{playerTitle}Back{i+1}GameGoals'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameAssists'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePoints'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePlusMinus'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameHits'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameBlockedShots'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayGoals'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayPoints'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedGoals'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedPoints'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameShots'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameFaceoffs'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameFaceoffWinPercentage'] = -1
          lastGames[f'{playerTitle}Back{i+1}GamePowerPlayTOI'] = -1
          lastGames[f'{playerTitle}Back{i+1}GameShorthandedTOI'] = -1
      return lastGames
  except Exception as error:
    print('get_last_game_player_stats',error)


def get_last_game_team_stats(db,teamId,teamTitle='',previousGames=GAMES_BACK):
  try:
    if teamId != -1:
      gameList = list_team_games(db=db,teamId=teamId)
      lastGameIDs = gameList[0:previousGames]
      teamQuery = teamFind(teamId=teamId)
      Boxscores = db['dev_boxscores']
      teamPipeline = [
        {
          '$match': {
            '$and': [
              {'id': {'$in': lastGameIDs}},
              teamQuery
            ]
          }
        },
        {
          '$project': {
            '_id': 0,
            'id': 1,
            'gameDate': 1,
            'gameType': 1,
            'awayTeam': 1,
            'homeTeam': 1,
            'venue.default': 1,
            'period': 1,
            'startTimeUTC': 1,
            'easternUTCOffset': 1,
            'venueUTCOffset': 1,
          }
        }
      ]
      teamGames = list(Boxscores.aggregate(teamPipeline))
      statList = []
      for game in teamGames:
        team = game['awayTeam'] if game['awayTeam']['id'] == teamId else game['homeTeam']
        homeAway = 'away' if game['awayTeam']['id'] == teamId else 'home'
        opponent = game['homeTeam'] if game['awayTeam']['id'] == teamId else game['awayTeam']
        if team['score'] > opponent['score']:
          outcome = 'win'
        elif team['score'] < opponent['score']:
          outcome = 'loss'
        else:
          outcome = 'tie'
        statLine = {
          'id': game['id'],
          'gameDate':game['gameDate'],
          'gameType': game['gameType'],
          'venue': game['venue']['default'],
          'period': game['period'],
          'startTimeUTC': game['startTimeUTC'],
          'easternUTCOffset': game['easternUTCOffset'],
          'venueUTCOffset': game['venueUTCOffset'],
          'team': team,
          'opponent': opponent,
          'homeAway': homeAway,
          'outcome': outcome
        }
        statList.append(statLine)

      lastGames = {}
      for i in range(0,previousGames):
        if i > len(statList) - 1:
          lastGames[f'{teamTitle}Back{i+1}GameId'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameDate'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameType'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameVenue'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameStartTime'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameEasternOffset'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameVenueOffset'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOutcome'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameHomeAway'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameFinalPeriod'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameScore'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameShots'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameFaceoffWinPercentage'] = -1
          lastGames[f'{teamTitle}Back{i+1}GamePowerPlays'] = -1
          lastGames[f'{teamTitle}Back{i+1}GamePowerPlayPercentage'] = -1
          lastGames[f'{teamTitle}Back{i+1}GamePIM'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameHits'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameBlocks'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponent'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentScore'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentShots'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentFaceoffWinPercentage'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentPowerPlays'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentPowerPlayPercentage'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentPIM'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentHits'] = -1
          lastGames[f'{teamTitle}Back{i+1}GameOpponentBlocks'] = -1
        else:
          if type(safe_chain(statList,i,'team','powerPlayConversion')) == str:
            teamPowerPlaySplit = safe_chain(statList,i,'team','powerPlayConversion').split('/')
            teamPowerPlays = int(teamPowerPlaySplit[1])
            teamPowerPlayPercentage = float(int(teamPowerPlaySplit[0])/teamPowerPlays) if teamPowerPlays != 0 else 0
          else:
            teamPowerPlaySplit = safe_chain(statList,i,'team','powerPlayConversion')
            teamPowerPlays = -1
            teamPowerPlayPercentage = -1
          if type(safe_chain(statList,i,'opponent','powerPlayConversion')) == str:
            opponentPowerPlaySplit = safe_chain(statList,i,'opponent','powerPlayConversion').split('/')
            opponentPowerPlays = int(opponentPowerPlaySplit[1])
            opponentPowerPlayPercentage = float(int(opponentPowerPlaySplit[0])/opponentPowerPlays) if opponentPowerPlays != 0 else 0
          else:
            opponentPowerPlaySplit = safe_chain(statList,i,'opponent','powerPlayConversion')
            opponentPowerPlays = -1
            opponentPowerPlayPercentage = -1

          lastGames[f'{teamTitle}Back{i+1}GameId'] = safe_chain(statList,i,'id')
          lastGames[f'{teamTitle}Back{i+1}GameDate'] = formatDate(safe_chain(statList,i,'gameDate'))
          lastGames[f'{teamTitle}Back{i+1}GameType'] = safe_chain(statList,i,'gameType')
          lastGames[f'{teamTitle}Back{i+1}GameVenue'] = n2n(safe_chain(statList,i,'venue'))
          lastGames[f'{teamTitle}Back{i+1}GameStartTime'] = formatDatetime(safe_chain(statList,i,'startTimeUTC'))
          lastGames[f'{teamTitle}Back{i+1}GameEasternOffset'] = formatTime(safe_chain(statList,i,'easternUTCOffset'))
          lastGames[f'{teamTitle}Back{i+1}GameVenueOffset'] = formatTime(safe_chain(statList,i,'venueUTCOffset'))
          lastGames[f'{teamTitle}Back{i+1}GameOutcome'] = n2n(safe_chain(statList,i,'outcome'))
          lastGames[f'{teamTitle}Back{i+1}GameHomeAway'] = n2n(safe_chain(statList,i,'homeAway'))
          lastGames[f'{teamTitle}Back{i+1}GameFinalPeriod'] = safe_chain(statList,i,'period')
          lastGames[f'{teamTitle}Back{i+1}GameScore'] = safe_chain(statList,i,'team','score')
          lastGames[f'{teamTitle}Back{i+1}GameShots'] = safe_chain(statList,i,'team','sog')
          lastGames[f'{teamTitle}Back{i+1}GameFaceoffWinPercentage'] = safe_chain(statList,i,'team','faceoffWinningPctg')
          lastGames[f'{teamTitle}Back{i+1}GamePowerPlays'] = teamPowerPlays
          lastGames[f'{teamTitle}Back{i+1}GamePowerPlayPercentage'] = teamPowerPlayPercentage
          lastGames[f'{teamTitle}Back{i+1}GamePIM'] = safe_chain(statList,i,'team','pim')
          lastGames[f'{teamTitle}Back{i+1}GameHits'] = safe_chain(statList,i,'team','hits')
          lastGames[f'{teamTitle}Back{i+1}GameBlocks'] = safe_chain(statList,i,'team','blocks')
          lastGames[f'{teamTitle}Back{i+1}GameOpponent'] = safe_chain(statList,i,'opponent','id')
          lastGames[f'{teamTitle}Back{i+1}GameOpponentScore'] = safe_chain(statList,i,'opponent','score')
          lastGames[f'{teamTitle}Back{i+1}GameOpponentShots'] = safe_chain(statList,i,'opponent','sog')
          lastGames[f'{teamTitle}Back{i+1}GameOpponentFaceoffWinPercentage'] = safe_chain(statList,i,'opponent','faceoffWinningPctg')
          lastGames[f'{teamTitle}Back{i+1}GameOpponentPowerPlays'] = opponentPowerPlays
          lastGames[f'{teamTitle}Back{i+1}GameOpponentPowerPlayPercentage'] = opponentPowerPlayPercentage
          lastGames[f'{teamTitle}Back{i+1}GameOpponentPIM'] = safe_chain(statList,i,'opponent','pim')
          lastGames[f'{teamTitle}Back{i+1}GameOpponentHits'] = safe_chain(statList,i,'opponent','hits')
          lastGames[f'{teamTitle}Back{i+1}GameOpponentBlocks'] = safe_chain(statList,i,'opponent','blocks')
      return lastGames
    else:
      lastGames = {}
      for i in range(0,previousGames):
        lastGames[f'{teamTitle}Back{i+1}GameId'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameDate'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameType'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameVenue'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameStartTime'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameEasternOffset'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameVenueOffset'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOutcome'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameHomeAway'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameFinalPeriod'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameScore'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameShots'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameFaceoffWinPercentage'] = -1
        lastGames[f'{teamTitle}Back{i+1}GamePowerPlays'] = -1
        lastGames[f'{teamTitle}Back{i+1}GamePowerPlayPercentage'] = -1
        lastGames[f'{teamTitle}Back{i+1}GamePIM'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameHits'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameBlocks'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponent'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentScore'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentShots'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentFaceoffWinPercentage'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentPowerPlays'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentPowerPlayPercentage'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentPIM'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentHits'] = -1
        lastGames[f'{teamTitle}Back{i+1}GameOpponentBlocks'] = -1
      return lastGames
  except Exception as error:
    print('get_last_game_team_stats',error)


# client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
# db = client["hockey"]

# playerIDs = list_players_in_game(db,2023020215)
# last_player_game_stats(db=db,gameId=2023020215,playerIDs=playerIDs['away'])