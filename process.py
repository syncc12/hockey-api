import requests
from pymongo import MongoClient
import math
from datetime import datetime
import os
from util.helpers import safe_chain, false_chain, n2n, isNaN, getAge, getPlayer, getPlayerData

REPLACE_VALUE = -1

def format_start_time(game):
  # print('startTimeUTC',game['startTimeUTC'])
  startTime = REPLACE_VALUE
  if 'startTimeUTC' in game and game['startTimeUTC']:
    st = datetime.strptime(game['startTimeUTC'], "%Y-%m-%dT%H:%M:%SZ")  # Adjust the format if necessary
    # print('st',st)
    startTime = int(f"{st.hour}{st.minute:02d}")
    # print('startTime',startTime)
  return startTime

def format_game_date(game):
  date = REPLACE_VALUE
  if 'gameDate' in game and game['gameDate']:
    date = int(game['gameDate'].replace('-', ''))
  return date

def nhl_data(game={},boxscore={},test=False):
  db_username = os.getenv('DB_USERNAME')
  db_name = os.getenv('DB_NAME')
  db_password = os.getenv('DB_PASSWORD')
  db_url = f"mongodb+srv://{db_username}:{db_password}@{db_name}"
  client = MongoClient(db_url)
  db = client['hockey']
  Players = db['dev_players']

  pbgs = boxscore['boxscore']['playerByGameStats']
  af = [{'id':p['playerId'],'position':p['position']} for p in pbgs['awayTeam']['forwards'] if 'playerId' in p]
  ad = [{'id':p['playerId'],'position':p['position']} for p in pbgs['awayTeam']['defense'] if 'playerId' in p]
  ag = [{'id':p['playerId'],'position':p['position']} for p in pbgs['awayTeam']['goalies'] if 'playerId' in p]
  hf = [{'id':p['playerId'],'position':p['position']} for p in pbgs['homeTeam']['forwards'] if 'playerId' in p]
  hd = [{'id':p['playerId'],'position':p['position']} for p in pbgs['homeTeam']['defense'] if 'playerId' in p]
  hg = [{'id':p['playerId'],'position':p['position']} for p in pbgs['homeTeam']['goalies'] if 'playerId' in p]
  a = []
  for p in af: a.append({'playerId':p['id']})
  for p in ad: a.append({'playerId':p['id']})
  for p in ag: a.append({'playerId':p['id']})
  for p in hf: a.append({'playerId':p['id']})
  for p in hd: a.append({'playerId':p['id']})
  for p in hg: a.append({'playerId':p['id']})

  positions = {}
  for p in af: positions[p['id']] = p['position']
  for p in ad: positions[p['id']] = p['position']
  for p in ag: positions[p['id']] = p['position']
  for p in hf: positions[p['id']] = p['position']
  for p in hd: positions[p['id']] = p['position']
  for p in hg: positions[p['id']] = p['position']


  players = list(Players.find(
    {'$or': a},
    {'_id':0, 'playerId':1, 'shootsCatches':1, 'birthDate': 1, 'position': 1}
  ))
  gameDate = format_game_date(boxscore)
  gameStartTime = format_start_time(boxscore)

  
  startTime = gameStartTime
  date = gameDate
  awayHeadCoach = n2n(boxscore['boxscore']['gameInfo']['awayTeam']['headCoach']['default'])
  homeHeadCoach = n2n(boxscore['boxscore']['gameInfo']['homeTeam']['headCoach']['default'])
  ref1 = n2n(boxscore['boxscore']['gameInfo']['referees'][0]['default'])
  ref2 = n2n(boxscore['boxscore']['gameInfo']['referees'][1]['default'])
  linesman1 = n2n(boxscore['boxscore']['gameInfo']['linesmen'][0]['default'])
  linesman1 = n2n(boxscore['boxscore']['gameInfo']['linesmen'][1]['default'])
  awayForward1 = safe_chain(af,0,'id')
  awayForward1Position = n2n(safe_chain(positions,safe_chain(af,0,'id')))
  awayForward1Age = getAge(getPlayer(players, safe_chain(af,0,'id')),gameDate)
  awayForward1Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,0,'id')),0,'shootsCatches'))
  awayForward2 = safe_chain(af,1,'id')
  awayForward2Position = n2n(safe_chain(positions,safe_chain(af,1,'id')))
  awayForward2Age = getAge(getPlayer(players, safe_chain(af,1,'id')),gameDate)
  awayForward2Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,1,'id')),0,'shootsCatches'))
  awayForward3 = safe_chain(af,2,'id')
  awayForward3Position = n2n(safe_chain(positions,safe_chain(af,2,'id')))
  awayForward3Age = getAge(getPlayer(players, safe_chain(af,2,'id')),gameDate)
  awayForward3Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,2,'id')),0,'shootsCatches'))
  awayForward4 = safe_chain(af,3,'id')
  awayForward4Position = n2n(safe_chain(positions,safe_chain(af,3,'id')))
  awayForward4Age = getAge(getPlayer(players, safe_chain(af,3,'id')),gameDate)
  awayForward4Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,3,'id')),0,'shootsCatches'))
  awayForward5 = safe_chain(af,4,'id')
  awayForward5Position = n2n(safe_chain(positions,safe_chain(af,4,'id')))
  awayForward5Age = getAge(getPlayer(players, safe_chain(af,4,'id')),gameDate)
  awayForward5Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,4,'id')),0,'shootsCatches'))
  awayForward6 = safe_chain(af,5,'id')
  awayForward6Position = n2n(safe_chain(positions,safe_chain(af,5,'id')))
  awayForward6Age = getAge(getPlayer(players, safe_chain(af,5,'id')),gameDate)
  awayForward6Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,5,'id')),0,'shootsCatches'))
  awayForward7 = safe_chain(af,6,'id')
  awayForward7Position = n2n(safe_chain(positions,safe_chain(af,6,'id')))
  awayForward7Age = getAge(getPlayer(players, safe_chain(af,6,'id')),gameDate)
  awayForward7Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,6,'id')),0,'shootsCatches'))
  awayForward8 = safe_chain(af,7,'id')
  awayForward8Position = n2n(safe_chain(positions,safe_chain(af,7,'id')))
  awayForward8Age = getAge(getPlayer(players, safe_chain(af,7,'id')),gameDate)
  awayForward8Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,7,'id')),0,'shootsCatches'))
  awayForward9 = safe_chain(af,8,'id')
  awayForward9Position = n2n(safe_chain(positions,safe_chain(af,8,'id')))
  awayForward9Age = getAge(getPlayer(players, safe_chain(af,8,'id')),gameDate)
  awayForward9Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,8,'id')),0,'shootsCatches'))
  awayForward10 = safe_chain(af,9,'id')
  awayForward10Position = n2n(safe_chain(positions,safe_chain(af,9,'id')))
  awayForward10Age = getAge(getPlayer(players, safe_chain(af,9,'id')),gameDate)
  awayForward10Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,9,'id')),0,'shootsCatches'))
  awayForward11 = safe_chain(af,10,'id')
  awayForward11Position = n2n(safe_chain(positions,safe_chain(af,10,'id')))
  awayForward11Age = getAge(getPlayer(players, safe_chain(af,10,'id')),gameDate)
  awayForward11Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,10,'id')),0,'shootsCatches'))
  awayForward12 = safe_chain(af,11,'id')
  awayForward12Position = n2n(safe_chain(positions,safe_chain(af,11,'id')))
  awayForward12Age = getAge(getPlayer(players, safe_chain(af,11,'id')),gameDate)
  awayForward12Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,11,'id')),0,'shootsCatches'))
  awayForward13 = safe_chain(af,12,'id')
  awayForward13Position = n2n(safe_chain(positions,safe_chain(af,12,'id')))
  awayForward13Age = getAge(getPlayer(players, safe_chain(af,12,'id')),gameDate)
  awayForward13Shoots = n2n(safe_chain(getPlayer(players, safe_chain(af,12,'id')),0,'shootsCatches'))
  awayDefenseman1 = safe_chain(ad,0,'id')
  awayDefenseman1Position = n2n(safe_chain(positions,safe_chain(ad,0,'id')))
  awayDefenseman1Age = getAge(getPlayer(players, safe_chain(ad,0,'id')),gameDate)
  awayDefenseman1Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,0,'id')),0,'shootsCatches'))
  awayDefenseman2 = safe_chain(ad,1,'id')
  awayDefenseman2Position = n2n(safe_chain(positions,safe_chain(ad,1,'id')))
  awayDefenseman2Age = getAge(getPlayer(players, safe_chain(ad,1,'id')),gameDate)
  awayDefenseman2Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,1,'id')),0,'shootsCatches'))
  awayDefenseman3 = safe_chain(ad,2,'id')
  awayDefenseman3Position = n2n(safe_chain(positions,safe_chain(ad,2,'id')))
  awayDefenseman3Age = getAge(getPlayer(players, safe_chain(ad,2,'id')),gameDate)
  awayDefenseman3Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,2,'id')),0,'shootsCatches'))
  awayDefenseman4 = safe_chain(ad,3,'id')
  awayDefenseman4Position = n2n(safe_chain(positions,safe_chain(ad,3,'id')))
  awayDefenseman4Age = getAge(getPlayer(players, safe_chain(ad,3,'id')),gameDate)
  awayDefenseman4Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,3,'id')),0,'shootsCatches'))
  awayDefenseman5 = safe_chain(ad,4,'id')
  awayDefenseman5Position = n2n(safe_chain(positions,safe_chain(ad,4,'id')))
  awayDefenseman5Age = getAge(getPlayer(players, safe_chain(ad,4,'id')),gameDate)
  awayDefenseman5Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,4,'id')),0,'shootsCatches'))
  awayDefenseman6 = safe_chain(ad,5,'id')
  awayDefenseman6Position = n2n(safe_chain(positions,safe_chain(ad,5,'id')))
  awayDefenseman6Age = getAge(getPlayer(players, safe_chain(ad,5,'id')),gameDate)
  awayDefenseman6Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,5,'id')),0,'shootsCatches'))
  awayDefenseman7 = safe_chain(ad,6,'id')
  awayDefenseman7Position = n2n(safe_chain(positions,safe_chain(ad,6,'id')))
  awayDefenseman7Age = getAge(getPlayer(players, safe_chain(ad,6,'id')),gameDate)
  awayDefenseman7Shoots = n2n(safe_chain(getPlayer(players, safe_chain(ad,6,'id')),0,'shootsCatches'))
  awayStartingGoalie = safe_chain(ag,0,'id')
  awayStartingGoalieCatches = n2n(safe_chain(getPlayer(players, safe_chain(ag,0,'id')),0,'shootsCatches'))
  awayStartingGoalieAge = getAge(getPlayer(players, safe_chain(ag,0,'id')),gameDate)
  awayStartingGoalieHeight = safe_chain(getPlayer(players, safe_chain(ag,0,'id')),0,'heightInInches')
  awayStartingGoalieWeight = safe_chain(getPlayer(players, safe_chain(ag,0,'id')),0,'weightInPounds')
  awayBackupGoalie = safe_chain(ag,1,'id')
  awayBackupGoalieCatches = n2n(safe_chain(getPlayer(players, safe_chain(ag,1,'id')),0,'shootsCatches'))
  awayBackupGoalieAge = getAge(getPlayer(players, safe_chain(ag,1,'id')),gameDate)
  awayBackupGoalieHeight = safe_chain(getPlayer(players, safe_chain(ag,1,'id')),0,'heightInInches')
  awayBackupGoalieWeight = safe_chain(getPlayer(players, safe_chain(ag,1,'id')),0,'weightInPounds')
  awayThirdGoalie = safe_chain(ag,2,'id')
  awayThirdGoalieCatches = n2n(safe_chain(getPlayer(players, safe_chain(ag,2,'id')),0,'shootsCatches'))
  awayThirdGoalieAge = getAge(getPlayer(players, safe_chain(ag,2,'id')),gameDate)
  awayThirdGoalieHeight = safe_chain(getPlayer(players, safe_chain(ag,2,'id')),0,'heightInInches')
  awayThirdGoalieWeight = safe_chain(getPlayer(players, safe_chain(ag,2,'id')),0,'weightInPounds')
  homeForward1 = safe_chain(hf,0,'id')
  homeForward1Position = n2n(safe_chain(positions,safe_chain(hf,0,'id')))
  homeForward1Age = getAge(getPlayer(players, safe_chain(hf,0,'id')),gameDate)
  homeForward1Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,0,'id')),0,'shootsCatches'))
  homeForward2 = safe_chain(hf,1,'id')
  homeForward2Position = n2n(safe_chain(positions,safe_chain(hf,1,'id')))
  homeForward2Age = getAge(getPlayer(players, safe_chain(hf,1,'id')),gameDate)
  homeForward2Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,1,'id')),0,'shootsCatches'))
  homeForward3 = safe_chain(hf,2,'id')
  homeForward3Position = n2n(safe_chain(positions,safe_chain(hf,2,'id')))
  homeForward3Age = getAge(getPlayer(players, safe_chain(hf,2,'id')),gameDate)
  homeForward3Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,2,'id')),0,'shootsCatches'))
  homeForward4 = safe_chain(hf,3,'id')
  homeForward4Position = n2n(safe_chain(positions,safe_chain(hf,3,'id')))
  homeForward4Age = getAge(getPlayer(players, safe_chain(hf,3,'id')),gameDate)
  homeForward4Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,3,'id')),0,'shootsCatches'))
  homeForward5 = safe_chain(hf,4,'id')
  homeForward5Position = n2n(safe_chain(positions,safe_chain(hf,4,'id')))
  homeForward5Age = getAge(getPlayer(players, safe_chain(hf,4,'id')),gameDate)
  homeForward5Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,4,'id')),0,'shootsCatches'))
  homeForward6 = safe_chain(hf,5,'id')
  homeForward6Position = n2n(safe_chain(positions,safe_chain(hf,5,'id')))
  homeForward6Age = getAge(getPlayer(players, safe_chain(hf,5,'id')),gameDate)
  homeForward6Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,5,'id')),0,'shootsCatches'))
  homeForward7 = safe_chain(hf,6,'id')
  homeForward7Position = n2n(safe_chain(positions,safe_chain(hf,6,'id')))
  homeForward7Age = getAge(getPlayer(players, safe_chain(hf,6,'id')),gameDate)
  homeForward7Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,6,'id')),0,'shootsCatches'))
  homeForward8 = safe_chain(hf,7,'id')
  homeForward8Position = n2n(safe_chain(positions,safe_chain(hf,7,'id')))
  homeForward8Age = getAge(getPlayer(players, safe_chain(hf,7,'id')),gameDate)
  homeForward8Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,7,'id')),0,'shootsCatches'))
  homeForward9 = safe_chain(hf,8,'id')
  homeForward9Position = n2n(safe_chain(positions,safe_chain(hf,8,'id')))
  homeForward9Age = getAge(getPlayer(players, safe_chain(hf,8,'id')),gameDate)
  homeForward9Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,8,'id')),0,'shootsCatches'))
  homeForward10 = safe_chain(hf,9,'id')
  homeForward10Position = n2n(safe_chain(positions,safe_chain(hf,9,'id')))
  homeForward10Age = getAge(getPlayer(players, safe_chain(hf,9,'id')),gameDate)
  homeForward10Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,9,'id')),0,'shootsCatches'))
  homeForward11 = safe_chain(hf,10,'id')
  homeForward11Position = n2n(safe_chain(positions,safe_chain(hf,10,'id')))
  homeForward11Age = getAge(getPlayer(players, safe_chain(hf,10,'id')),gameDate)
  homeForward11Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,10,'id')),0,'shootsCatches'))
  homeForward12 = safe_chain(hf,11,'id')
  homeForward12Position = n2n(safe_chain(positions,safe_chain(hf,11,'id')))
  homeForward12Age = getAge(getPlayer(players, safe_chain(hf,11,'id')),gameDate)
  homeForward12Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,11,'id')),0,'shootsCatches'))
  homeForward13 = safe_chain(hf,12,'id')
  homeForward13Position = n2n(safe_chain(positions,safe_chain(hf,12,'id')))
  homeForward13Age = getAge(getPlayer(players, safe_chain(hf,12,'id')),gameDate)
  homeForward13Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hf,12,'id')),0,'shootsCatches'))
  homeDefenseman1 = safe_chain(hd,0,'id')
  homeDefenseman1Position = n2n(safe_chain(positions,safe_chain(hd,0,'id')))
  homeDefenseman1Age = getAge(getPlayer(players, safe_chain(hd,0,'id')),gameDate)
  homeDefenseman1Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,0,'id')),0,'shootsCatches'))
  homeDefenseman2 = safe_chain(hd,1,'id')
  homeDefenseman2Position = n2n(safe_chain(positions,safe_chain(hd,1,'id')))
  homeDefenseman2Age = getAge(getPlayer(players, safe_chain(hd,1,'id')),gameDate)
  homeDefenseman2Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,1,'id')),0,'shootsCatches'))
  homeDefenseman3 = safe_chain(hd,2,'id')
  homeDefenseman3Position = n2n(safe_chain(positions,safe_chain(hd,2,'id')))
  homeDefenseman3Age = getAge(getPlayer(players, safe_chain(hd,2,'id')),gameDate)
  homeDefenseman3Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,2,'id')),0,'shootsCatches'))
  homeDefenseman4 = safe_chain(hd,3,'id')
  homeDefenseman4Position = n2n(safe_chain(positions,safe_chain(hd,3,'id')))
  homeDefenseman4Age = getAge(getPlayer(players, safe_chain(hd,3,'id')),gameDate)
  homeDefenseman4Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,3,'id')),0,'shootsCatches'))
  homeDefenseman5 = safe_chain(hd,4,'id')
  homeDefenseman5Position = n2n(safe_chain(positions,safe_chain(hd,4,'id')))
  homeDefenseman5Age = getAge(getPlayer(players, safe_chain(hd,4,'id')),gameDate)
  homeDefenseman5Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,4,'id')),0,'shootsCatches'))
  homeDefenseman6 = safe_chain(hd,5,'id')
  homeDefenseman6Position = n2n(safe_chain(positions,safe_chain(hd,5,'id')))
  homeDefenseman6Age = getAge(getPlayer(players, safe_chain(hd,5,'id')),gameDate)
  homeDefenseman6Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,5,'id')),0,'shootsCatches'))
  homeDefenseman7 = safe_chain(hd,6,'id')
  homeDefenseman7Position = n2n(safe_chain(positions,safe_chain(hd,6,'id')))
  homeDefenseman7Age = getAge(getPlayer(players, safe_chain(hd,6,'id')),gameDate)
  homeDefenseman7Shoots = n2n(safe_chain(getPlayer(players, safe_chain(hd,6,'id')),0,'shootsCatches'))
  homeStartingGoalie = safe_chain(hg,0,'id')
  homeStartingGoalieCatches = n2n(safe_chain(getPlayer(players, safe_chain(hg,0,'id')),0,'shootsCatches'))
  homeStartingGoalieAge = getAge(getPlayer(players, safe_chain(hg,0,'id')),gameDate)
  homeStartingGoalieHeight = safe_chain(getPlayer(players, safe_chain(hg,0,'id')),0,'heightInInches')
  homeStartingGoalieWeight = safe_chain(getPlayer(players, safe_chain(hg,0,'id')),0,'weightInPounds')
  homeBackupGoalie = safe_chain(hg,1,'id')
  homeBackupGoalieCatches = n2n(safe_chain(getPlayer(players, safe_chain(hg,1,'id')),0,'shootsCatches'))
  homeBackupGoalieAge = getAge(getPlayer(players, safe_chain(hg,1,'id')),gameDate)
  homeBackupGoalieHeight = safe_chain(getPlayer(players, safe_chain(hg,1,'id')),0,'heightInInches')
  homeBackupGoalieWeight = safe_chain(getPlayer(players, safe_chain(hg,1,'id')),0,'weightInPounds')
  homeThirdGoalie = safe_chain(hg,2,'id')
  homeThirdGoalieCatches = n2n(safe_chain(getPlayer(players, safe_chain(hg,2,'id')),0,'shootsCatches'))
  homeThirdGoalieAge = getAge(getPlayer(players, safe_chain(hg,2,'id')),gameDate)
  homeThirdGoalieHeight = safe_chain(getPlayer(players, safe_chain(hg,2,'id')),0,'heightInInches')
  homeThirdGoalieWeight = safe_chain(getPlayer(players, safe_chain(hg,2,'id')),0,'weightInPounds')

  if test:
    id = boxscore['id']
    season = boxscore['season']
    gameType = boxscore['gameType']
    venue = n2n(boxscore['venue']['default'])
    neutralSite = 0
    homeTeam = boxscore['homeTeam']['id']
    awayTeam = boxscore['awayTeam']['id']

    awayScore = boxscore['awayTeam']['score']
    homeScore = boxscore['homeTeam']['score']
    winner = homeTeam if homeScore > awayScore else awayTeam

    test_data = [[id,season,gameType,venue,neutralSite,homeTeam,awayTeam,startTime,date,awayHeadCoach,homeHeadCoach,ref1,ref2,linesman1,linesman1,awayForward1,
                 awayForward1Position,awayForward1Age,awayForward1Shoots,awayForward2,awayForward2Position,awayForward2Age,awayForward2Shoots,awayForward3,awayForward3Position,awayForward3Age,
                 awayForward3Shoots,awayForward4,awayForward4Position,awayForward4Age,awayForward4Shoots,awayForward5,awayForward5Position,awayForward5Age,awayForward5Shoots,awayForward6,
                 awayForward6Position,awayForward6Age,awayForward6Shoots,awayForward7,awayForward7Position,awayForward7Age,awayForward7Shoots,awayForward8,awayForward8Position,awayForward8Age,
                 awayForward8Shoots,awayForward9,awayForward9Position,awayForward9Age,awayForward9Shoots,awayForward10,awayForward10Position,awayForward10Age,awayForward10Shoots,awayForward11,
                 awayForward11Position,awayForward11Age,awayForward11Shoots,awayForward12,awayForward12Position,awayForward12Age,awayForward12Shoots,awayForward13,awayForward13Position,
                 awayForward13Age,awayForward13Shoots,awayDefenseman1,awayDefenseman1Position,awayDefenseman1Age,awayDefenseman1Shoots,awayDefenseman2,awayDefenseman2Position,awayDefenseman2Age,
                 awayDefenseman2Shoots,awayDefenseman3,awayDefenseman3Position,awayDefenseman3Age,awayDefenseman3Shoots,awayDefenseman4,awayDefenseman4Position,awayDefenseman4Age,
                 awayDefenseman4Shoots,awayDefenseman5,awayDefenseman5Position,awayDefenseman5Age,awayDefenseman5Shoots,awayDefenseman6,awayDefenseman6Position,awayDefenseman6Age,
                 awayDefenseman6Shoots,awayDefenseman7,awayDefenseman7Position,awayDefenseman7Age,awayDefenseman7Shoots,awayStartingGoalie,awayStartingGoalieCatches,awayStartingGoalieAge,
                 awayStartingGoalieHeight,awayStartingGoalieWeight,awayBackupGoalie,awayBackupGoalieCatches,awayBackupGoalieAge,awayBackupGoalieHeight,awayBackupGoalieWeight,awayThirdGoalie,
                 awayThirdGoalieCatches,awayThirdGoalieAge,awayThirdGoalieHeight,awayThirdGoalieWeight,homeForward1,homeForward1Position,homeForward1Age,homeForward1Shoots,homeForward2,
                 homeForward2Position,homeForward2Age,homeForward2Shoots,homeForward3,homeForward3Position,homeForward3Age,homeForward3Shoots,homeForward4,homeForward4Position,homeForward4Age,
                 homeForward4Shoots,homeForward5,homeForward5Position,homeForward5Age,homeForward5Shoots,homeForward6,homeForward6Position,homeForward6Age,homeForward6Shoots,homeForward7,
                 homeForward7Position,homeForward7Age,homeForward7Shoots,homeForward8,homeForward8Position,homeForward8Age,homeForward8Shoots,homeForward9,homeForward9Position,homeForward9Age,
                 homeForward9Shoots,homeForward10,homeForward10Position,homeForward10Age,homeForward10Shoots,homeForward11,homeForward11Position,homeForward11Age,homeForward11Shoots,
                 homeForward12,homeForward12Position,homeForward12Age,homeForward12Shoots,homeForward13,homeForward13Position,homeForward13Age,homeForward13Shoots,homeDefenseman1,
                 homeDefenseman1Position,homeDefenseman1Age,homeDefenseman1Shoots,homeDefenseman2,homeDefenseman2Position,homeDefenseman2Age,homeDefenseman2Shoots,homeDefenseman3,
                 homeDefenseman3Position,homeDefenseman3Age,homeDefenseman3Shoots,homeDefenseman4,homeDefenseman4Position,homeDefenseman4Age,homeDefenseman4Shoots,homeDefenseman5,
                 homeDefenseman5Position,homeDefenseman5Age,homeDefenseman5Shoots,homeDefenseman6,homeDefenseman6Position,homeDefenseman6Age,homeDefenseman6Shoots,homeDefenseman7,
                 homeDefenseman7Position,homeDefenseman7Age,homeDefenseman7Shoots,homeStartingGoalie,homeStartingGoalieCatches,homeStartingGoalieAge,homeStartingGoalieHeight,
                 homeStartingGoalieWeight,homeBackupGoalie,homeBackupGoalieCatches,homeBackupGoalieAge,homeBackupGoalieHeight,homeBackupGoalieWeight,homeThirdGoalie,homeThirdGoalieCatches,
                 homeThirdGoalieAge,homeThirdGoalieHeight,homeThirdGoalieWeight]]
    test_result = [[homeScore,awayScore,winner]]
    return {'data':test_data,'result':test_result}
  else:
    id = game['id']
    season = game['season']
    gameType = game['gameType']
    venue = n2n(game['venue']['default'])
    neutralSite = 1 if game['neutralSite'] else 0
    homeTeam = game['homeTeam']['id']
    awayTeam = game['awayTeam']['id']

    check_data = [[id,season,gameType,venue,neutralSite,homeTeam,awayTeam,startTime,date,awayHeadCoach,homeHeadCoach,ref1,ref2,linesman1,linesman1,awayForward1,
                  awayForward1Position,awayForward1Age,awayForward1Shoots,awayForward2,awayForward2Position,awayForward2Age,awayForward2Shoots,awayForward3,awayForward3Position,awayForward3Age,
                  awayForward3Shoots,awayForward4,awayForward4Position,awayForward4Age,awayForward4Shoots,awayForward5,awayForward5Position,awayForward5Age,awayForward5Shoots,awayForward6,
                  awayForward6Position,awayForward6Age,awayForward6Shoots,awayForward7,awayForward7Position,awayForward7Age,awayForward7Shoots,awayForward8,awayForward8Position,awayForward8Age,
                  awayForward8Shoots,awayForward9,awayForward9Position,awayForward9Age,awayForward9Shoots,awayForward10,awayForward10Position,awayForward10Age,awayForward10Shoots,awayForward11,
                  awayForward11Position,awayForward11Age,awayForward11Shoots,awayForward12,awayForward12Position,awayForward12Age,awayForward12Shoots,awayForward13,awayForward13Position,
                  awayForward13Age,awayForward13Shoots,awayDefenseman1,awayDefenseman1Position,awayDefenseman1Age,awayDefenseman1Shoots,awayDefenseman2,awayDefenseman2Position,awayDefenseman2Age,
                  awayDefenseman2Shoots,awayDefenseman3,awayDefenseman3Position,awayDefenseman3Age,awayDefenseman3Shoots,awayDefenseman4,awayDefenseman4Position,awayDefenseman4Age,
                  awayDefenseman4Shoots,awayDefenseman5,awayDefenseman5Position,awayDefenseman5Age,awayDefenseman5Shoots,awayDefenseman6,awayDefenseman6Position,awayDefenseman6Age,
                  awayDefenseman6Shoots,awayDefenseman7,awayDefenseman7Position,awayDefenseman7Age,awayDefenseman7Shoots,awayStartingGoalie,awayStartingGoalieCatches,awayStartingGoalieAge,
                  awayStartingGoalieHeight,awayStartingGoalieWeight,awayBackupGoalie,awayBackupGoalieCatches,awayBackupGoalieAge,awayBackupGoalieHeight,awayBackupGoalieWeight,awayThirdGoalie,
                  awayThirdGoalieCatches,awayThirdGoalieAge,awayThirdGoalieHeight,awayThirdGoalieWeight,homeForward1,homeForward1Position,homeForward1Age,homeForward1Shoots,homeForward2,
                  homeForward2Position,homeForward2Age,homeForward2Shoots,homeForward3,homeForward3Position,homeForward3Age,homeForward3Shoots,homeForward4,homeForward4Position,homeForward4Age,
                  homeForward4Shoots,homeForward5,homeForward5Position,homeForward5Age,homeForward5Shoots,homeForward6,homeForward6Position,homeForward6Age,homeForward6Shoots,homeForward7,
                  homeForward7Position,homeForward7Age,homeForward7Shoots,homeForward8,homeForward8Position,homeForward8Age,homeForward8Shoots,homeForward9,homeForward9Position,homeForward9Age,
                  homeForward9Shoots,homeForward10,homeForward10Position,homeForward10Age,homeForward10Shoots,homeForward11,homeForward11Position,homeForward11Age,homeForward11Shoots,
                  homeForward12,homeForward12Position,homeForward12Age,homeForward12Shoots,homeForward13,homeForward13Position,homeForward13Age,homeForward13Shoots,homeDefenseman1,
                  homeDefenseman1Position,homeDefenseman1Age,homeDefenseman1Shoots,homeDefenseman2,homeDefenseman2Position,homeDefenseman2Age,homeDefenseman2Shoots,homeDefenseman3,
                  homeDefenseman3Position,homeDefenseman3Age,homeDefenseman3Shoots,homeDefenseman4,homeDefenseman4Position,homeDefenseman4Age,homeDefenseman4Shoots,homeDefenseman5,
                  homeDefenseman5Position,homeDefenseman5Age,homeDefenseman5Shoots,homeDefenseman6,homeDefenseman6Position,homeDefenseman6Age,homeDefenseman6Shoots,homeDefenseman7,
                  homeDefenseman7Position,homeDefenseman7Age,homeDefenseman7Shoots,homeStartingGoalie,homeStartingGoalieCatches,homeStartingGoalieAge,homeStartingGoalieHeight,
                  homeStartingGoalieWeight,homeBackupGoalie,homeBackupGoalieCatches,homeBackupGoalieAge,homeBackupGoalieHeight,homeBackupGoalieWeight,homeThirdGoalie,homeThirdGoalieCatches,
                  homeThirdGoalieAge,homeThirdGoalieHeight,homeThirdGoalieWeight]]
  
    period = boxscore['period']

    if boxscore['periodDescriptor']['periodType'] != 'REG':
      period = boxscore['periodDescriptor']['periodType']

    return {
      'data': {
        'data': check_data,
        'game_id': id,
        'date': boxscore['gameDate'],
        'state': boxscore['gameState'],
        'home_team': {
          'id': homeTeam,
          'city': game['homeTeam']['placeName']['default'],
          'name': boxscore['homeTeam']['name']['default'],
          'abbreviation': boxscore['homeTeam']['abbrev'],
        },
        'away_team': {
          'id': awayTeam,
          'city': game['awayTeam']['placeName']['default'],
          'name': boxscore['awayTeam']['name']['default'],
          'abbreviation': boxscore['awayTeam']['abbrev'],
        },
        'live': {
          'home_score': boxscore['homeTeam']['score'],
          'away_score': boxscore['awayTeam']['score'],
          'period': period,
          'clock': boxscore['clock']['timeRemaining'],
          'stopped': not boxscore['clock']['running'],
          'intermission': boxscore['clock']['inIntermission'],
        },
      },
      'message': '',
    }

def nhl_ai(game_data):

  game = game_data

  boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
  check_pbgs = false_chain(boxscore,'boxscore','playerByGameStats')
  if not check_pbgs:
    return {
      'data': {
        'data': [[]],
        'date': -1,
        'game_id': game['id'],
        'home_team': {
          'id': game['homeTeam']['id'],
          'city': game['homeTeam']['placeName']['default'],
          'name': -1,
          'abbreviation': -1,
        },
        'away_team': {
          'id': game['awayTeam']['id'],
          'city': game['awayTeam']['placeName']['default'],
          'name': -1,
          'abbreviation': -1,
        },
        'live': {
          'home_score': -1,
          'away_score': -1,
          'period': 0,
          'clock': 0,
          'stopped': True,
          'intermission': False,
        },
      },
      'message': 'no boxscore for game',
    }
  
  return nhl_data(game=game,boxscore=boxscore,test=False)

def nhl_test(boxscore):
  return nhl_data(boxscore=boxscore,test=True)