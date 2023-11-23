import requests
from pymongo import MongoClient
import math
from joblib import load
from datetime import datetime

REPLACE_VALUE = -1

def n2n(in_name):
  try:
    if in_name and not isNaN(in_name):
      char_to_num = {'b':1,'c':2,'d':3,'f':4,'g':5,'h':6,'j':7,'k':8,
                     'l':9,'m':10,'n':11,'p':12,'q':13,'r':14,'s':15,
                     't':16,'v':17,'w':18,'x':19}
      nums = []
      for char in str(in_name).lower().replace(' ', ''):
        if char in char_to_num.keys():
          nums.append(char_to_num[char])
      # return int(''.join(str(char_to_num[char]) for char in in_name.lower().replace(' ', '')))
      return int(''.join(str(char) for char in nums))
    else:
      if (not isNaN(in_name)):
        print(in_name)
      return REPLACE_VALUE
  except Exception as error:
    return REPLACE_VALUE

def isNaN(inCheck):
  try:
    float_value = float(inCheck)
    return math.isnan(float_value)
  except (ValueError, TypeError):
    return False

def getAge(player, game_date):
  try:
    if isNaN(player) or isNaN(game_date):
      return REPLACE_VALUE
    birthday = player['birthDate']
    b_year, b_month, b_day = map(int, birthday.split('-'))
    g_year, g_month, g_day = map(int, game_date.split('-'))
    age = g_year - b_year
    if (b_month > g_month) or (b_month == g_month and b_day > g_day):
      age -= 1

    return age

  except Exception as error:
    return REPLACE_VALUE

def getPlayerData (players, player_id):
  if not isNaN(players) and not isNaN(player_id):
    return next((player for player in players if player['playerId'] == player_id), REPLACE_VALUE)
  else:
    return REPLACE_VALUE

def safe_chain(obj, *keys, default=REPLACE_VALUE):
  for key in keys:
    try:
      obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
    except (KeyError, TypeError, AttributeError):
      return default
  return obj

def false_chain(obj, *keys, default=False):
  for key in keys:
    try:
      obj = getattr(obj, key, default) if hasattr(obj, key) else obj[key]
    except (KeyError, TypeError, AttributeError):
      return default
  return obj

def getPlayer(allPlayers,playerId):
  if not isNaN(allPlayers) and not isNaN(playerId):
    return [p for p in allPlayers if p['playerId'] == playerId]
  else:
    return REPLACE_VALUE

def format_start_time(game):
  startTime = REPLACE_VALUE
  if 'startTimeUTC' in game and game['startTimeUTC']:
    st = datetime.strptime(game['startTimeUTC'], "%Y-%m-%dT%H:%M:%SZ")  # Adjust the format if necessary
    startTime = int(f"{st.hour}{st.minute:02d}")
  return startTime

def format_game_date(game):
  date = REPLACE_VALUE
  if 'gameDate' in game and game['gameDate']:
    date = int(game['gameDate'].replace('-', ''))
  return date

def nhl_ai(game_data):
  client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
  db = client['hockey']
  Players = db['dev_players']
  # Teams = db['dev_teams']

  game = game_data

  # boxscore1 = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game1['id']}/boxscore").json()
  # boxscore2 = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game2['id']}/boxscore").json()

  boxscore = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{game['id']}/boxscore").json()
  pbgs = boxscore['boxscore']['playerByGameStats']
  af = [p['playerId'] for p in pbgs['awayTeam']['forwards'] if 'playerId' in p]
  ad = [p['playerId'] for p in pbgs['awayTeam']['defense'] if 'playerId' in p]
  ag = [p['playerId'] for p in pbgs['awayTeam']['goalies'] if 'playerId' in p]
  hf = [p['playerId'] for p in pbgs['homeTeam']['forwards'] if 'playerId' in p]
  hd = [p['playerId'] for p in pbgs['homeTeam']['defense'] if 'playerId' in p]
  hg = [p['playerId'] for p in pbgs['homeTeam']['goalies'] if 'playerId' in p]
  a = []
  for p in af: a.append({'playerId':p})
  for p in ad: a.append({'playerId':p})
  for p in ag: a.append({'playerId':p})
  for p in hf: a.append({'playerId':p})
  for p in hd: a.append({'playerId':p})
  for p in hg: a.append({'playerId':p})

  players = Players.find(
    {'$or': a},
    {'_id':0, 'playerId':1, 'shootsCatches':1, 'birthDate': 1, 'position': 1}
  )

  gameDate = format_game_date(game)
  gameStartTime = format_start_time(game)

  id = game['id']
  season = game['season']
  gameType = game['gameType']
  venue = n2n(game['venue']['default'])
  neutralSite = 1 if game['neutralSite'] else 0
  homeTeam = game['homeTeam']['id']
  awayTeam = game['awayTeam']['id']
  awaySplitSquad = 1 if game['awayTeam']['awaySplitSquad'] else 0
  homeSplitSquad = 1 if game['homeTeam']['homeSplitSquad'] else 0
  startTime = gameStartTime
  date = gameDate
  awayHeadCoach = n2n(boxscore['boxscore']['gameInfo']['awayTeam']['headCoach']['default'])
  homeHeadCoach = n2n(boxscore['boxscore']['gameInfo']['homeTeam']['headCoach']['default'])
  ref1 = n2n(boxscore['boxscore']['gameInfo']['referees'][0]['default'])
  ref2 = n2n(boxscore['boxscore']['gameInfo']['referees'][1]['default'])
  linesman1 = n2n(boxscore['boxscore']['gameInfo']['linesmen'][0]['default'])
  linesman1 = n2n(boxscore['boxscore']['gameInfo']['linesmen'][1]['default'])
  awayForward1 = safe_chain(af,0)
  awayForward1Position = safe_chain(getPlayer(players, safe_chain(af,0)),'position')
  awayForward1Age = getAge(getPlayer(players, safe_chain(af,0)),gameDate)
  awayForward1Shoots = safe_chain(getPlayer(players, safe_chain(af,0)),'shootsCatches')
  awayForward2 = safe_chain(af,1)
  awayForward2Position = safe_chain(getPlayer(players, safe_chain(af,1)),'position')
  awayForward2Age = getAge(getPlayer(players, safe_chain(af,1)),gameDate)
  awayForward2Shoots = safe_chain(getPlayer(players, safe_chain(af,1)),'shootsCatches')
  awayForward3 = safe_chain(af,2)
  awayForward3Position = safe_chain(getPlayer(players, safe_chain(af,2)),'position')
  awayForward3Age = getAge(getPlayer(players, safe_chain(af,2)),gameDate)
  awayForward3Shoots = safe_chain(getPlayer(players, safe_chain(af,2)),'shootsCatches')
  awayForward4 = safe_chain(af,3)
  awayForward4Position = safe_chain(getPlayer(players, safe_chain(af,3)),'position')
  awayForward4Age = getAge(getPlayer(players, safe_chain(af,3)),gameDate)
  awayForward4Shoots = safe_chain(getPlayer(players, safe_chain(af,3)),'shootsCatches')
  awayForward5 = safe_chain(af,4)
  awayForward5Position = safe_chain(getPlayer(players, safe_chain(af,4)),'position')
  awayForward5Age = getAge(getPlayer(players, safe_chain(af,4)),gameDate)
  awayForward5Shoots = safe_chain(getPlayer(players, safe_chain(af,4)),'shootsCatches')
  awayForward6 = safe_chain(af,5)
  awayForward6Position = safe_chain(getPlayer(players, safe_chain(af,5)),'position')
  awayForward6Age = getAge(getPlayer(players, safe_chain(af,5)),gameDate)
  awayForward6Shoots = safe_chain(getPlayer(players, safe_chain(af,5)),'shootsCatches')
  awayForward7 = safe_chain(af,6)
  awayForward7Position = safe_chain(getPlayer(players, safe_chain(af,6)),'position')
  awayForward7Age = getAge(getPlayer(players, safe_chain(af,6)),gameDate)
  awayForward7Shoots = safe_chain(getPlayer(players, safe_chain(af,6)),'shootsCatches')
  awayForward8 = safe_chain(af,7)
  awayForward8Position = safe_chain(getPlayer(players, safe_chain(af,7)),'position')
  awayForward8Age = getAge(getPlayer(players, safe_chain(af,7)),gameDate)
  awayForward8Shoots = safe_chain(getPlayer(players, safe_chain(af,7)),'shootsCatches')
  awayForward9 = safe_chain(af,8)
  awayForward9Position = safe_chain(getPlayer(players, safe_chain(af,8)),'position')
  awayForward9Age = getAge(getPlayer(players, safe_chain(af,8)),gameDate)
  awayForward9Shoots = safe_chain(getPlayer(players, safe_chain(af,8)),'shootsCatches')
  awayForward10 = safe_chain(af,9)
  awayForward10Position = safe_chain(getPlayer(players, safe_chain(af,9)),'position')
  awayForward10Age = getAge(getPlayer(players, safe_chain(af,9)),gameDate)
  awayForward10Shoots = safe_chain(getPlayer(players, safe_chain(af,9)),'shootsCatches')
  awayForward11 = safe_chain(af,10)
  awayForward11Position = safe_chain(getPlayer(players, safe_chain(af,10)),'position')
  awayForward11Age = getAge(getPlayer(players, safe_chain(af,10)),gameDate)
  awayForward11Shoots = safe_chain(getPlayer(players, safe_chain(af,10)),'shootsCatches')
  awayForward12 = safe_chain(af,11)
  awayForward12Position = safe_chain(getPlayer(players, safe_chain(af,11)),'position')
  awayForward12Age = getAge(getPlayer(players, safe_chain(af,11)),gameDate)
  awayForward12Shoots = safe_chain(getPlayer(players, safe_chain(af,11)),'shootsCatches')
  awayForward13 = safe_chain(af,12)
  awayForward13Position = safe_chain(getPlayer(players, safe_chain(af,12)),'position')
  awayForward13Age = getAge(getPlayer(players, safe_chain(af,12)),gameDate)
  awayForward13Shoots = safe_chain(getPlayer(players, safe_chain(af,12)),'shootsCatches')
  awayDefenseman1 = safe_chain(ad,0)
  awayDefenseman1Position = safe_chain(getPlayer(players, safe_chain(ad,0)),'position')
  awayDefenseman1Age = getAge(getPlayer(players, safe_chain(ad,0)),gameDate)
  awayDefenseman1Shoots = safe_chain(getPlayer(players, safe_chain(ad,0)),'shootsCatches')
  awayDefenseman2 = safe_chain(ad,1)
  awayDefenseman2Position = safe_chain(getPlayer(players, safe_chain(ad,1)),'position')
  awayDefenseman2Age = getAge(getPlayer(players, safe_chain(ad,1)),gameDate)
  awayDefenseman2Shoots = safe_chain(getPlayer(players, safe_chain(ad,1)),'shootsCatches')
  awayDefenseman3 = safe_chain(ad,2)
  awayDefenseman3Position = safe_chain(getPlayer(players, safe_chain(ad,2)),'position')
  awayDefenseman3Age = getAge(getPlayer(players, safe_chain(ad,2)),gameDate)
  awayDefenseman3Shoots = safe_chain(getPlayer(players, safe_chain(ad,2)),'shootsCatches')
  awayDefenseman4 = safe_chain(ad,3)
  awayDefenseman4Position = safe_chain(getPlayer(players, safe_chain(ad,3)),'position')
  awayDefenseman4Age = getAge(getPlayer(players, safe_chain(ad,3)),gameDate)
  awayDefenseman4Shoots = safe_chain(getPlayer(players, safe_chain(ad,3)),'shootsCatches')
  awayDefenseman5 = safe_chain(ad,4)
  awayDefenseman5Position = safe_chain(getPlayer(players, safe_chain(ad,4)),'position')
  awayDefenseman5Age = getAge(getPlayer(players, safe_chain(ad,4)),gameDate)
  awayDefenseman5Shoots = safe_chain(getPlayer(players, safe_chain(ad,4)),'shootsCatches')
  awayDefenseman6 = safe_chain(ad,5)
  awayDefenseman6Position = safe_chain(getPlayer(players, safe_chain(ad,5)),'position')
  awayDefenseman6Age = getAge(getPlayer(players, safe_chain(ad,5)),gameDate)
  awayDefenseman6Shoots = safe_chain(getPlayer(players, safe_chain(ad,5)),'shootsCatches')
  awayDefenseman7 = safe_chain(ad,6)
  awayDefenseman7Position = safe_chain(getPlayer(players, safe_chain(ad,6)),'position')
  awayDefenseman7Age = getAge(getPlayer(players, safe_chain(ad,6)),gameDate)
  awayDefenseman7Shoots = safe_chain(getPlayer(players, safe_chain(ad,6)),'shootsCatches')
  awayStartingGoalie = safe_chain(ag,0)
  awayStartingGoalieCatches = safe_chain(getPlayer(players, safe_chain(ag,0)),'shootsCatches')
  awayStartingGoalieAge = getAge(getPlayer(players, safe_chain(ag,0)),gameDate)
  awayStartingGoalieHeight = safe_chain(getPlayer(players, safe_chain(ag,0)),'heightInInches')
  awayStartingGoalieWeight = safe_chain(getPlayer(players, safe_chain(ag,0)),'weightInPounds')
  awayBackupGoalie = safe_chain(ag,1)
  awayBackupGoalieCatches = safe_chain(getPlayer(players, safe_chain(ag,1)),'shootsCatches')
  awayBackupGoalieAge = getAge(getPlayer(players, safe_chain(ag,1)),gameDate)
  awayBackupGoalieHeight = safe_chain(getPlayer(players, safe_chain(ag,1)),'heightInInches')
  awayBackupGoalieWeight = safe_chain(getPlayer(players, safe_chain(ag,1)),'weightInPounds')
  awayThirdGoalie = safe_chain(ag,2)
  awayThirdGoalieCatches = safe_chain(getPlayer(players, safe_chain(ag,2)),'shootsCatches')
  awayThirdGoalieAge = getAge(getPlayer(players, safe_chain(ag,2)),gameDate)
  awayThirdGoalieHeight = safe_chain(getPlayer(players, safe_chain(ag,2)),'heightInInches')
  awayThirdGoalieWeight = safe_chain(getPlayer(players, safe_chain(ag,2)),'weightInPounds')
  homeForward1 = safe_chain(hf,0)
  homeForward1Position = safe_chain(getPlayer(players, safe_chain(hf,0)),'position')
  homeForward1Age = getAge(getPlayer(players, safe_chain(hf,0)),gameDate)
  homeForward1Shoots = safe_chain(getPlayer(players, safe_chain(hf,0)),'shootsCatches')
  homeForward2 = safe_chain(hf,1)
  homeForward2Position = safe_chain(getPlayer(players, safe_chain(hf,1)),'position')
  homeForward2Age = getAge(getPlayer(players, safe_chain(hf,1)),gameDate)
  homeForward2Shoots = safe_chain(getPlayer(players, safe_chain(hf,1)),'shootsCatches')
  homeForward3 = safe_chain(hf,2)
  homeForward3Position = safe_chain(getPlayer(players, safe_chain(hf,2)),'position')
  homeForward3Age = getAge(getPlayer(players, safe_chain(hf,2)),gameDate)
  homeForward3Shoots = safe_chain(getPlayer(players, safe_chain(hf,2)),'shootsCatches')
  homeForward4 = safe_chain(hf,3)
  homeForward4Position = safe_chain(getPlayer(players, safe_chain(hf,3)),'position')
  homeForward4Age = getAge(getPlayer(players, safe_chain(hf,3)),gameDate)
  homeForward4Shoots = safe_chain(getPlayer(players, safe_chain(hf,3)),'shootsCatches')
  homeForward5 = safe_chain(hf,4)
  homeForward5Position = safe_chain(getPlayer(players, safe_chain(hf,4)),'position')
  homeForward5Age = getAge(getPlayer(players, safe_chain(hf,4)),gameDate)
  homeForward5Shoots = safe_chain(getPlayer(players, safe_chain(hf,4)),'shootsCatches')
  homeForward6 = safe_chain(hf,5)
  homeForward6Position = safe_chain(getPlayer(players, safe_chain(hf,5)),'position')
  homeForward6Age = getAge(getPlayer(players, safe_chain(hf,5)),gameDate)
  homeForward6Shoots = safe_chain(getPlayer(players, safe_chain(hf,5)),'shootsCatches')
  homeForward7 = safe_chain(hf,6)
  homeForward7Position = safe_chain(getPlayer(players, safe_chain(hf,6)),'position')
  homeForward7Age = getAge(getPlayer(players, safe_chain(hf,6)),gameDate)
  homeForward7Shoots = safe_chain(getPlayer(players, safe_chain(hf,6)),'shootsCatches')
  homeForward8 = safe_chain(hf,7)
  homeForward8Position = safe_chain(getPlayer(players, safe_chain(hf,7)),'position')
  homeForward8Age = getAge(getPlayer(players, safe_chain(hf,7)),gameDate)
  homeForward8Shoots = safe_chain(getPlayer(players, safe_chain(hf,7)),'shootsCatches')
  homeForward9 = safe_chain(hf,8)
  homeForward9Position = safe_chain(getPlayer(players, safe_chain(hf,8)),'position')
  homeForward9Age = getAge(getPlayer(players, safe_chain(hf,8)),gameDate)
  homeForward9Shoots = safe_chain(getPlayer(players, safe_chain(hf,8)),'shootsCatches')
  homeForward10 = safe_chain(hf,9)
  homeForward10Position = safe_chain(getPlayer(players, safe_chain(hf,9)),'position')
  homeForward10Age = getAge(getPlayer(players, safe_chain(hf,9)),gameDate)
  homeForward10Shoots = safe_chain(getPlayer(players, safe_chain(hf,9)),'shootsCatches')
  homeForward11 = safe_chain(hf,10)
  homeForward11Position = safe_chain(getPlayer(players, safe_chain(hf,10)),'position')
  homeForward11Age = getAge(getPlayer(players, safe_chain(hf,10)),gameDate)
  homeForward11Shoots = safe_chain(getPlayer(players, safe_chain(hf,10)),'shootsCatches')
  homeForward12 = safe_chain(hf,11)
  homeForward12Position = safe_chain(getPlayer(players, safe_chain(hf,11)),'position')
  homeForward12Age = getAge(getPlayer(players, safe_chain(hf,11)),gameDate)
  homeForward12Shoots = safe_chain(getPlayer(players, safe_chain(hf,11)),'shootsCatches')
  homeForward13 = safe_chain(hf,12)
  homeForward13Position = safe_chain(getPlayer(players, safe_chain(hf,12)),'position')
  homeForward13Age = getAge(getPlayer(players, safe_chain(hf,12)),gameDate)
  homeForward13Shoots = safe_chain(getPlayer(players, safe_chain(hf,12)),'shootsCatches')
  homeDefenseman1 = safe_chain(hd,0)
  homeDefenseman1Position = safe_chain(getPlayer(players, safe_chain(hd,0)),'position')
  homeDefenseman1Age = getAge(getPlayer(players, safe_chain(hd,0)),gameDate)
  homeDefenseman1Shoots = safe_chain(getPlayer(players, safe_chain(hd,0)),'shootsCatches')
  homeDefenseman2 = safe_chain(hd,1)
  homeDefenseman2Position = safe_chain(getPlayer(players, safe_chain(hd,1)),'position')
  homeDefenseman2Age = getAge(getPlayer(players, safe_chain(hd,1)),gameDate)
  homeDefenseman2Shoots = safe_chain(getPlayer(players, safe_chain(hd,1)),'shootsCatches')
  homeDefenseman3 = safe_chain(hd,2)
  homeDefenseman3Position = safe_chain(getPlayer(players, safe_chain(hd,2)),'position')
  homeDefenseman3Age = getAge(getPlayer(players, safe_chain(hd,2)),gameDate)
  homeDefenseman3Shoots = safe_chain(getPlayer(players, safe_chain(hd,2)),'shootsCatches')
  homeDefenseman4 = safe_chain(hd,3)
  homeDefenseman4Position = safe_chain(getPlayer(players, safe_chain(hd,3)),'position')
  homeDefenseman4Age = getAge(getPlayer(players, safe_chain(hd,3)),gameDate)
  homeDefenseman4Shoots = safe_chain(getPlayer(players, safe_chain(hd,3)),'shootsCatches')
  homeDefenseman5 = safe_chain(hd,4)
  homeDefenseman5Position = safe_chain(getPlayer(players, safe_chain(hd,4)),'position')
  homeDefenseman5Age = getAge(getPlayer(players, safe_chain(hd,4)),gameDate)
  homeDefenseman5Shoots = safe_chain(getPlayer(players, safe_chain(hd,4)),'shootsCatches')
  homeDefenseman6 = safe_chain(hd,5)
  homeDefenseman6Position = safe_chain(getPlayer(players, safe_chain(hd,5)),'position')
  homeDefenseman6Age = getAge(getPlayer(players, safe_chain(hd,5)),gameDate)
  homeDefenseman6Shoots = safe_chain(getPlayer(players, safe_chain(hd,5)),'shootsCatches')
  homeDefenseman7 = safe_chain(hd,6)
  homeDefenseman7Position = safe_chain(getPlayer(players, safe_chain(hd,6)),'position')
  homeDefenseman7Age = getAge(getPlayer(players, safe_chain(hd,6)),gameDate)
  homeDefenseman7Shoots = safe_chain(getPlayer(players, safe_chain(hd,6)),'shootsCatches')
  homeStartingGoalie = safe_chain(hg,0)
  homeStartingGoalieCatches = safe_chain(getPlayer(players, safe_chain(hg,0)),'shootsCatches')
  homeStartingGoalieAge = getAge(getPlayer(players, safe_chain(hg,0)),gameDate)
  homeStartingGoalieHeight = safe_chain(getPlayer(players, safe_chain(hg,0)),'heightInInches')
  homeStartingGoalieWeight = safe_chain(getPlayer(players, safe_chain(hg,0)),'weightInPounds')
  homeBackupGoalie = safe_chain(hg,1)
  homeBackupGoalieCatches = safe_chain(getPlayer(players, safe_chain(hg,1)),'shootsCatches')
  homeBackupGoalieAge = getAge(getPlayer(players, safe_chain(hg,1)),gameDate)
  homeBackupGoalieHeight = safe_chain(getPlayer(players, safe_chain(hg,1)),'heightInInches')
  homeBackupGoalieWeight = safe_chain(getPlayer(players, safe_chain(hg,1)),'weightInPounds')
  homeThirdGoalie = safe_chain(hg,2)
  homeThirdGoalieCatches = safe_chain(getPlayer(players, safe_chain(hg,2)),'shootsCatches')
  homeThirdGoalieAge = getAge(getPlayer(players, safe_chain(hg,2)),gameDate)
  homeThirdGoalieHeight = safe_chain(getPlayer(players, safe_chain(hg,2)),'heightInInches')
  homeThirdGoalieWeight = safe_chain(getPlayer(players, safe_chain(hg,2)),'weightInPounds')

  check_data = [[id,season,gameType,venue,neutralSite,homeTeam,awayTeam,
                 awaySplitSquad,homeSplitSquad,startTime,date,
                 awayHeadCoach,homeHeadCoach,ref1,ref2,linesman1,linesman1,
                 awayForward1,awayForward1Position,awayForward1Age,
                 awayForward1Shoots,awayForward2,awayForward2Position,
                 awayForward2Age,awayForward2Shoots,awayForward3,
                 awayForward3Position,awayForward3Age,awayForward3Shoots,
                 awayForward4,awayForward4Position,awayForward4Age,
                 awayForward4Shoots,awayForward5,awayForward5Position,
                 awayForward5Age,awayForward5Shoots,awayForward6,
                 awayForward6Position,awayForward6Age,awayForward6Shoots,
                 awayForward7,awayForward7Position,awayForward7Age,
                 awayForward7Shoots,awayForward8,awayForward8Position,
                 awayForward8Age,awayForward8Shoots,awayForward9,
                 awayForward9Position,awayForward9Age,awayForward9Shoots,
                 awayForward10,awayForward10Position,awayForward10Age,
                 awayForward10Shoots,awayForward11,awayForward11Position,
                 awayForward11Age,awayForward11Shoots,awayForward12,
                 awayForward12Position,awayForward12Age,
                 awayForward12Shoots,awayForward13,awayForward13Position,
                 awayForward13Age,awayForward13Shoots,awayDefenseman1,
                 awayDefenseman1Position,awayDefenseman1Age,awayDefenseman1Shoots,
                 awayDefenseman2,awayDefenseman2Position,awayDefenseman2Age,
                 awayDefenseman2Shoots,awayDefenseman3,awayDefenseman3Position,
                 awayDefenseman3Age,awayDefenseman3Shoots,awayDefenseman4,
                 awayDefenseman4Position,awayDefenseman4Age,awayDefenseman4Shoots,
                 awayDefenseman5,awayDefenseman5Position,awayDefenseman5Age,
                 awayDefenseman5Shoots,awayDefenseman6,awayDefenseman6Position,
                 awayDefenseman6Age,awayDefenseman6Shoots,awayDefenseman7,
                 awayDefenseman7Position,awayDefenseman7Age,awayDefenseman7Shoots,
                 awayStartingGoalie,awayStartingGoalieCatches,awayStartingGoalieAge,
                 awayStartingGoalieHeight,awayStartingGoalieWeight,awayBackupGoalie,
                 awayBackupGoalieCatches,awayBackupGoalieAge,awayBackupGoalieHeight,
                 awayBackupGoalieWeight,awayThirdGoalie,awayThirdGoalieCatches,
                 awayThirdGoalieAge,awayThirdGoalieHeight,awayThirdGoalieWeight,
                 homeForward1,homeForward1Position,homeForward1Age,homeForward1Shoots,
                 homeForward2,homeForward2Position,homeForward2Age,homeForward2Shoots,
                 homeForward3,homeForward3Position,homeForward3Age,homeForward3Shoots,
                 homeForward4,homeForward4Position,homeForward4Age,homeForward4Shoots,
                 homeForward5,homeForward5Position,homeForward5Age,homeForward5Shoots,
                 homeForward6,homeForward6Position,homeForward6Age,homeForward6Shoots,
                 homeForward7,homeForward7Position,homeForward7Age,homeForward7Shoots,
                 homeForward8,homeForward8Position,homeForward8Age,homeForward8Shoots,
                 homeForward9,homeForward9Position,homeForward9Age,homeForward9Shoots,
                 homeForward10,homeForward10Position,homeForward10Age,
                 homeForward10Shoots,homeForward11,homeForward11Position,
                 homeForward11Age,homeForward11Shoots,homeForward12,
                 homeForward12Position,homeForward12Age,homeForward12Shoots,
                 homeForward13,homeForward13Position,homeForward13Age,
                 homeForward13Shoots,homeDefenseman1,homeDefenseman1Position,
                 homeDefenseman1Age,homeDefenseman1Shoots,homeDefenseman2,
                 homeDefenseman2Position,homeDefenseman2Age,homeDefenseman2Shoots,
                 homeDefenseman3,homeDefenseman3Position,homeDefenseman3Age,
                 homeDefenseman3Shoots,homeDefenseman4,homeDefenseman4Position,
                 homeDefenseman4Age,homeDefenseman4Shoots,homeDefenseman5,
                 homeDefenseman5Position,homeDefenseman5Age,homeDefenseman5Shoots,
                 homeDefenseman6,homeDefenseman6Position,homeDefenseman6Age,
                 homeDefenseman6Shoots,homeDefenseman7,homeDefenseman7Position,
                 homeDefenseman7Age,homeDefenseman7Shoots,homeStartingGoalie,
                 homeStartingGoalieCatches,homeStartingGoalieAge,
                 homeStartingGoalieHeight,homeStartingGoalieWeight,
                 homeBackupGoalie,homeBackupGoalieCatches,homeBackupGoalieAge,
                 homeBackupGoalieHeight,homeBackupGoalieWeight,homeThirdGoalie,
                 homeThirdGoalieCatches,homeThirdGoalieAge,homeThirdGoalieHeight,
                 homeThirdGoalieWeight]]
  return {
    'data': check_data,
    'game_id': id,
    'home_team_id': homeTeam,
    'away_team_id': awayTeam,
    'home_team': game['homeTeam']['placeName']['default'],
    'away_team': game['awayTeam']['placeName']['default'],
  }