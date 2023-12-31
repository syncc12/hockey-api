import math
import requests
from pymongo import MongoClient
from datetime import datetime
import numpy as np

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

def getPlayerData (players, player_id):
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


def compile_training_data(db, game):
  Players = db['dev_players']
  startTime = REPLACE_VALUE
  if 'startTimeUTC' in game and game['startTimeUTC']:
    st = datetime.strptime(game['startTimeUTC'], "%Y-%m-%dT%H:%M:%SZ")  # Adjust the format if necessary
    startTime = int(f"{st.hour}{st.minute:02d}")
  date = REPLACE_VALUE
  if 'gameDate' in game and game['gameDate']:
    date = int(str(game['gameDate']).replace('-', ''))
  homeTeam = safe_chain(game,'homeTeam')
  awayTeam = safe_chain(game,'awayTeam')
  gi = safe_chain(game,'boxscore','gameInfo')
  pbgs = safe_chain(game,'boxscore','playerByGameStats')
  allPlayerIds = []
  if safe_chain(pbgs,'awayTeam','forwards') != REPLACE_VALUE:
    for p in pbgs['awayTeam']['forwards']:
      allPlayerIds.append({'playerId': p['playerId']})
  if safe_chain(pbgs,'awayTeam','defense') != REPLACE_VALUE:
    for p in pbgs['awayTeam']['defense']:
      allPlayerIds.append({'playerId': p['playerId']})
  if safe_chain(pbgs,'awayTeam','goalies') != REPLACE_VALUE:
    for p in pbgs['awayTeam']['goalies']:
      allPlayerIds.append({'playerId': p['playerId']})
  if safe_chain(pbgs,'homeTeam','forwards') != REPLACE_VALUE:
    for p in pbgs['homeTeam']['forwards']:
      allPlayerIds.append({'playerId': p['playerId']})
  if safe_chain(pbgs,'homeTeam','defense') != REPLACE_VALUE:
    for p in pbgs['homeTeam']['defense']:
      allPlayerIds.append({'playerId': p['playerId']})
  if safe_chain(pbgs,'homeTeam','goalies') != REPLACE_VALUE:
    for p in pbgs['homeTeam']['goalies']:
      allPlayerIds.append({'playerId': p['playerId']})
  if len(allPlayerIds) > 0:
    allPlayers = list(Players.find(
      {'$or': allPlayerIds},
      {'_id': 0, 'playerId': 1, 'birthDate': 1, 'shootsCatches': 1, 'weightInPounds': 1, 'heightInInches': 1}
    ))
  else:
    allPlayers = []
  try:
    return {
      'id': safe_chain(game,'id'),
      'season': safe_chain(game,'season'),
      'gameType': safe_chain(game,'gameType'),
      'venue': n2n(safe_chain(game,'venue','default')),
      'neutralSite': b2n(safe_chain(game,'neutralSite')),
      'homeTeam': homeTeam['id'],
      'awayTeam': awayTeam['id'],
      'homeScore': homeTeam['score'],
      'awayScore': awayTeam['score'],
      'winner': homeTeam['id'] if homeTeam['score'] > awayTeam['score'] else awayTeam['id'],
      # 'awaySplitSquad': safe_chain(awayTeam,'awaySplitSquad'),
      # 'homeSplitSquad': safe_chain(homeTeam,'homeSplitSquad'),
      'startTime': startTime,
      'date': date,
      'awayHeadCoach': n2n(safe_chain(gi,'awayTeam','headCoach','default')),
      'homeHeadCoach': n2n(safe_chain(gi,'homeTeam','headCoach','default')),
      'ref1': n2n(safe_chain(gi,'referees',0,'default')),
      'ref2': n2n(safe_chain(gi,'referees',1,'default')),
      'linesman1': n2n(safe_chain(gi,'linesmen',0,'default')),
      'linesman2': n2n(safe_chain(gi,'linesmen',1,'default')),
      'awayForward1': safe_chain(pbgs,'awayTeam','forwards',0,'playerId'),
      'awayForward1Position': n2n(safe_chain(pbgs,'awayTeam','forwards',0,'position')),
      'awayForward1Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',0,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',0,'playerId') else REPLACE_VALUE,
      'awayForward1Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',0,'playerId')),0,'shootsCatches')),
      'awayForward2': safe_chain(pbgs,'awayTeam','forwards',1,'playerId'),
      'awayForward2Position': n2n(safe_chain(pbgs,'awayTeam','forwards',1,'position')),
      'awayForward2Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',1,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',1,'playerId') else REPLACE_VALUE,
      'awayForward2Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',1,'playerId')),0,'shootsCatches')),
      'awayForward3': safe_chain(pbgs,'awayTeam','forwards',2,'playerId'),
      'awayForward3Position': n2n(safe_chain(pbgs,'awayTeam','forwards',2,'position')),
      'awayForward3Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',2,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',2,'playerId') else REPLACE_VALUE,
      'awayForward3Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',2,'playerId')),0,'shootsCatches')),
      'awayForward4': safe_chain(pbgs,'awayTeam','forwards',3,'playerId'),
      'awayForward4Position': n2n(safe_chain(pbgs,'awayTeam','forwards',3,'position')),
      'awayForward4Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',3,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',3,'playerId') else REPLACE_VALUE,
      'awayForward4Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',3,'playerId')),0,'shootsCatches')),
      'awayForward5': safe_chain(pbgs,'awayTeam','forwards',4,'playerId'),
      'awayForward5Position': n2n(safe_chain(pbgs,'awayTeam','forwards',4,'position')),
      'awayForward5Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',4,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',4,'playerId') else REPLACE_VALUE,
      'awayForward5Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',4,'playerId')),0,'shootsCatches')),
      'awayForward6': safe_chain(pbgs,'awayTeam','forwards',5,'playerId'),
      'awayForward6Position': n2n(safe_chain(pbgs,'awayTeam','forwards',5,'position')),
      'awayForward6Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',5,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',5,'playerId') else REPLACE_VALUE,
      'awayForward6Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',5,'playerId')),0,'shootsCatches')),
      'awayForward7': safe_chain(pbgs,'awayTeam','forwards',6,'playerId'),
      'awayForward7Position': n2n(safe_chain(pbgs,'awayTeam','forwards',6,'position')),
      'awayForward7Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',6,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',6,'playerId') else REPLACE_VALUE,
      'awayForward7Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',6,'playerId')),0,'shootsCatches')),
      'awayForward8': safe_chain(pbgs,'awayTeam','forwards',7,'playerId'),
      'awayForward8Position': n2n(safe_chain(pbgs,'awayTeam','forwards',7,'position')),
      'awayForward8Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',7,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',7,'playerId') else REPLACE_VALUE,
      'awayForward8Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',7,'playerId')),0,'shootsCatches')),
      'awayForward9': safe_chain(pbgs,'awayTeam','forwards',8,'playerId'),
      'awayForward9Position': n2n(safe_chain(pbgs,'awayTeam','forwards',8,'position')),
      'awayForward9Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',8,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',8,'playerId') else REPLACE_VALUE,
      'awayForward9Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',8,'playerId')),0,'shootsCatches')),
      'awayForward10': safe_chain(pbgs,'awayTeam','forwards',9,'playerId'),
      'awayForward10Position': n2n(safe_chain(pbgs,'awayTeam','forwards',9,'position')),
      'awayForward10Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',9,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',9,'playerId') else REPLACE_VALUE,
      'awayForward10Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',9,'playerId')),0,'shootsCatches')),
      'awayForward11': safe_chain(pbgs,'awayTeam','forwards',10,'playerId'),
      'awayForward11Position': n2n(safe_chain(pbgs,'awayTeam','forwards',10,'position')),
      'awayForward11Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',10,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',10,'playerId') else REPLACE_VALUE,
      'awayForward11Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',10,'playerId')),0,'shootsCatches')),
      'awayForward12': safe_chain(pbgs,'awayTeam','forwards',11,'playerId'),
      'awayForward12Position': n2n(safe_chain(pbgs,'awayTeam','forwards',11,'position')),
      'awayForward12Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',11,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',11,'playerId') else REPLACE_VALUE,
      'awayForward12Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',11,'playerId')),0,'shootsCatches')),
      'awayForward13': safe_chain(pbgs,'awayTeam','forwards',12,'playerId'),
      'awayForward13Position': n2n(safe_chain(pbgs,'awayTeam','forwards',12,'position')),
      'awayForward13Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',12,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','forwards',12,'playerId') else REPLACE_VALUE,
      'awayForward13Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','forwards',12,'playerId')),0,'shootsCatches')),
      'awayDefenseman1': safe_chain(pbgs,'awayTeam','defense',0,'playerId'),
      'awayDefenseman1Position': n2n(safe_chain(pbgs,'awayTeam','defense',0,'position')),
      'awayDefenseman1Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',0,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',0,'playerId') else REPLACE_VALUE,
      'awayDefenseman1Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',0,'playerId')),0,'shootsCatches')),
      'awayDefenseman2': safe_chain(pbgs,'awayTeam','defense',1,'playerId'),
      'awayDefenseman2Position': n2n(safe_chain(pbgs,'awayTeam','defense',1,'position')),
      'awayDefenseman2Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',1,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',1,'playerId') else REPLACE_VALUE,
      'awayDefenseman2Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',1,'playerId')),0,'shootsCatches')),
      'awayDefenseman3': safe_chain(pbgs,'awayTeam','defense',2,'playerId'),
      'awayDefenseman3Position': n2n(safe_chain(pbgs,'awayTeam','defense',2,'position')),
      'awayDefenseman3Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',2,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',2,'playerId') else REPLACE_VALUE,
      'awayDefenseman3Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',2,'playerId')),0,'shootsCatches')),
      'awayDefenseman4': safe_chain(pbgs,'awayTeam','defense',3,'playerId'),
      'awayDefenseman4Position': n2n(safe_chain(pbgs,'awayTeam','defense',3,'position')),
      'awayDefenseman4Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',3,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',3,'playerId') else REPLACE_VALUE,
      'awayDefenseman4Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',3,'playerId')),0,'shootsCatches')),
      'awayDefenseman5': safe_chain(pbgs,'awayTeam','defense',4,'playerId'),
      'awayDefenseman5Position': n2n(safe_chain(pbgs,'awayTeam','defense',4,'position')),
      'awayDefenseman5Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',4,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',4,'playerId') else REPLACE_VALUE,
      'awayDefenseman5Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',4,'playerId')),0,'shootsCatches')),
      'awayDefenseman6': safe_chain(pbgs,'awayTeam','defense',5,'playerId'),
      'awayDefenseman6Position': n2n(safe_chain(pbgs,'awayTeam','defense',5,'position')),
      'awayDefenseman6Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',5,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',5,'playerId') else REPLACE_VALUE,
      'awayDefenseman6Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',5,'playerId')),0,'shootsCatches')),
      'awayDefenseman7': safe_chain(pbgs,'awayTeam','defense',6,'playerId'),
      'awayDefenseman7Position': n2n(safe_chain(pbgs,'awayTeam','defense',6,'position')),
      'awayDefenseman7Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',6,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','defense',6,'playerId') else REPLACE_VALUE,
      'awayDefenseman7Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','defense',6,'playerId')),0,'shootsCatches')),
      'awayStartingGoalie': safe_chain(pbgs,'awayTeam','goalies',0,'playerId'),
      'awayStartingGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',0,'playerId')),0,'shootsCatches')),
      'awayStartingGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',0,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','goalies',0,'playerId') else REPLACE_VALUE,
      'awayStartingGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',0,'playerId')),0,'heightInInches'),
      'awayStartingGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',0,'playerId')),0,'weightInPounds'),
      'awayBackupGoalie': safe_chain(pbgs,'awayTeam','goalies',1,'playerId'),
      'awayBackupGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',1,'playerId')),0,'shootsCatches')),
      'awayBackupGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',1,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','goalies',1,'playerId') else REPLACE_VALUE,
      'awayBackupGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',1,'playerId')),0,'heightInInches'),
      'awayBackupGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',1,'playerId')),0,'weightInPounds'),
      'awayThirdGoalie': safe_chain(pbgs,'awayTeam','goalies',2,'playerId'),
      'awayThirdGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',2,'playerId')),0,'shootsCatches')),
      'awayThirdGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',2,'playerId')),game['gameDate']) if false_chain(pbgs,'awayTeam','goalies',2,'playerId') else REPLACE_VALUE,
      'awayThirdGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',2,'playerId')),0,'heightInInches'),
      'awayThirdGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'awayTeam','goalies',2,'playerId')),0,'weightInPounds'),
      'homeForward1': safe_chain(pbgs,'homeTeam','forwards',0,'playerId'),
      'homeForward1Position': n2n(safe_chain(pbgs,'homeTeam','forwards',0,'position')),
      'homeForward1Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',0,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',0,'playerId') else REPLACE_VALUE,
      'homeForward1Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',0,'playerId')),0,'shootsCatches')),
      'homeForward2': safe_chain(pbgs,'homeTeam','forwards',1,'playerId'),
      'homeForward2Position': n2n(safe_chain(pbgs,'homeTeam','forwards',1,'position')),
      'homeForward2Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',1,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',1,'playerId') else REPLACE_VALUE,
      'homeForward2Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',1,'playerId')),0,'shootsCatches')),
      'homeForward3': safe_chain(pbgs,'homeTeam','forwards',2,'playerId'),
      'homeForward3Position': n2n(safe_chain(pbgs,'homeTeam','forwards',2,'position')),
      'homeForward3Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',2,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',2,'playerId') else REPLACE_VALUE,
      'homeForward3Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',2,'playerId')),0,'shootsCatches')),
      'homeForward4': safe_chain(pbgs,'homeTeam','forwards',3,'playerId'),
      'homeForward4Position': n2n(safe_chain(pbgs,'homeTeam','forwards',3,'position')),
      'homeForward4Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',3,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',3,'playerId') else REPLACE_VALUE,
      'homeForward4Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',3,'playerId')),0,'shootsCatches')),
      'homeForward5': safe_chain(pbgs,'homeTeam','forwards',4,'playerId'),
      'homeForward5Position': n2n(safe_chain(pbgs,'homeTeam','forwards',4,'position')),
      'homeForward5Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',4,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',4,'playerId') else REPLACE_VALUE,
      'homeForward5Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',4,'playerId')),0,'shootsCatches')),
      'homeForward6': safe_chain(pbgs,'homeTeam','forwards',5,'playerId'),
      'homeForward6Position': n2n(safe_chain(pbgs,'homeTeam','forwards',5,'position')),
      'homeForward6Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',5,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',5,'playerId') else REPLACE_VALUE,
      'homeForward6Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',5,'playerId')),0,'shootsCatches')),
      'homeForward7': safe_chain(pbgs,'homeTeam','forwards',6,'playerId'),
      'homeForward7Position': n2n(safe_chain(pbgs,'homeTeam','forwards',6,'position')),
      'homeForward7Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',6,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',6,'playerId') else REPLACE_VALUE,
      'homeForward7Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',6,'playerId')),0,'shootsCatches')),
      'homeForward8': safe_chain(pbgs,'homeTeam','forwards',7,'playerId'),
      'homeForward8Position': n2n(safe_chain(pbgs,'homeTeam','forwards',7,'position')),
      'homeForward8Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',7,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',7,'playerId') else REPLACE_VALUE,
      'homeForward8Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',7,'playerId')),0,'shootsCatches')),
      'homeForward9': safe_chain(pbgs,'homeTeam','forwards',8,'playerId'),
      'homeForward9Position': n2n(safe_chain(pbgs,'homeTeam','forwards',8,'position')),
      'homeForward9Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',8,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',8,'playerId') else REPLACE_VALUE,
      'homeForward9Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',8,'playerId')),0,'shootsCatches')),
      'homeForward10': safe_chain(pbgs,'homeTeam','forwards',9,'playerId'),
      'homeForward10Position': n2n(safe_chain(pbgs,'homeTeam','forwards',9,'position')),
      'homeForward10Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',9,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',9,'playerId') else REPLACE_VALUE,
      'homeForward10Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',9,'playerId')),0,'shootsCatches')),
      'homeForward11': safe_chain(pbgs,'homeTeam','forwards',10,'playerId'),
      'homeForward11Position': n2n(safe_chain(pbgs,'homeTeam','forwards',10,'position')),
      'homeForward11Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',10,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',10,'playerId') else REPLACE_VALUE,
      'homeForward11Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',10,'playerId')),0,'shootsCatches')),
      'homeForward12': safe_chain(pbgs,'homeTeam','forwards',11,'playerId'),
      'homeForward12Position': n2n(safe_chain(pbgs,'homeTeam','forwards',11,'position')),
      'homeForward12Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',11,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',11,'playerId') else REPLACE_VALUE,
      'homeForward12Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',11,'playerId')),0,'shootsCatches')),
      'homeForward13': safe_chain(pbgs,'homeTeam','forwards',12,'playerId'),
      'homeForward13Position': n2n(safe_chain(pbgs,'homeTeam','forwards',12,'position')),
      'homeForward13Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',12,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','forwards',12,'playerId') else REPLACE_VALUE,
      'homeForward13Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','forwards',12,'playerId')),0,'shootsCatches')),
      'homeDefenseman1': safe_chain(pbgs,'homeTeam','defense',0,'playerId'),
      'homeDefenseman1Position': n2n(safe_chain(pbgs,'homeTeam','defense',0,'position')),
      'homeDefenseman1Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',0,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',0,'playerId') else REPLACE_VALUE,
      'homeDefenseman1Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',0,'playerId')),0,'shootsCatches')),
      'homeDefenseman2': safe_chain(pbgs,'homeTeam','defense',1,'playerId'),
      'homeDefenseman2Position': n2n(safe_chain(pbgs,'homeTeam','defense',1,'position')),
      'homeDefenseman2Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',1,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',1,'playerId') else REPLACE_VALUE,
      'homeDefenseman2Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',1,'playerId')),0,'shootsCatches')),
      'homeDefenseman3': safe_chain(pbgs,'homeTeam','defense',2,'playerId'),
      'homeDefenseman3Position': n2n(safe_chain(pbgs,'homeTeam','defense',2,'position')),
      'homeDefenseman3Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',2,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',2,'playerId') else REPLACE_VALUE,
      'homeDefenseman3Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',2,'playerId')),0,'shootsCatches')),
      'homeDefenseman4': safe_chain(pbgs,'homeTeam','defense',3,'playerId'),
      'homeDefenseman4Position': n2n(safe_chain(pbgs,'homeTeam','defense',3,'position')),
      'homeDefenseman4Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',3,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',3,'playerId') else REPLACE_VALUE,
      'homeDefenseman4Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',3,'playerId')),0,'shootsCatches')),
      'homeDefenseman5': safe_chain(pbgs,'homeTeam','defense',4,'playerId'),
      'homeDefenseman5Position': n2n(safe_chain(pbgs,'homeTeam','defense',4,'position')),
      'homeDefenseman5Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',4,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',4,'playerId') else REPLACE_VALUE,
      'homeDefenseman5Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',4,'playerId')),0,'shootsCatches')),
      'homeDefenseman6': safe_chain(pbgs,'homeTeam','defense',5,'playerId'),
      'homeDefenseman6Position': n2n(safe_chain(pbgs,'homeTeam','defense',5,'position')),
      'homeDefenseman6Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',5,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',5,'playerId') else REPLACE_VALUE,
      'homeDefenseman6Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',5,'playerId')),0,'shootsCatches')),
      'homeDefenseman7': safe_chain(pbgs,'homeTeam','defense',6,'playerId'),
      'homeDefenseman7Position': n2n(safe_chain(pbgs,'homeTeam','defense',6,'position')),
      'homeDefenseman7Age': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',6,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','defense',6,'playerId') else REPLACE_VALUE,
      'homeDefenseman7Shoots': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','defense',6,'playerId')),0,'shootsCatches')),
      'homeStartingGoalie': safe_chain(pbgs,'homeTeam','goalies',0,'playerId'),
      'homeStartingGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',0,'playerId')),0,'shootsCatches')),
      'homeStartingGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',0,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','goalies',0,'playerId') else REPLACE_VALUE,
      'homeStartingGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',0,'playerId')),0,'heightInInches'),
      'homeStartingGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',0,'playerId')),0,'weightInPounds'),
      'homeBackupGoalie': safe_chain(pbgs,'homeTeam','goalies',1,'playerId'),
      'homeBackupGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',1,'playerId')),0,'shootsCatches')),
      'homeBackupGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',1,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','goalies',1,'playerId') else REPLACE_VALUE,
      'homeBackupGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',1,'playerId')),0,'heightInInches'),
      'homeBackupGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',1,'playerId')),0,'weightInPounds'),
      'homeThirdGoalie': safe_chain(pbgs,'homeTeam','goalies',2,'playerId'),
      'homeThirdGoalieCatches': n2n(safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',2,'playerId')),0,'shootsCatches')),
      'homeThirdGoalieAge': getAge(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',2,'playerId')),game['gameDate']) if false_chain(pbgs,'homeTeam','goalies',2,'playerId') else REPLACE_VALUE,
      'homeThirdGoalieHeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',2,'playerId')),0,'heightInInches'),
      'homeThirdGoalieWeight': safe_chain(getPlayer(allPlayers,safe_chain(pbgs,'homeTeam','goalies',2,'playerId')),0,'weightInPounds')
    }
  except Exception as error:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('ERROR','HELPERS')
    print('id', safe_chain(game,'id'))
    # print('homeTeam',homeTeam)
    # print('awayTeam',awayTeam)
    print('error',error)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

def projectedLineup(team,gameId):
  TEAM_SEASON_SCHEDULE = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/now'
  data = requests.get(TEAM_SEASON_SCHEDULE).json()
  gameIDs = [game['id'] for game in data['games'] if game['id'] < gameId]
  last_game = min(gameIDs, key=lambda x:abs(x-gameId))
  return last_game

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