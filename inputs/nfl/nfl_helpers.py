from util.helpers import isNaN
from datetime import datetime

REPLACE_VALUE = -1


def formatTime(inTime):
  if (type(inTime) == str):
    st = inTime.split(':')
    return float(int(st[0]) * (int(st[1])/60))
  else:
    return inTime

def formatDatetime(inDatetime):
  dt = inDatetime
  outTime = float(int(dt.hour) * ((int(dt.minute) * (int(dt.second)/60))/60))
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