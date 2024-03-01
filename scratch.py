# import sys
# sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages\nhl')
# sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

# from pages.nhl.nhl_helpers import ai2
# from pymongo import MongoClient
# from util.models import MODELS
import csv
import json

if __name__ == '__main__':
  # db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
  # # db_url = f"mongodb+srv://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_NAME')}"
  # client = MongoClient(db_url)
  # db = client['hockey']

  # models = MODELS
  # # print('models:',models)

  # test_games = [
  #   {'id': 2023020866, 'season': 20232024, 'gameType': 2, 'venue': {'default': 'KeyBank Center'}, 'neutralSite': False, 'startTimeUTC': '2024-02-19T17:30:00Z', 'easternUTCOffset': '-05:00', 'venueUTCOffset': '-05:00', 'venueTimezone': 'America/New_York', 'gameState': 'OFF', 'gameScheduleState': 'OK', 'tvBroadcasts': [{'id': 28, 'market': 'H', 'countryCode': 'US', 'network': 'MSG-B', 'sequenceNumber': 62}, {'id': 341, 'market': 'A', 'countryCode': 'US', 'network': 'BSSC', 'sequenceNumber': 84}, {'id': 365, 'market': 'A', 'countryCode': 'US', 'network': 'BSSD', 'sequenceNumber': 81}], 'awayTeam': {'id': 24, 'placeName': {'default': 'Anaheim'}, 'abbrev': 'ANA', 'logo': 'https://assets.nhle.com/logos/nhl/svg/ANA_light.svg', 'darkLogo': 'https://assets.nhle.com/logos/nhl/svg/ANA_dark.svg', 'awaySplitSquad': False, 'score': 4}, 'homeTeam': {'id': 7, 'placeName': {'default': 'Buffalo'}, 'abbrev': 'BUF', 'logo': 'https://assets.nhle.com/logos/nhl/svg/BUF_light.svg', 'darkLogo': 'https://assets.nhle.com/logos/nhl/svg/BUF_dark.svg', 'homeSplitSquad': False, 'score': 3}, 'periodDescriptor': {'number': 3, 'periodType': 'REG'}, 'gameOutcome': {'lastPeriodType': 'REG'}, 'winningGoalie': {'playerId': 8476434, 'firstInitial': {'default': 'J.'}, 'lastName': {'default': 'Gibson'}}, 'winningGoalScorer': {'playerId': 8478873, 'firstInitial': {'default': 'T.'}, 'lastName': {'default': 'Terry'}}, 'threeMinRecap': '/video/recap-ducks-at-sabres-2-19-24-6347215970112', 'gameCenterLink': '/gamecenter/ana-vs-buf/2024/02/19/2023020866'},
  #   {'id': 2023020867, 'season': 20232024, 'gameType': 2, 'venue': {'default': 'TD Garden'}, 'neutralSite': False, 'startTimeUTC': '2024-02-19T18:00:00Z', 'easternUTCOffset': '-05:00', 'venueUTCOffset': '-05:00', 'venueTimezone': 'US/Eastern', 'gameState': 'FINAL', 'gameScheduleState': 'OK', 'tvBroadcasts': [{'id': 31, 'market': 'H', 'countryCode': 'US', 'network': 'NESN', 'sequenceNumber': 78}, {'id': 283, 'market': 'N', 'countryCode': 'CA', 'network': 'SN360', 'sequenceNumber': 1}, {'id': 284, 'market': 'N', 'countryCode': 'CA', 'network': 'SN1', 'sequenceNumber': 23}, {'id': 287, 'market': 'N', 'countryCode': 'CA', 'network': 'SNE', 'sequenceNumber': 27}, {'id': 289, 'market': 'N', 'countryCode': 'CA', 'network': 'SNW', 'sequenceNumber': 31}, {'id': 349, 'market': 'A', 'countryCode': 'US', 'network': 'BSSW', 'sequenceNumber': 61}], 'awayTeam': {'id': 25, 'placeName': {'default': 'Dallas'}, 'abbrev': 'DAL', 'logo': 'https://assets.nhle.com/logos/nhl/svg/DAL_light.svg', 'darkLogo': 'https://assets.nhle.com/logos/nhl/svg/DAL_dark.svg', 'awaySplitSquad': False, 'score': 3}, 'homeTeam': {'id': 6, 'placeName': {'default': 'Boston'}, 'abbrev': 'BOS', 'logo': 'https://assets.nhle.com/logos/nhl/svg/BOS_light.svg', 'darkLogo': 'https://assets.nhle.com/logos/nhl/svg/BOS_dark.svg', 'homeSplitSquad': False, 'score': 4}, 'periodDescriptor': {'number': 5, 'periodType': 'SO'}, 'gameOutcome': {'lastPeriodType': 'SO'}, 'winningGoalie': {'playerId': 8480280, 'firstInitial': {'default': 'J.'}, 'lastName': {'default': 'Swayman'}}, 'gameCenterLink': '/gamecenter/dal-vs-bos/2024/02/19/2023020867'},
  # ]

  # ai2(db,test_games,[False,False],models)
  odds_csv = 'C:/Users/syncc/code/Hockey API/hockey_api/odds.csv'
  odds_json = 'C:/Users/syncc/code/Hockey API/hockey_api/odds.json'

  odds_dicts = []
  with open(odds_csv, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
      odds_dicts.append(row)
  
  with open(odds_json, 'w') as file:
    json.dump(odds_dicts, file, indent=2, ensure_ascii=False)
  