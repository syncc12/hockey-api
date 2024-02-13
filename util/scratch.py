import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')

from joblib import load, dump
import pandas as pd
import os
import requests
from helpers import n2n, getGamesPlayed, getAllGamesPlayed, getTotalGamesPlayed, getAllAges, getPlayerStats, projected_roster
from pymongo import MongoClient, ASCENDING
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split


db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
client = MongoClient(db_url)
db = client['hockey']

# # Odds = db['dev_odds']

# # game_odds = Odds.find_one(
# #   {'id': 2024020528},
# #   {'_id':0,'odds':1}
# # )
# # # game_odds = Odds.find_one(
# # #   {'id': 2023020555},
# # #   {'_id':0,'odds':1}
# # # )
# # print(game_odds)

# # # Games = db['dev_games']

# # # pipeline = [
# # #   {"$group": {
# # #     "_id": "$id", 
# # #     "uniqueIds": {"$addToSet": "$_id"},
# # #     "count": {"$sum": 1}
# # #   }},
# # #   {"$match": {
# # #     "count": {"$gt": 1}
# # #   }}
# # # ]
# # # duplicates = Games.aggregate(pipeline)

# # # for document in duplicates:
# # #   for id in document["uniqueIds"][1:]:
# # #     Games.delete_one({"_id": id})
      
# # # Games.create_index([("id", ASCENDING)], unique=True)

# def hyperparameter_tuning():
#   imputer = SimpleImputer(strategy='constant', fill_value=-1)
#   training_data = load('training_data/training_data_v6.joblib')
#   data = pd.DataFrame(training_data)
#   x = data [[
#     'id','season','gameType','venue','neutralSite','homeTeam','awayTeam','startTime','date','awayHeadCoach','homeHeadCoach','ref1','ref2','linesman1','linesman1','awayForward1','awayForward1Age',
#     'awayForward2','awayForward2Age','awayForward3','awayForward3Age','awayForward4','awayForward4Age','awayForward5','awayForward5Age','awayForward6','awayForward6Age','awayForward7',
#     'awayForward7Age','awayForward8','awayForward8Age','awayForward9','awayForward9Age','awayForward10','awayForward10Age','awayForward11','awayForward11Age','awayForward12','awayForward12Age',
#     'awayForward13','awayForward13Age','awayDefenseman1','awayDefenseman1Age','awayDefenseman2','awayDefenseman2Age','awayDefenseman3','awayDefenseman3Age','awayDefenseman4','awayDefenseman4Age',
#     'awayDefenseman5','awayDefenseman5Age','awayDefenseman6','awayDefenseman6Age','awayDefenseman7','awayDefenseman7Age','awayStartingGoalie','awayStartingGoalieCatches','awayStartingGoalieAge',
#     'awayStartingGoalieHeight','awayBackupGoalie','awayBackupGoalieCatches','awayBackupGoalieAge','awayBackupGoalieHeight','homeForward1','homeForward1Age','homeForward2','homeForward2Age',
#     'homeForward3','homeForward3Age','homeForward4','homeForward4Age','homeForward5','homeForward5Age','homeForward6','homeForward6Age','homeForward7','homeForward7Age','homeForward8',
#     'homeForward8Age','homeForward9','homeForward9Age','homeForward10','homeForward10Age','homeForward11','homeForward11Age','homeForward12','homeForward12Age','homeForward13','homeForward13Age',
#     'homeDefenseman1','homeDefenseman1Age','homeDefenseman2','homeDefenseman2Age','homeDefenseman3','homeDefenseman3Age','homeDefenseman4','homeDefenseman4Age','homeDefenseman5',
#     'homeDefenseman5Age','homeDefenseman6','homeDefenseman6Age','homeDefenseman7','homeDefenseman7Age','homeStartingGoalie','homeStartingGoalieCatches','homeStartingGoalieAge',
#     'homeStartingGoalieHeight','homeBackupGoalie','homeBackupGoalieCatches','homeBackupGoalieAge','homeBackupGoalieHeight'
#   ]].values
#   # y = data [['homeScore','awayScore','winner','totalGoals','goalDifferential']].values
#   y = data [['winner']].values.ravel()

#   imputer.fit(x)
#   x = imputer.transform(x)

#   x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12)
#   # Define the parameter grid
#   print(y_train.shape)
#   param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [4, 5, 6, 7, 8],
#     'criterion': ['gini', 'entropy']
#   }

#   # Instantiate RandomForestClassifier
#   clf = RandomForestClassifier(random_state=12)

#   # Instantiate GridSearchCV
#   grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5)
  
#   # Fit GridSearchCV
#   grid_search.fit(x_train, y_train)

#   # Best parameters and best score
#   best_params = grid_search.best_params_
#   best_score = grid_search.best_score_

#   print('best_params',best_params)
#   print('best_score',best_score)


# hyperparameter_tuning()

# check_number = 18658
# for i in range(1,31):
#   print(i,check_number/i)

# players = [
#   8478585,
#   8475786,
#   8477934,
# ]
# games_played = getAllGamesPlayed(db,players,2023020220,'forwards')
# # games_played = getAllGamesPlayed(db,players,2023020220)
# print(games_played)

# ages = getAllAges(db,players,'2024-1-6')
# print(ages)
# stats = getPlayerStats(db,8478585,20232024,2023020220,'forwards')
# print(stats)
# gameId = 2023020822
# landing_url = f'https://api-web.nhle.com/v1_1/gamecenter/{gameId}/landing'
# roster = f'https://api-web.nhle.com/v1/roster/TEAM/SEASON'
# landing = requests.get(landing_url).json()
# # pr = projected_roster(2023020822)
# print(landing)


Rosters = db['dev_rosters']

all_rosters = list(Rosters.find({}))

roster_lens = []
forwards_lens = []
defensemen_lens = []
goalies_lens = []
for roster in all_rosters:
  roster_len = 0
  roster_len += len(roster['roster']['forwards'])
  forwards_lens.append(len(roster['roster']['forwards']))
  roster_len += len(roster['roster']['defensemen'])
  defensemen_lens.append(len(roster['roster']['defensemen']))
  roster_len += len(roster['roster']['goalies'])
  goalies_lens.append(len(roster['roster']['goalies']))
  roster_lens.append(roster_len)

# print(roster_lens)
print('roster min:',min(roster_lens))
print('roster max:',max(roster_lens))
print('roster average:',sum(roster_lens) / len(roster_lens))
print('forwards min:',min(forwards_lens))
print('forwards max:',max(forwards_lens))
print('forwards average:',sum(forwards_lens) / len(forwards_lens))
print('defensemen min:',min(defensemen_lens))
print('defensemen max:',max(defensemen_lens))
print('defensemen average:',sum(defensemen_lens) / len(defensemen_lens))
print('goalies min:',min(goalies_lens))
print('goalies max:',max(goalies_lens))
print('goalies average:',sum(goalies_lens) / len(goalies_lens))
