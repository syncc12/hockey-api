import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey-api\util')

from joblib import load, dump
import os
from helpers import n2n
from pymongo import MongoClient, ASCENDING


db_url = "mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net"
client = MongoClient(db_url)
db = client['hockey']

Odds = db['dev_odds']

game_odds = Odds.find_one(
  {'id': 2024020528},
  {'_id':0,'odds':1}
)
# game_odds = Odds.find_one(
#   {'id': 2023020555},
#   {'_id':0,'odds':1}
# )
print(game_odds)

# Games = db['dev_games']

# pipeline = [
#   {"$group": {
#     "_id": "$id", 
#     "uniqueIds": {"$addToSet": "$_id"},
#     "count": {"$sum": 1}
#   }},
#   {"$match": {
#     "count": {"$gt": 1}
#   }}
# ]
# duplicates = Games.aggregate(pipeline)

# for document in duplicates:
#   for id in document["uniqueIds"][1:]:
#     Games.delete_one({"_id": id})
      
# Games.create_index([("id", ASCENDING)], unique=True)
