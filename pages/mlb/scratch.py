
from pymongo import MongoClient
from mlb_helpers import team_lookup

client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.2zn0c.mongodb.net")
db = client["mlb"]

teamLookup = team_lookup(db,True)

if __name__ == "__main__":
  for i in teamLookup:
    print(teamLookup[i]['abbrev'],i)