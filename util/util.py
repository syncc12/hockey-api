# from pymongo import MongoClient
# from joblib import dump, load

# client = MongoClient("mongodb+srv://syncc12:mEU7TnbyzROdnJ1H@hockey.zl50pnb.mongodb.net")
# db = client["hockey"]

# training_data = load('training_data/training_data.joblib')

# # STEP = 500
# # START = 0
# # END = len(training_data)
# STEP = 1
# START = 2500
# END = 2515

# def save_training_data():
#   Trainings = db['dev_trainings']
#   Trainings.insert_many(training_data)
#   # for i in range(START,END,STEP):
#   #   try:
#   #     print(f'{training_data[i]["id"]} - {training_data[i+(STEP-1)]["id"]} | {i} - {i+(STEP-1)}')
#   #     Trainings.insert_many(training_data[i:i+STEP])
#   #   except Exception as error:
#   #     # print(training_data[i:i+STEP])
#   #     print(f'error {training_data[i]["id"]} - {training_data[i+(STEP-1)]["id"]} | {i} - {i+(STEP-1)}')
#   #     print(error)
#   #     break

# def check_training_data():
#   print(training_data[2506])

# save_training_data()
# # check_training_data()

