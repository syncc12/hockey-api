import time
from helpers import all_combinations


def alignment():
  left = 'Left aligned'
  center = 'Centered'
  right = 'Right aligned'
  print(f'{left:<30}{center:^10}{right:>50}')
  print(f'{left}{center:^10}{right:>50}')
  print(f'{left:<30}{right:>50}')

def loading_bar(progress, total):
  percent = 100 * (progress / total)
  bar = '|' * int(percent) + '-' * (100 - int(percent))
  print(f'\r[{bar}] {percent:.2f}%', end='')

# total = 100
# for i in range(total + 1):
#   loading_bar(i, total)
#   time.sleep(0.1)  # Simulate work being done

# label = "Name"
# value = "John Doe"
# fixed_width = 30  # Adjust this value based on your needs

# formatted_line = "{0}:{1:>{2}}".format(label, value, fixed_width - len(label))
# print(formatted_line)

# # For a different label
# label = "Occupation"
# value = "Software Engineer"
# formatted_line = "{0}:{1:>{2}}".format(label, value, fixed_width - len(label))
# print(formatted_line)


# print(len('winnerB Accuracy:52.54%|eta:0.16|max_depth:27|seasons:15,23||Best: Accuracy:59.53%|eta:0.32|max_depth:28|seasons:08,11'))
# print(len('winnerB Accuracy:54.00%|eta:0.17|max_depth:27|seasons:15,23||Best: Accuracy:59.53%|eta:0.32|max_depth:28|seasons:08,11'))
# print(len('winnerB Accuracy:52.42%|eta:0.18000000000000002|max_depth:27|seasons:15,23||Best: Accuracy:59.53%|eta:0.32|max_depth:28|seasons:08,11'))
# print(len('winnerB Accuracy:55.58%|eta:0.19|max_depth:27|seasons:15,23||Best: Accuracy:59.53%|eta:0.32|max_depth:28|seasons:08,11'))
# print(len('winnerB Accuracy:54.00%|eta:0.2|max_depth:27|seasons:15,23||Best: Accuracy:59.53%|eta:0.32|max_depth:28|seasons:08,11'))
# print(len('winnerB Accuracy:54.79%|eta:0.21000000000000002|max_depth:27|seasons:15,23||Best: Accuracy:59.53%|eta:0.32|max_depth:28|seasons:08,11'))
  
SEASONS = [
  # 20052006,
  # 20062007,
  # 20072008,
  # 20082009,
  # 20092010,
  # 20102011,
  # 20112012,
  # 20122013,
  # 20132014,
  # 20142015,
  # 20152016,
  # 20162017,
  20172018,
  20182019,
  20192020,
  20202021,
  20212022,
  20222023,
]
print(len(SEASONS))
# SEASONS = [1,2,3,4]
ALL_SEASONS = all_combinations(SEASONS)
print(len(ALL_SEASONS))
lower_limit = 2
upper_limit = 4
# min_length = 10
ALL_SEASONS = list(filter(lambda x: len(x) <= lower_limit or len(x) >= upper_limit, ALL_SEASONS))
# print(ALL_SEASONS)
print(len(ALL_SEASONS))