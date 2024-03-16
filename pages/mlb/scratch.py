from inputs import mlb_training_input

SEASONS = [
  2023,
]

def inspect():
  inputs = mlb_training_input(SEASONS)
  print(inputs[0])

if __name__ == "__main__":
  inspect()