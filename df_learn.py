import pandas as pd
import torch


def l1():
  dl = [
    {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5},
    {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10},
    {'a': 11, 'b': 12, 'c': 13, 'd': 14, 'e': 15},
    {'a': 16, 'b': 17, 'c': 18, 'd': 19, 'e': 20},
    {'a': 21, 'b': 22, 'c': 23, 'd': 24, 'e': 25},
  ]
  data = pd.DataFrame(dl)
  v = data[['a']].values
  print(type(v))
  t = torch.tensor(v)
  print(t.unsqueeze(1))

def l2():
  # Example dataset
  features = torch.randn(100, 1, 28) # 100 samples, 1 channel, 28 data points per sample
  labels = torch.randint(0, 2, (100,)) # 100 binary labels
  print(features)
  print(labels)


if __name__ == '__main__':
  l2()