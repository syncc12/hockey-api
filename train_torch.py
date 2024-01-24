import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from joblib import load
from constants.inputConstants import X_INPUTS, Y_OUTPUTS
from constants.constants import VERSION, FILE_VERSION, RANDOM_STATE, START_SEASON, END_SEASON, VERBOSE
import time

NUM_EPOCHS = 10
BATCH_SIZE = 32

training_data_load = load(f'training_data/training_data_v{FILE_VERSION}.joblib')
training_data = [[key for key in X_INPUTS]]
for i in training_data_load:
  training_data.append([i[key] for key in X_INPUTS])
# training_data = [[key for key in X_INPUTS],([i[key] for key in X_INPUTS] for i in training_data)]
# print(training_data[0])
# print(training_data[1])
class CustomDataset(Dataset):
  def __init__(self, data):
    self.data = data

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    item = self.data[idx]
    tensors = torch.tensor(item, dtype=torch.float)
    return tensors


class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()
    # Define layers here
    self.output_layer = nn.Linear(10,1)

  def forward(self, x):
    # Define forward pass
    x = self.winnerB
    return x


dataset = CustomDataset(training_data)
total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% for training
test_size = total_size - train_size  # 20% for testing

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Net().to(device)


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
# Prepare data
# ...

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(NUM_EPOCHS):
  for data in train_loader:
    inputs, labels = data[0].to(device), data[1].to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

model.eval()  # Set the model to evaluation mode
with torch.no_grad():
  correct = 0
  total = 0
  for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    # Calculate accuracy based on outputs and labels
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the test set: {100 * correct / total}%')

# Save/Load the model
# ...