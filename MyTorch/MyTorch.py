import sys
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\constants')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\util')
sys.path.append(r'C:\Users\syncc\code\Hockey API\hockey_api\pages')

from pymongo import MongoClient
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from joblib import load
from pages.mlb.inputs import X_INPUTS_MLB, X_INPUTS_MLB_S, ENCODE_COLUMNS, mlb_training_input, mlb_test_input
from pages.mlb.mlb_helpers import team_lookup, away_rename, home_rename
from constants.constants import MLB_VERSION, FILE_MLB_VERSION, RANDOM_STATE, START_SEASON, END_SEASON
import pandas as pd
import numpy as np
from util.torch_helpers import HingeLoss, FocalLoss, binary_accuracy, errorAnalysis
from util.torch_layers import MemoryModule, NoiseInjection
import time

class MyTorch:
  def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, test_loader, device, num_epochs, batch_size, num_workers, lr, l2, model_name, model_version, model_file, model_dir):
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.test_loader = test_loader
    self.device = device
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.lr = lr
    self.l2 = l2
    self.model_name = model_name
    self.model_version = model_version
    self.model_file = model_file
    self.model_dir = model_dir

  def train(self):
    self.model.to(self.device)
    self.loss_fn.to(self.device)
    self.model.train()
    for epoch in range(self.num_epochs):
      running_loss = 0.0
      for i, data in enumerate(self.train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        self.optimizer.step()
        running_loss += loss.item()
      print(f"Epoch {epoch+1}, loss: {running_loss}")
    print('Finished Training')

  def validate(self):
    self.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for data in self.val_loader:
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Validation accuracy: {100 * correct / total}")

  def test(self):
    self.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for data in self.test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)


def torchTrain(num_epochs, train_loader, model, criterion, optimizer, device):
  running_total_accuracy = []
  for epoch in range(num_epochs):
    running_loss = 0.0
    running_accuracy = 0.0
    running_batches = 0
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      # set optimizer to zero grad to remove previous epoch gradients
      optimizer.zero_grad()
      # forward propagation
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      # backward propagation
      loss.backward()
      # optimize
      optimizer.step()
      running_loss += loss.item()
      running_accuracy += binary_accuracy(outputs, labels)
      running_batches += 1
    running_total_accuracy.append(running_accuracy / running_batches)
    running_average_accuracy = sum(running_total_accuracy) / len(running_total_accuracy) if len(running_total_accuracy) > 0 else 0.0
    # display statistics
    print(f'[{epoch + 1}/{num_epochs}] accuracy: {running_accuracy / running_batches:.2f}% ({running_average_accuracy:.2f}%), loss: {running_loss / running_batches:.4f}')

def torchTest(test_loader, model, device, eval_mode=False):
  if eval_mode:
    model.eval()
  correct, total = 0, 0
  total_accuracy = 0
  test_batches = 0
  with torch.no_grad():
    for data in test_loader:
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      # calculate output by running through the network
      outputs = model(inputs)
      # get the predictions
      __, predicted = torch.max(outputs.data, 1)
      # update results
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      current_accuracy = binary_accuracy(outputs, labels)
      total_accuracy += current_accuracy
      test_batches += 1

  average_accuracy = total_accuracy / test_batches
  print(f'Average Accuracy: {average_accuracy}')