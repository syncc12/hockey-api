import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt



class HingeLoss(torch.nn.Module):
  def __init__(self):
    super(HingeLoss, self).__init__()

  def forward(self, outputs, labels):
    # Ensure the labels are -1 or 1
    labels = 2 * labels - 1
    loss = 1 - labels * outputs
    loss[loss < 0] = 0  # Equivalent to max(0, 1 - y * pred)
    return loss.mean()

class FocalLoss(torch.nn.Module):
  def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
    super(FocalLoss, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduction = reduction

  def forward(self, inputs, targets):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
    F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

    if self.reduction == 'mean':
      return torch.mean(F_loss)
    elif self.reduction == 'sum':
      return torch.sum(F_loss)
    else:
      return F_loss

def binary_accuracy(y_pred, y_true):
  y_pred_sig = torch.sigmoid(y_pred)
  y_pred_tag = (y_pred_sig > 0.5).float()
  correct_results = (y_pred_tag == y_true).float()
  accuracy = correct_results.sum() / len(correct_results)
  return accuracy.item() * 100

def calculate_accuracy(outputs, labels):
    # Convert logits to predicted class indices
    _, predicted = torch.max(outputs, dim=1)
    
    # Calculate the number of correctly predicted examples
    correct = (predicted == labels).sum().item()
    
    # Calculate the accuracy
    accuracy = correct / labels.size(0)
    return accuracy

def errorAnalysis(model,validation_loader,device):
  # Assume model, validation_loader are defined
  # Forward pass over the validation set
  all_preds = []
  all_labels = []
  model.eval()
  with torch.no_grad():
    for data in validation_loader:
      inputs, labels = data
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      _, preds = torch.max(outputs, 1)
      all_preds.extend(preds.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

  # Analyze misclassifications
  # misclassified = [(pred, label) for pred, label in zip(all_preds, all_labels) if pred != label]
  # print(misclassified)

  # Error analysis
  # Example: Confusion Matrix
  cm = confusion_matrix(all_labels, all_preds)
  sns.heatmap(cm, annot=True)
  plt.show()