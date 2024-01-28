import torch
import torch.nn.functional as F



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