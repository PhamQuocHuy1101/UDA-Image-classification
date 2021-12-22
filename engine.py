import torch
import numpy as np

def get_tas_threshold(iter, max_iter, n_class, method='linear'):
    if method == 'linear':
        alpha = iter / max_iter
    elif method == 'log':
        alpha = 1 - np.e ** (5 * (-iter / max_iter))
    elif method == 'exp':
        alpha = np.e ** ((iter / max_iter - 1) * 5)
    else:
        return 0
    return alpha * (1 - 1 / n_class) + 1 / n_class

def get_supervised_loss(logits, Y_true, criterion, threshold):
    loss_item = criterion(logits, Y_true) # [batch] reduction = 'none'
    probs, predict_label = torch.max(torch.softmax(logits, dim = 1), dim = 1)
    mask = torch.ones_like(loss_item)
    ignore_idx = (probs > threshold)*(predict_label == Y_true) # ignore all sample that its probs > threshold
    mask[ignore_idx] = 0.0
    return torch.sum(loss_item * mask)/max(torch.sum(mask).item(), 1)

def get_unsupervised_loss(unsup_logits, aug_logits, criterion, beta, teta):
    unsup_probs, _ = torch.softmax(unsup_logits, dim = 1).max(dim = 1)
    sharp_unsup_logits = torch.softmax(unsup_logits / teta, dim = 1) # sharp unsup label
    aug_logits = torch.log_softmax(aug_logits, dim = 1)
    loss_item = criterion(aug_logits, sharp_unsup_logits)
    mask = torch.ones_like(loss_item) # masking
    ignore_idx = unsup_probs < beta
    mask[ignore_idx] = 0.0
    return torch.mean(loss_item * mask)
