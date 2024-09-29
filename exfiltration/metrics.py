from typing import Callable
import torch

def precision_score(preds, targets):
        
    try:
        precision = ((preds == targets) & preds).sum().item() / preds.sum().item()
    except Exception as e:
        precision = 0.

    return precision

def recall_score(preds, targets):
        
    try:
        recall = ((preds == targets) & preds).sum().item() / targets.sum().item()
    except Exception as e:
        recall = 0.

    return recall

def f1_score(preds, targets):

    precision = precision_score(preds, targets)
    recall = recall_score(preds, targets)
    f1 = 2*precision*recall / (precision + recall + 1e-9)
    return f1

def perplexity_score(loss):
    return torch.exp(loss).item()