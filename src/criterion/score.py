import numpy as np
from sklearn.metrics import mean_squared_error
import torch

def get_score(y_trues, y_preds):
    """
    Args:
        y_trues: batch_size x 1, true labels
        y_pred: batch_size x n_classes, predictions
    Returns:
        accuraccy 
    """
    y_preds = torch.softmax(torch.from_numpy(y_preds),1).argmax(1)
    y_trues = torch.from_numpy(y_trues)
    accuracy = (y_preds == y_trues).numpy().sum()/len(y_trues)
    return accuracy
