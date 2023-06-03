
import torch


def get_criterion(config):
    if config.criterion.criterion_type == 'cross_entropy':        
        weights = config.criterion.cross_entropy.weights
        if(weights is not None):
            weights = torch.Tensor(weights).cuda()
        return torch.nn.CrossEntropyLoss(weight=weights)
    return torch.nn.MSELoss()
