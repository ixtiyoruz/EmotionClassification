from .parameters import get_optimizer_params
from torch.optim import AdamW

def get_optimizer(model, config):
    optimizer_parameters = get_optimizer_params(model,
                                                config.optimizer.encoder_lr,
                                                config.optimizer.decoder_lr,
                                                weight_decay=config.optimizer.weight_decay)
    optimizer = AdamW(optimizer_parameters,
                      lr=config.optimizer.encoder_lr,
                      eps=config.optimizer.eps,
                      betas=config.optimizer.betas)
    return optimizer
