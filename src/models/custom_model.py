# MIT License
# Copyright (c) 2022 Rohit Singh and Yevhenii Maslov
# link: https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution
# Modified by Majidov Ikhtiyor

import torch
import torch.nn as nn
from .pooling_layers import get_pooling_layer
from transformers import AutoModel, AutoConfig
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
# from https://github.com/digantamisra98/Mish/blob/b5f006660ac0b4c46e2c6958ad0301d7f9c59651/Mish/Torch/mish.py
@torch.jit.script
def mish(input):
    return input * torch.tanh(F.softplus(input))
  
class Mish(nn.Module):
    def forward(self, input):
        return mish(input)
    
class CustomModel(nn.Module):
    def __init__(self, cfg, backbone_config):
        super().__init__()
        self.cfg = cfg
        self.backbone_config = backbone_config

        if self.cfg.model.pretrained_backbone:
            self.backbone = AutoModel.from_pretrained(cfg.model.backbone_type, config=self.backbone_config)
        else:
            self.backbone = AutoModel.from_config(self.backbone_config)

        self.backbone.resize_token_embeddings(len(cfg.tokenizer))
        self.pool = get_pooling_layer(cfg, backbone_config)
        
        self.classifier = nn.Sequential(
            nn.Dropout(cfg.model.classifier_dropout),
            nn.Linear(self.pool.output_dim, self.pool.output_dim//2),
            Mish(),
            nn.Dropout(cfg.model.classifier_dropout),
            nn.Linear(self.pool.output_dim//2, len(self.cfg.general.classes))            
        )
        self.classifier.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if(getattr(self.backbone_config, 'initializer_range', None)) is not None:
                std = self.backbone_config.initializer_range
            else:
                std = 1.0
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        feature = self.pool(inputs, outputs)
        output = self.classifier(feature)
        return output
