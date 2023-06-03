import time 
import math 
from transformers import AutoConfig
import torch
import yaml
from types import SimpleNamespace

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minutes(s), as_minutes(rs))

def dictionary_to_namespace(data):
    if type(data) is list:
        return list(map(dictionary_to_namespace, data))
    elif type(data) is dict:
        sns = SimpleNamespace()
        for key, value in data.items():
            setattr(sns, key, dictionary_to_namespace(value))
        return sns
    else:
        return data

def get_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config