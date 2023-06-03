# MIT License
# Copyright (c) 2022 Rohit Singh and Yevhenii Maslov
# link: https://github.com/rohitsingh02/kaggle-feedback-english-language-learning-1st-place-solution
# Modified by Majidov Ikhtiyor

import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "true"
sys.path.insert(0, os.getcwd())

import gc
import time
import random
import warnings

warnings.filterwarnings("ignore")
import wandb
import os
import sys
import argparse
from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from src.data.preprocess import preprocess_text, get_max_len_from_df
from src.dataset.collators import collate
from src.dataset.EmotionDataset import get_valid_dataloader, get_train_dataloader
from src.adversarial_learning.awp import AWP
import numpy as np
from src.utils import AverageMeter, time_since, get_config, dictionary_to_namespace
from src.models.utils import get_model
from optimizer.optimizer import get_optimizer
from criterion.criterion import get_criterion
from scheduler.scheduler import get_scheduler
from criterion.score import get_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--run_id', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    arguments = parser.parse_args()
    return arguments

def load_data():
    # load the dataset
    dataset = load_dataset("dair-ai/emotion")
    dataset.set_format(type="pandas")
    train_df = dataset['train'][:]
    val_df = dataset['validation'][:]
    test_df = dataset['test'][:]
    return train_df, val_df, test_df

def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model.backbone_type,
                                              use_fast=True
                                              )
    tokenizer.save_pretrained(config.general.tokenizer_dir_path)
    return tokenizer 

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def valid_fn(config, valid_dataloader, model, criterion, epoch, device):
    valid_losses = AverageMeter()
    model.eval()
    predictions = []
    start = time.time()

    for step, (inputs, labels) in enumerate(valid_dataloader):
        inputs = collate(inputs)

        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = labels.to(device)

        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)

        valid_losses.update(loss.item(), batch_size)
        predictions.append(y_preds.to('cpu').numpy())

        if step % config.general.valid_print_frequency == 0 or step == (len(valid_dataloader) - 1):
            remain = time_since(start, float(step + 1) / len(valid_dataloader))
            print('EVAL: [{0}][{1}/{2}] '
                        'Elapsed: {remain:s} '
                        'Loss: {loss.avg:.4f} '
                        .format(epoch+1, step+1, len(valid_dataloader),
                                remain=remain,
                                loss=valid_losses))

        if config.logging.use_wandb:
            wandb.log({f"Validation loss": valid_losses.val})

    predictions = np.concatenate(predictions)
    return valid_losses, predictions

    
    
def train_loop(config, train_df, val_df, device):
    # get the validation label
    valid_labels = val_df['label'].to_numpy()

    # prepare the data loaders
    train_dataloader = get_train_dataloader(config, train_df)
    valid_dataloader = get_valid_dataloader(config, val_df)
    
    

    # prepare the model
    model = get_model(config)
    model.to(device)
    
    # preprare the optimizer
    optimizer = get_optimizer(model, config)
    
    # get the steps to finish one epoch
    train_steps_per_epoch = int(len(train_df) / config.general.train_batch_size)
    # overall steps 
    num_train_steps = train_steps_per_epoch * config.training.epochs
    # evaluation steps to finish one epoch
    eval_steps_per_epoch = int(len(val_df) / config.general.valid_batch_size)
    # overall eval steps
    num_eval_steps = eval_steps_per_epoch * config.training.epochs

    # according to the author's and other users experiences 
    # it helped to increase accuracy significantly
    awp = AWP(model=model,
              optimizer=optimizer,
              adv_lr=config.adversarial_learning.adversarial_lr,
              adv_eps=config.adversarial_learning.adversarial_eps,
              adv_epoch=config.adversarial_learning.adversarial_epoch_start)
    
    # get the loss function 
    criterion = get_criterion(config)

    # get scheduler 
    scheduler = get_scheduler(optimizer, config, num_train_steps)
    
    

    best_score = np.inf
    for epoch in range(config.training.epochs):
        start_time = time.time()
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=config.training.apex)
        train_losses = AverageMeter()
        valid_losses = None
        score, scores = None, None

        start = time.time()
        global_step = 0
        
        for step, (inputs, labels) in enumerate(train_dataloader):
            # collate
            inputs = collate(inputs)
            for k, v in inputs.items():
                inputs[k] = v.to(device)
            labels = labels.to(device)
            awp.perturb(epoch)
            batch_size = labels.size(0)
            with torch.cuda.amp.autocast(enabled=config.training.apex):
                y_preds = model(inputs)
                loss = criterion(y_preds, labels)
            train_losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            awp.restore()

            if config.training.unscale:
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()

            if (step % config.general.train_print_frequency == 0) or \
                    (step == (len(train_dataloader) - 1)):

                remain = time_since(start, float(step + 1) / len(train_dataloader))
                print(f'Epoch: [{epoch+1}][{step+1}/{len(train_dataloader)}] '
                            f'Elapsed {remain:s} '
                            f'Loss: {train_losses.val:.4f}({train_losses.avg:.4f}) '
                            f'Grad: {grad_norm:.4f}  '
                            f'LR: {scheduler.get_lr()[0]:.8f}  ')


        valid_losses, predictions = valid_fn(config, valid_dataloader, model, criterion, epoch, device)
        score = valid_losses.avg
        accuracy = get_score(valid_labels, predictions)

        print(f'Epoch {epoch+1} - Score: {score:.4f} - Accuracy: {accuracy:.4f}')
        # here we are assuming the score as accuracy
        if score < best_score:
            best_score = score

            torch.save({'model': model.state_dict(), 'predictions': predictions}, os.path.join(config.general.model_fn_path, "best.pth"))
            print(f'\nEpoch {epoch + 1} - Save Best Score: {best_score:.4f} Model\n')
        
        # save the full state of the model as last.pth
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
            }, os.path.join(config.general.model_fn_path, "last.pth"))
        
        unique_parameters = ['.'.join(name.split('.')[:4]) for name, _ in model.named_parameters()]
        learning_rates = list(set(zip(unique_parameters, scheduler.get_lr())))

        if config.logging.use_wandb:
            wandb.log({f'{parameter} lr': lr for parameter, lr in learning_rates})
            wandb.log({f'Best Score': best_score})    
        elapsed = time.time() - start_time

        print(f'Epoch {epoch + 1} - avg_train_loss: {train_losses.avg:.4f} '
                    f'avg_val_loss: {valid_losses.avg:.4f} time: {elapsed:.0f}s '
                    f'Epoch {epoch + 1} - Score: {score:.4f}  Accuracy{accuracy}\n, '
                    '=============================================================================\n')
        if config.logging.use_wandb:
            wandb.log({f"Epoch": epoch + 1,
                       f"avg_train_loss": train_losses.avg,
                       f"avg_val_loss": valid_losses.avg,
                       f"accuracy":accuracy,
                       f"Score": score,
                       })
    torch.cuda.empty_cache()
    gc.collect()

def main(config):

    # load the data
    train_df, val_df, test_df = load_data()
    feat_col = 'text'
    label_col = 'label'
    # apply preprocessing 
    train_df[feat_col] = train_df[feat_col].apply(preprocess_text)
    val_df[feat_col] = val_df[feat_col].apply(preprocess_text)
    test_df[feat_col] = test_df[feat_col].apply(preprocess_text)

    tokenizer = get_tokenizer(config)
    config.tokenizer = tokenizer
    
    if config.general.set_max_length_from_data:
        print('Setting max length from data')
        config.general.max_length = get_max_len_from_df(train_df, tokenizer)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    # start the trainng loop 
    train_loop(config, train_df, val_df, device)

def create_folders(config):
    config.general.model_fn_path = os.path.join(config.general.model_fn_path, config.run_id)
    config.general.log_path = os.path.join(config.general.log_path, config.run_id)
    config.general.tokenizer_dir_path = os.path.join(config.general.tokenizer_dir_path, config.model.backbone_type)
    if(os.path.exists(config.general.model_fn_path)):
        raise IOError('This run id is already used pls use different')
    
    os.makedirs(config.general.model_fn_path)
    os.makedirs(config.general.log_path)
    os.makedirs(config.general.tokenizer_dir_path, exist_ok=True)

def init_wandb(config, args):
    backbone_type = config.model.backbone_type
    criterion_type = config.criterion.criterion_type
    pooling_type = config.model.pooling_type

    wandb.login(key='')

    wandb_run = wandb.init(
                    project=config.logging.wandb.project,
                    group=args.run_id,
                    job_type='train',
                    tags=[backbone_type, criterion_type, pooling_type, args.run_id],
                    config=config,
                    name=f'{args.run_id}'
    )
    return wandb_run

if __name__ == '__main__':
    args = parse_args()

    config_path = os.path.join(args.cfg)
    config = get_config(config_path)
    config = dictionary_to_namespace(config)
    config.run_id = args.run_id
    config.logging.use_wandb = args.use_wandb
    
    create_folders(config)
    init_wandb(config, args)
    if(config.logging.use_wandb):
        seed_everything(seed=config.general.seed)
    main(config)
    