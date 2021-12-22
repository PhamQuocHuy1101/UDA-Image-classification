import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import config
from model import Model_v1
from model.utils import preprocessing
from data import SuperviedData, UnsuperviedData, augmenter
import engine

# global config
global_device = config.global_device

# define model
model = Model_v1(config)
model.to(device = config.global_device)

param_dict = [
    {'params': [p for n, p in model.named_parameters() if 'fc' in n]}, 
    {
        'params': [p for n, p in model.named_parameters() if 'fc' not in n],
        'lr': config.optim['lr'] * 0.1
    }
]

optimizer = optim.AdamW(params = param_dict, lr = config.optim['lr'], weight_decay = config.optim['weight_decay'])

def warn_up_scheduler(cur_step):
    if cur_step <= config.optim['warn_up_step']:
        return cur_step / config.optim['warn_up_step'] * config.optim['lr']
    else:
        next_lr = config.optim['lr'] * np.exp(-5 * cur_step / config.optim['max_iter'])
        return torch.tensor(max(next_lr, config.optim['min_lr']))

scheduler = optim.lr_scheduler.LambdaLB(optimizer, lr_lambda = warn_up_scheduler)

# data
sup_train_csv = pd.read_csv(config.label_data['train_path'])
sup_val_csv = pd.read_csv(config.label_data['val_path'])
unsup_train_csv = pd.read_csv(config.unlabel_data['train_path'])
unsup_val_csv = pd.read_csv(config.unlabel_data['val_path'])

sup_train_dataset = SuperviedData(sup_train_csv.path, sup_train_csv.label, config.label_data['dir_path'], preprocessing)
sup_val_dataset = SuperviedData(sup_val_csv.path, sup_val_csv.label, config.label_data['dir_path'], preprocessing)
unsup_train_dataset = UnsuperviedData(unsup_train_csv.path, config.unlabel_data['dir_path'], preprocessing, augmenter)
unsup_val_dataset = UnsuperviedData(unsup_val_csv.path, config.unlabel_data['dir_path'], preprocessing, augmenter)

sup_train_loader = DataLoader(sup_train_dataset, batch_size = config.optim['sup_batch_size'], shuffle = True)
sup_val_loader = DataLoader(sup_val_dataset, batch_size = 8, shuffle = False) # reduce GPU
unsup_train_loader = DataLoader(unsup_train_dataset, batch_size = config.optim['unsup_batch_size'], shuffle = True)
unsup_val_loader = DataLoader(unsup_val_dataset, batch_size = 8, shuffle = False)

# criterion
cross_entropy = torch.nn.CrossEntropyLoss(reduction='none')
kl_diverence = torch.nn.KLDivLoss(reduction='none')

# load checkpoint
model_name = os.path.join(config.model['checkpoint'], f'{model.name}.pt')
cur_iter = 0
best = -1
if config.optim.continue_training and os.path.exists(model_name):
    print("Load from checkpoint")
    checkpoint = torch.load(model_name, map_location=global_device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best = float(checkpoint['loss'])
    cur_iter = int(checkpoint['iter'])

# train
train_loss = np.array([0.0, 0.0, 0.0]) # total, sup_loss, unsup_loss
for i in tqdm(range(cur_iter, config.optim['max_iter'])):
    print("======================= Iterator ", i)
    model.train()

    sup_X, sup_Y = next(iter(sup_train_loader))
    unsup_X, aug_X = next(iter(unsup_train_loader))
    sub_size, unsup_size = len(sup_X), len(unsup_X)

    # concat
    X = torch.cat((sup_X, unsup_X, aug_X))
    X = X.to(device = global_device)
    Y = sup_Y.to(device = global_device)

    optimizer.zero_grad()
    
    output = model(X)
    sup_logits = output[:sub_size]
    unsup_logits = output[sub_size:sub_size + unsup_size].detach() # no grad
    aug_logits = output[sub_size + unsup_size:]

    sup_threshold = engine.get_tas_threshold(i, config.optim['max_iter'], config.model['n_class'], config.optim['tsa_method'])
    sup_loss = engine.get_supervised_loss(sup_logits, Y, cross_entropy, sup_threshold)
    unsup_loss = engine.get_unsupervised_loss(unsup_logits, aug_logits, kl_diverence, config.optim['unsup_beta'], config.optim['unsup_teta'])
    
    loss = sup_loss + config.optim['unsup_lambda'] * unsup_loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    train_loss += np.array([loss.item(), sup_loss.item(), unsup_loss.item()])

    if i % config.display_step == 0 and i > 0:
        model.eval()
        val_loss = np.array([0.0, 0.0, 0.0])
        with torch.no_grad():
            for ((sup_X, sup_Y), (unsup_X, aug_X)) in zip(sup_val_loader, unsup_val_loader):
                # sup_X, sup_Y = next(iter(sup_val_loader))
                # unsup_X, aug_X = next(iter(unsup_val_loader))
                sub_size, unsup_size = len(sup_X), len(unsup_X)

                # concat
                X = torch.cat((sup_X, unsup_X, aug_X))
                X = X.to(device = global_device)
                Y = sup_Y.to(device = global_device)

                output = model(X)
                sup_logits = output[:sub_size]
                unsup_logits = output[sub_size:sub_size + unsup_size].detach() # no grad
                aug_logits = output[sub_size + unsup_size:]

                sup_threshold = engine.get_tas_threshold(i, config.optim['max_iter'], config.model['n_class'], config.optim['tsa_method'])
                sup_loss = engine.get_supervised_loss(sup_logits, Y, cross_entropy, sup_threshold)
                unsup_loss = engine.get_unsupervised_loss(unsup_logits, aug_logits, kl_diverence, config.optim['unsup_beta'], config.optim['unsup_teta'])
                
                loss = sup_loss + config.optim['unsup_lambda'] * unsup_loss
                val_loss += np.array([loss.item(), sup_loss.item(), unsup_loss.item()])

        train_loss /= config.display_step
        val_loss /= config.display_step

        print('Train loss: ', np.round(train_loss, 5))
        print('Val loss: ', np.round(val_loss, 5))

        if best == -1 or val_loss[0] < best:
            best = val_loss[0]
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best,
                'iter': i
            }, model_name)

print('Finished Training')