import torch
import math
import sys
from tools.utils import AverageMeter, accuracy

def create_net(model_cls, exp_type, **kwargs):
    models = []
    # flag is a legacy from NexusNet (for routing decisions)
    # We pass it as all 1s for compatibility
    flag = [1, 1, 1, 1]
    
    if exp_type in ['train', 'test', 'duration_train', 'duration_test']:        
        models.append(model_cls(flag, **kwargs))
    elif exp_type == 'single_task':   
        for _ in range(4):
            models.append(model_cls(flag, **kwargs))
    else:
        # For ablations 
        for i in range(0b0000, 0b1111):
            binary_str = bin(i)[2:].zfill(4)
            flag = [int(bit) for bit in binary_str]
            models.append(model_cls(flag, **kwargs))
            
    return models

def train_one_epoch_classifier(iterator, data, model, device, opt, criterion):
    model.to(device)
    criterion = criterion.to(device)
    model.train()
    
    data_X, data_labels = data
    for features, labels in iterator.get_batches(data_X, data_labels, shuffle=True):
        features = features.to(device)
        labels = labels.to(device)
        
        logits, _ = model(features)
        loss = criterion(logits, labels)
        
        if not math.isfinite(loss.item()):
            print(f'Loss is {loss.item()}, stopping training')
            sys.exit(1)
            
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

def evaluate_one_epoch_classifier(iterator, data, model, device, criterion):
    model.to(device)
    model.eval()
    avg_acc = AverageMeter()
    
    data_X, data_labels = data
    for features, labels in iterator.get_batches(data_X, data_labels, shuffle=False):
        features = features.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            logits, _ = model(features)
            loss = criterion(logits, labels)
            
            if not math.isfinite(loss.item()):
                print(f'Loss is {loss.item()}, stopping evaluation')
                sys.exit(1)
                
            acc, _ = accuracy(logits.detach(), labels.detach())
            acc = acc[0]
            avg_acc.update(acc.item(), len(features))
            
    return avg_acc.avg
