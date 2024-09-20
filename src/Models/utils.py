import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2 as transforms_v2
from torcheval.metrics.functional import binary_auroc

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from copy import deepcopy
from collections import defaultdict

import os
import gc

class Dotenv:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                if key != 'kwargs':
                    self.__dict__[key] = Dotenv(value)

class HCD_Dataset_for_training(torch.utils.data.Dataset):

    def __init__(self, data_path, df, data_size, transforms = None, device = torch.device('cuda')):

        super(HCD_Dataset_for_training, self).__init__()

        self.data_path = data_path
        self.positive_df = df[df['label'] == 1]
        self.negative_df = df[df['label'] == 0]
        self.size = int(np.floor(data_size / 2))
        self.device = device
        self.transforms = transforms
    
    def __len__(self):

        return self.size * 2
    
    def __getitem__(self, ix):

        p_df = self.positive_df.sample(self.size).reset_index(drop=True)
        n_df = self.negative_df.sample(self.size).reset_index(drop=True)

        df = pd.concat([p_df,n_df], axis=0).sample(frac=1).reset_index(drop=True)

        image = np.array(Image.open(self.data_path / (df['id'][ix] + '.tif')))
        label = torch.tensor(df['label'][ix])

        if self.transforms:
            image = self.transforms(image=image)["image"]
        else:
            transform = transforms_v2.ToTensor()
            image = transform(image)

        return {'image': image.to(self.device).float(), 'target': label.to(self.device).float()}

class HCD_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, df, transforms = None, device = torch.device('cuda')):

        super(HCD_Dataset, self).__init__()

        self.data_path = data_path
        self.labels = df['label']
        self.image_id = df['id']
        self.device = device
        self.transforms = transforms
    
    def __len__(self):

        return len(self.image_id)
    
    def __getitem__(self, ix):
        
        image = np.array(Image.open(self.data_path / (self.image_id[ix] + '.tif')))
        label = torch.tensor(self.labels[ix])

        if self.transforms:
            image = self.transforms(image=image)["image"]
        else:
            transform = transforms_v2.ToTensor()
            image = transform(image)
            
        return {'image': image.to(self.device).float(), 'target': label.to(self.device).float()}

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_one_epoch(model, criterion, optimizer, scheduler, dataloader):
    model.train()

    loss_hist = []
    out_hist = []
    label_hist = []

    for data in dataloader:

        images = data['image']
        targets = data['target']

        optimizer.zero_grad()
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        loss_hist.append(loss.item())
        out_hist.append(outputs)
        label_hist.append(targets)

    out_hist = torch.cat(out_hist)
    label_hist = torch.cat(label_hist)

    loss = np.array(loss_hist).mean()

    auroc = binary_auroc(input=out_hist, target=label_hist).item()
    
    gc.collect()
    
    return loss, auroc

def eval_one_epoch(model, criterion, dataloader):

    model.eval()
    loss_hist = []
    out_hist = []
    label_hist = []

    with torch.no_grad():
        for data in dataloader:
            output = model(data['image']).squeeze()
            loss = criterion(output, data['target'])
            loss_hist.append(loss.item())

            out_hist.append(output)
            label_hist.append(data['target'])
        
        out_hist = torch.cat(out_hist)
        label_hist = torch.cat(label_hist)

        loss = np.array(loss_hist).mean()
        auroc = binary_auroc(input=out_hist, target=label_hist).item()

        gc.collect()
    
    return loss, auroc

def predict_model(model, dataloader):

    model.eval()

    predictions = []

    with torch.no_grad():
        for data in dataloader:

            outputs = model(data['image']).squeeze()

            predictions.append(outputs)

        soft_preds = torch.cat(predictions)

    
    gc.collect()

    return soft_preds


def train(model, epochs, criterion, optimizer, train_dataloader, val_dataloader = None, scheduler = None, early_stopping = 10, early_reset = None, min_eta = 1e-3, cv_fold = None, save_path = None, from_auroc = None):

    best_model_wts = deepcopy(model.state_dict())

    if early_reset is None:
        reset_count = np.inf
    else:
        reset_count = early_reset

    if from_auroc is None:
        best_epoch_auroc = -np.inf
        ft_str = ''
    else:
        best_epoch_auroc = from_auroc
        ft_str = 'ft_'

    if cv_fold is None:
        cv_fold = ''
    else:
        cv_fold = '_Fold' + str(cv_fold)

    history = defaultdict(list)
    max_count = early_stopping
    count = 0

    for epoch in range(1, epochs + 1):
        
        train_loss, train_auroc = train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, dataloader=train_dataloader)
        val_loss, val_auroc = eval_one_epoch(model=model, criterion=criterion, dataloader=val_dataloader)

        history['epoch'].append(epoch)
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Train AUROC'].append(train_auroc)
        history['Valid AUROC'].append(val_auroc)
        if scheduler is not None:
            history['lr'].append(scheduler.get_lr()[0])
        else:
            history['lr'].append(optimizer.param_groups[0]['lr'])

        if (best_epoch_auroc + min_eta) <= val_auroc:
            print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_auroc}), epoch: {epoch}")
            best_epoch_auroc = val_auroc
            best_model_wts = deepcopy(model.state_dict())
            FILE = ft_str + "AUROC{:.4f}_Loss{:.4f}".format(val_auroc, val_loss) + cv_fold + ".bin"
            if save_path:
                torch.save(model.state_dict(), save_path / FILE)
            count = 0
        else:
            print(f"No Validation AUROC Improved, epoch: {epoch}")
            count += 1

        if (count % reset_count) == 0 and count > 0:
            model.load_state_dict(best_model_wts)
            print('Best Weights loaded')

        if count >= max_count:
            print('Early Stopping. Num of epochs: {:.4f}'.format(epoch))
            break

    print("Best AUROC: {:.4f}".format(best_epoch_auroc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_loss(hist):

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(8,8))
    sns.lineplot(data=hist, x='epoch', y='Train Loss', label='Train Loss', ax=ax)
    sns.lineplot(data=hist, x='epoch', y='Valid Loss', label='Valid Loss', legend=True, ax=ax).set_title('Loss')

def plot_auroc(hist):

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(8,8))
    sns.lineplot(data=hist, x='epoch', y='Train AUROC', label='Train auroc', ax=ax)
    sns.lineplot(data=hist, x='epoch', y='Valid AUROC', label='Valid auroc', legend=True, ax=ax).set_title('AUROC')

def plot_lr(hist):

    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(5,5))
    sns.lineplot(data=hist, x='epoch', y='lr', ax=ax).set_title('Learning Rate')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))