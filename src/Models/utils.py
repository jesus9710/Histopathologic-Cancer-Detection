import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from torchvision.transforms import v2 as transforms_v2
from torcheval.metrics.functional import binary_auroc

from PIL import Image
from copy import deepcopy
from collections import defaultdict

import os
import gc

import yaml

class Dotenv:
    def __init__(self, dictionary):
        dict_variables = ['kwargs']
        self.__dict__.update(dictionary)
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                if not(key in dict_variables):
                    self.__dict__[key] = Dotenv(value)

class HCD_Dataset_for_training(torch.utils.data.Dataset):

    def __init__(self, data_path, df, data_size, transforms = None):

        super(HCD_Dataset_for_training, self).__init__()

        self.data_path = data_path
        self.positive_df = df[df['label'] == 1]
        self.negative_df = df[df['label'] == 0]
        self.size = int(np.floor(data_size / 2))
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

        return {'image': image.float(), 'target': label.float()}

class HCD_Dataset(torch.utils.data.Dataset):

    def __init__(self, data_path, df, transforms = None):

        super(HCD_Dataset, self).__init__()

        self.data_path = data_path
        self.labels = df['label']
        self.image_id = df['id']
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
            
        return {'image': image.float(), 'target': label.float()}

class SWA:
    
    def __init__(self, model, optimizer, scheduler, learning_rate, start):

        self.model = torch.optim.swa_utils.AveragedModel(model=model)
        self.start = start
        self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer=optimizer, swa_lr=learning_rate)
        self.scheduler  = scheduler

    def step(self, model, epoch):
        if epoch == self.start:
            self.model = torch.optim.swa_utils.AveragedModel(model=model)

        elif epoch > self.start:
          self.model.update_parameters(model)
          self.swa_scheduler.step()

        else:
          self.scheduler.step()

    def validate(self, model, train_loader, valid_loader, criterion, epoch, device):

        if epoch > self.start:
            update_bn(train_loader, self.model, device)
            val_loss, val_auroc = eval_one_epoch(self.model, criterion, valid_loader, device)

        else:
            val_loss, val_auroc = eval_one_epoch(model, criterion, valid_loader)
        
        return val_loss, val_auroc
    
    def get_current_model(self, model, epoch):
        
        if epoch > self.start:
            return self.model, 1
        
        else:
            return model, 0

def load_config(config_path):

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    config = Dotenv(config)

    config.misc.device = torch.device('cuda') if config.misc.device == 'cuda' else torch.device('cpu')

    return config

def set_seed(seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_Tmax(dataset, config):

    if config.data.sampling.Random_sampling:
        T_max = config.data.sampling.Rnd_sampling_q * config.model.parameters.epochs // config.data.parameters.train_batch_size
        
    elif config.model.parameters.SWA_enable:
        T_max = config.model.parameters.SWA_start

    else:
        T_max = len(dataset) * (config.data.sampling.n_fold-1) * config.model.parameters.epochs // config.data.parameters.train_batch_size // config.data.sampling.n_fold
    
    return T_max

def get_scheduler(dataset, optimizer, config):

    if config.model.parameters.scheduler == 'CosineAnnealing':
        T_max = get_Tmax(dataset, config)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.model.parameters.min_lr)

    elif config.model.parameters.scheduler == 'OneCycle':

        if config.model.parameters.SWA_enable:
            steps_per_epoch = 1
            epochs = config.model.parameters.SWA_start

        else:
            epochs=config.model.parameters.epochs
            steps_per_epoch=int(np.floor(len(dataset)/config.data.parameters.train_batch_size))

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.model.parameters.learning_rate, epochs=epochs, steps_per_epoch=steps_per_epoch)

    else:
        scheduler = None
    
    return scheduler

def train_one_epoch(model, criterion, optimizer, dataloader, scheduler = None, device = torch.device('cuda')):

    model.train()

    loss_hist = []
    out_hist = []
    label_hist = []

    for data in dataloader:

        images = data['image'].to(device)
        targets = data['target'].to(device)

        optimizer.zero_grad()
        
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if scheduler:
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

def eval_one_epoch(model, criterion, dataloader, device=torch.device('cuda')):

    model.eval()
    loss_hist = []
    out_hist = []
    label_hist = []

    with torch.no_grad():
        for data in dataloader:

            images = data['image'].to(device)
            targets = data['target'].to(device)

            output = model(images).squeeze()
            loss = criterion(output, targets)
            loss_hist.append(loss.item())

            out_hist.append(output)
            label_hist.append(targets)
        
        out_hist = torch.cat(out_hist)
        label_hist = torch.cat(label_hist)

        loss = np.array(loss_hist).mean()
        auroc = binary_auroc(input=out_hist, target=label_hist).item()

        gc.collect()
    
    return loss, auroc

def predict_model(model, dataloader, device=torch.device('cuda')):

    model.eval()

    predictions = []

    with torch.no_grad():
        for data in dataloader:

            images = data['image'].to(device)
            outputs = model(images).squeeze()
            predictions.append(outputs)

        soft_preds = torch.cat(predictions).cpu()
    
    gc.collect()

    return soft_preds

def save_model_if_better_auroc(model, epoch, best_epoch_auroc, best_model_wts, val_loss, val_auroc, save_path, min_eta = 0, pre_name = '', post_name= ''):

    if (best_epoch_auroc + min_eta) <= val_auroc:

        print(f"Validation AUROC Improved ({best_epoch_auroc} ---> {val_auroc}), epoch: {epoch}")
        
        best_epoch_auroc = val_auroc
        best_model_wts = deepcopy(model.state_dict())
        FILE = pre_name + "AUROC{:.4f}_Loss{:.4f}".format(val_auroc, val_loss) + post_name + ".bin"
        
        torch.save(model.state_dict(), save_path / FILE)

        flag = 1

    else:
        print(f"No Validation AUROC Improved, epoch: {epoch}")
        flag = 0

    return best_model_wts, best_epoch_auroc, flag

def update_history(history, optimizer, epoch, train_loss = None, val_loss = None, train_auroc = None, val_auroc = None, scheduler = None):

    current_lr = scheduler.get_lr()[0] if scheduler else optimizer.param_groups[0]['lr']

    history['epoch'].append(epoch)
    history['Train Loss'].append(train_loss)
    history['Valid Loss'].append(val_loss)
    history['Train AUROC'].append(train_auroc)
    history['Valid AUROC'].append(val_auroc)
    history['lr'].append(current_lr)

    return history

def standard_train(model, epochs, criterion, optimizer, train_dataloader, val_dataloader = None, scheduler = None, early_stopping = 10, early_reset = None, min_eta = 1e-3, cv_fold = None, save_path = None, from_auroc = None, device= torch.device('cuda'), **kwargs):

    best_model_wts = deepcopy(model.state_dict())

    pre_name = '' if from_auroc is None else 'ft_'
    post_name = '' if cv_fold is None else f'_Fold{cv_fold}'
    max_count = np.inf if early_stopping is None else early_stopping
    reset_count = np.inf if early_reset is None else early_reset
    best_epoch_auroc = -np.inf if from_auroc is None else from_auroc

    history = defaultdict(list)
    max_count = early_stopping
    count = 0

    for epoch in range(1, epochs + 1):

        # Train and validate one epoch
        train_loss, train_auroc = train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, dataloader=train_dataloader, device=device)
        val_loss, val_auroc = eval_one_epoch(model=model, criterion=criterion, dataloader=val_dataloader, device=device)

        # Update history
        history = update_history(history, optimizer, epoch, train_loss, val_loss, train_auroc, val_auroc, scheduler)

        # Save model if validation metric is improved
        best_model_wts, best_epoch_auroc, improved = save_model_if_better_auroc(model, epoch, best_epoch_auroc, best_model_wts, val_loss, val_auroc, save_path, min_eta, pre_name, post_name)

        count = count + 1 if not improved else 0

        # Load best model weights if no improvement during "reset_count" epochs
        if (count % reset_count == 0 and count > 0):
            model.load_state_dict(best_model_wts)
            print('Best Weights loaded')

        # Early stopping
        if count >= max_count:
            print('Early Stopping. Num of epochs: {:.4f}'.format(epoch))
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    print("Best AUROC: {:.4f}".format(best_epoch_auroc))

    return model, history

def swa_train(model, epochs, criterion, optimizer, swa, train_dataloader, val_dataloader = None, scheduler = None, early_reset = None, min_eta = 1e-3, save_path = None, from_auroc = None, device= torch.device('cuda'), **kwargs):

    best_model_wts = deepcopy(model.state_dict())
    best_epoch_auroc = -np.inf if from_auroc is None else from_auroc
    history = defaultdict(list)

    reset_count = np.inf if early_reset is None else early_reset
    count = 0

    for epoch in range(1, epochs + 1):
        
        # Train one epoch
        train_loss, train_auroc = train_one_epoch(model=model, criterion=criterion, optimizer=optimizer, dataloader=train_dataloader, device=device)

        # Validation and SWA parameters update
        swa.step(model, epoch)
        model_to_save, swa_flag = swa.get_current_model(model, epoch)
        pre_name = 'swa_' if swa_flag else ''

        # During normal training:
        if not swa_flag:

            # Save model if validation metric is improved during standard training
            val_loss, val_auroc = swa.validate(model=model, criterion=criterion, train_loader = train_dataloader, valid_loader =val_dataloader, epoch=epoch, device=device)
            best_model_wts, best_epoch_auroc, improved = save_model_if_better_auroc(model_to_save, epoch, best_epoch_auroc, best_model_wts, val_loss, val_auroc, save_path, min_eta, pre_name)
            count = count + 1 if not improved else 0

            # Update history
            history = update_history(history, optimizer, epoch, train_loss, val_loss, train_auroc, val_auroc, scheduler)

             # Load best weights if no improvements during "reset_count" epochs (standard train)
            if (count % reset_count == 0 and count > 0):
                model.load_state_dict(best_model_wts)
                print('Best Weights loaded')
                
        else:
            # validation is not performed in order to reduce training time
            print(f'SWA Training, epoch: {epoch}')
    
    model = swa.model

    # Calculate batch-normalization parameters
    update_bn(train_dataloader, model, device)

    # Save SWA model
    torch.save(model.state_dict(), save_path / ('SWA_Final_weights_epochs_' + str(epochs) + '.bin'))
    print(f"SWA Training ended")

    return model, history

def train(swa, **kwargs):
    '''
    Function for selecting the training mode according to the entered argument
    '''
    if swa:
        return swa_train(swa = swa, **kwargs)

    else:
        return standard_train(**kwargs)

def get_best_auroc_scored_model(model_list):

    AUROCS =[float(file.split(sep='AUROC')[-1].split(sep='_')[0]) for file in model_list]

    best_model = model_list[np.argmax(AUROCS)]

    return best_model, max(AUROCS)

@torch.no_grad()
def update_bn(loader, model, device=None):

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        if isinstance(input, (list, tuple)):
            input = input[0]
        if isinstance(input, dict):
            input = input['image']
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)