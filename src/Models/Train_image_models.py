# %% libs

import numpy as np
import pandas as pd

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

from sklearn.model_selection import StratifiedGroupKFold

from pathlib import Path
import json

from utils import *
from transformations import *
from model_definition import *

import os

# %% paths and configuration

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'config.yml'

config = load_config(CONFIG_PATH)

TRAIN_IMAGES_PATH = ROOT_PATH / 'Data/train-images'
TEST_IMAGES_PATH = ROOT_PATH / 'Data/test-images'
LABELS_PATH = ROOT_PATH / 'Data/raw-metadata/train_labels.csv'
ID_WSI_PATH = ROOT_PATH / 'Data/raw-metadata/patch_id_wsi_full.csv'
SUBMISSION_PATH = ROOT_PATH / 'Submissions'
MODEL_LOGS_PATH = ROOT_PATH / 'Models/' / (config.model.architecture.model_name + '/')

set_seed(config.misc.seed)

# %% load data

df = pd.read_csv(LABELS_PATH)
id_wsi_df = pd.read_csv(ID_WSI_PATH)

df = df.merge(id_wsi_df, on='id', how='left')

# %% K-folds

sgkf = StratifiedGroupKFold(n_splits=config.data.sampling.n_fold)

for fold, ( _, val_) in enumerate(sgkf.split(df, df.label, groups=df.wsi)):
      df.loc[val_ , "kfold"] = int(fold)

df.kfold = df.kfold.astype('int')

# %% Augmentations

train_data_transforms = get_transforms(config.data.parameters.img_size, validation=False)
valid_data_transforms = get_transforms(config.data.parameters.img_size, validation=True)

# %% Training

if __name__ == '__main__':

    # Modelling

    if not config.model.architecture.load_model:
        model = maping_model[config.model.architecture.model_class](config.model.architecture.model_name,
                                                                freeze_params = config.model.parameters.freeze_params,
                                                                device = config.misc.device,
                                                                **config.model.architecture.kwargs)
        model.to(config.misc.device)

    # Load model

    checkpoint_path = MODEL_LOGS_PATH / config.model.architecture.model_to_load

    if config.model.architecture.load_model:

        model = maping_model[config.model.architecture.model_class](config.model.architecture.model_name,
                                                                    pretrained = False,
                                                                    freeze_params = config.model.parameters.freeze_params,
                                                                    device = config.misc.device,
                                                                    **config.model.architecture.kwargs)
        model.to(config.misc.device)

        state_dict = torch.load(checkpoint_path)

        model.load_state_dict(state_dict)

    # optimizer and criterion

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.parameters.learning_rate, 
                        weight_decay=config.model.parameters.weight_decay)

    # Dataloaders

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    if config.data.sampling.valid_downsample:

        half_len_dataset = int(np.floor(config.data.sampling.downsample_quantity / 2))

        df_positive = df_valid[df_valid['label'] == 1].sample(half_len_dataset, random_state=config.misc.seed)
        df_negative = df_valid[df_valid['label'] == 0].sample(half_len_dataset, random_state=config.misc.seed)

        df_valid = pd.concat([df_positive, df_negative], axis=0).reset_index(drop=True)

    if config.data.sampling.Random_sampling:
        train_dataset = HCD_Dataset_for_training(TRAIN_IMAGES_PATH, df_train, data_size=config.data.sampling.Rnd_sampling_q, transforms=train_data_transforms)

    else:
        train_dataset = HCD_Dataset(TRAIN_IMAGES_PATH, df_train, transforms=train_data_transforms)

    valid_dataset = HCD_Dataset(TRAIN_IMAGES_PATH, df_valid, transforms=valid_data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.data.parameters.train_batch_size, 
                                num_workers=config.data.parameters.num_workers, pin_memory=config.data.parameters.pin_memory, shuffle=True, drop_last=False)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.data.parameters.valid_batch_size, 
                                num_workers=config.data.parameters.num_workers, pin_memory=config.data.parameters.pin_memory, shuffle=False, drop_last=False)

    # lr scheduler

    scheduler = get_scheduler(train_dataset, optimizer, config)

    if config.model.swa.SWA_enable:
        swa_strat = SWA(model, optimizer, scheduler, config.model.swa.SWA_lr, config.model.swa.SWA_start)
    else:
        swa_strat = None
        
    # Training

    if config.model.parameters.save_checkpoints:
        os.makedirs(MODEL_LOGS_PATH, exist_ok=True)
        save_path = MODEL_LOGS_PATH
    else:
        save_path = None

    model, history = train(model = model,
                        epochs = config.model.parameters.epochs,
                        criterion = criterion,
                        optimizer = optimizer,
                        scheduler = scheduler,
                        train_dataloader = train_loader,
                        val_dataloader = valid_loader,
                        swa=swa_strat,
                        early_stopping = config.model.parameters.es_count,
                        early_reset= config.model.parameters.es_reset,
                        min_eta = config.model.parameters.min_eta,
                        from_auroc= config.model.parameters.retrain_from_auroc,
                        save_path = save_path,
                        device=config.misc.device)

    # Save history

    hist_name = config.model.parameters.history_name if config.model.parameters.retrain_from_auroc is None else f"ft_{config.model.parameters.history_name}"

    history_path = MODEL_LOGS_PATH / hist_name

    if config.model.parameters.save_history:

        with open(history_path, 'w', encoding='utf-8') as file:
            json.dump(history, file, ensure_ascii=False, indent=4)
