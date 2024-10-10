# %% libs

import numpy as np
import pandas as pd

import torch
import torch.utils
import torch.utils.data
from torch.optim import lr_scheduler
import torch.utils.data.dataloader

from sklearn.model_selection import StratifiedGroupKFold

from pathlib import Path

from utils import *
from transformations import *
from model_definition import *

import os

# %% paths and configuration

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'Config.yml'

config = load_config(CONFIG_PATH)

TRAIN_IMAGES_PATH = ROOT_PATH / 'Data/train-images'
TEST_IMAGES_PATH = ROOT_PATH / 'Data/test-images'
LABELS_PATH = ROOT_PATH / 'Data/raw-metadata/train_labels.csv'
ID_WSI_PATH = ROOT_PATH / 'Data/raw-metadata/patch_id_wsi_full.csv'

MODEL_LOGS_PATH = ROOT_PATH / 'Models/' / (config.model.architecture.model_name + '/')
MODEL_CV_LOGS_PATH = ROOT_PATH / 'Models/Cross Val Ensemble/' / (config.model.architecture.model_name + '/')

SUBMISSION_PATH = ROOT_PATH / 'Submissions'

# set_seed(config.misc.seed)

# %% load data

df = pd.read_csv(LABELS_PATH)
id_wsi_df = pd.read_csv(ID_WSI_PATH)

df = df.merge(id_wsi_df, on='id', how='left')

# %% K-folds

sgkf = StratifiedGroupKFold(n_splits=config.data.sampling.n_fold)

for fold, ( _, val_) in enumerate(sgkf.split(df, df.label, groups=df.wsi)):
      df.loc[val_ , "kfold"] = fold

df.kfold = df.kfold.astype('int')

# %% Augmentations

data_transforms = get_transforms(config.data.parameters.img_size)

# %% Modelling

if __name__ == '__main__':

    if not config.model.architecture.load_model:
        model = maping_model[config.model.architecture.model_class](config.model.architecture.model_name,
                                                                freeze_params = config.model.parameters.freeze_params,
                                                                device = config.misc.device,
                                                                **config.model.architecture.kwargs)
        model.to(config.misc.device)

        original_state_dict = model.state_dict()

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
        original_state_dict = state_dict

    # optimizer and criterion

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.parameters.learning_rate, 
                        weight_decay=config.model.parameters.weight_decay)

    # Training

    scores = []

    if config.model.parameters.save_checkpoints:
        os.makedirs(MODEL_CV_LOGS_PATH, exist_ok=True)

    for fold in range(config.model.predictions.cv_folds):

        print("\nTraining fold " + str(fold) + ":\n")

        # create datasets and dataloaders

        df_train = df[df.kfold == (fold+1)].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)

        if config.data.sampling.valid_downsample:

            half_len_dataset = int(np.floor(config.data.sampling.downsample_quantity / 2))

            df_positive = df_valid[df_valid['label'] == 1].sample(half_len_dataset, random_state=config.misc.seed)
            df_negative = df_valid[df_valid['label'] == 0].sample(half_len_dataset, random_state=config.misc.seed)

            num_positive = df_positive.shape[0]

            df_valid = pd.concat([df_positive, df_negative], axis=0).reset_index(drop=True)

        if config.data.sampling.Random_sampling:
            train_dataset = HCD_Dataset_for_training(TRAIN_IMAGES_PATH, df_train, data_size=config.data.sampling.Rnd_sampling_q, transforms=data_transforms["train"])

        else:
            train_dataset = HCD_Dataset(TRAIN_IMAGES_PATH, df_train, transforms=data_transforms["train"])

        valid_dataset = HCD_Dataset(TRAIN_IMAGES_PATH, df_valid, transforms=data_transforms["valid"])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.data.parameters.train_batch_size, 
                                    num_workers=config.data.parameters.num_workers, pin_memory=config.data.parameters.pin_memory, shuffle=True, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.data.parameters.valid_batch_size, 
                                    num_workers=config.data.parameters.num_workers, pin_memory=config.data.parameters.pin_memory, shuffle=False, drop_last=False)
        
        # lr scheduler

        if config.model.parameters.scheduler == 'CosineAnnealing':

            if config.data.sampling.Random_sampling:

                T_max = config.data.sampling.Rnd_sampling_q * config.model.parameters.epochs // config.data.parameters.train_batch_size
                
            else:

                T_max = df.shape[0] * (config.data.sampling.n_fold-1) * config.model.parameters.epochs // config.data.parameters.train_batch_size // config.data.sampling.n_fold
            
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.model.parameters.min_lr)


        elif config.model.parameters.scheduler == 'OneCycle':

            scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=config.model.parameters.learning_rate, epochs=config.model.parameters.epochs, steps_per_epoch=int(np.floor(len(train_dataset)/config.data.parameters.train_batch_size)))

        else:

            scheduler = None

        # Train models

        if config.model.cross_val.retrain_cv:

            model_list = (MODEL_CV_LOGS_PATH / ('fold' + str(fold))).iterdir()
            model_list = [f.name for f in model_list if f.is_file()]
            best_model, best_auroc = get_best_auroc_scored_model(model_list)
            best_model_path = MODEL_CV_LOGS_PATH / ('fold' + str(fold)) / best_model
            model.load_state_dict(torch.load(best_model_path))
            retrain_from = best_auroc

        else:

            model.load_state_dict(original_state_dict)
            retrain_from = config.model.parameters.retrain_from_auroc

        if config.model.parameters.save_checkpoints:
            os.makedirs(MODEL_CV_LOGS_PATH / ('fold' + str(fold)), exist_ok=True)
            save_path = MODEL_CV_LOGS_PATH / ('fold' + str(fold))
        else:
            save_path = None

        model, history = train(model,
                            epochs = config.model.parameters.epochs,
                            criterion = criterion,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            train_dataloader = train_loader,
                            val_dataloader = valid_loader,
                            early_stopping = config.model.parameters.es_count,
                            early_reset= config.model.parameters.es_reset,
                            min_eta = config.model.parameters.min_eta,
                            cv_fold = fold,
                            swa = None,
                            from_auroc= retrain_from,
                            save_path = save_path,
                            device=config.misc.device)
        
        scores.append(max(history["Valid AUROC"]))

    print('CV Score: {:.4f}'.format(np.mean(scores)))
# %%