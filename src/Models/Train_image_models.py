# %% libs

import numpy as np
import pandas as pd

import torch
import torch.utils
import torch.utils.data
from torch.optim import lr_scheduler
import torch.utils.data.dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold

from pathlib import Path
import yaml
import json

from utils import *
from model_definition import *

import os
import gc

# %% paths and configuration

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'src/Configuration/Config.yml'

with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

config = Dotenv(config)
config.misc.device = torch.device('cuda') if config.misc.device == 'cuda' else torch.device('cpu')

TRAIN_IMAGES_PATH = ROOT_PATH / 'Data/train-images'
TEST_IMAGES_PATH = ROOT_PATH / 'Data/test-images'
LABELS_PATH = ROOT_PATH / 'Data/raw-metadata/train_labels.csv'
ID_WSI_PATH = ROOT_PATH / 'Data/raw-metadata/patch_id_wsi_full.csv'

MODEL_LOGS_PATH = ROOT_PATH / 'Models/' / (config.model.architecture.model_name + '/')

SUBMISSION_PATH = ROOT_PATH / 'Submissions'

# set_seed(config.misc.seed)

# %% load data

df = pd.read_csv(LABELS_PATH)
id_wsi_df = pd.read_csv(ID_WSI_PATH)

# df = df.merge(id_wsi_df, on='id', how='left')
df = pd.merge(df, id_wsi_df, on='id')

# %% K-folds

sgkf = StratifiedGroupKFold(n_splits=config.data.sampling.n_fold)

for fold, ( _, val_) in enumerate(sgkf.split(df, df.label, groups=df.wsi)):
      df.loc[val_ , "kfold"] = int(fold)

df.kfold = df.kfold.astype('int')

# %% Augmentations

data_transforms = {
    "train": A.Compose([
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.HorizontalFlip(p=0.3),
        A.HueSaturationValue(
                hue_shift_limit=0.1, 
                sat_shift_limit=0.1, 
                val_shift_limit=0.1, 
                p=0.2
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.2
            ),
        A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.),
    
    "valid": A.Compose([
        A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}


# %% Modelling

if not config.model.architecture.load_model:
    model = maping_model[config.model.architecture.model_class](config.model.architecture.model_name,
                                                               checkpoint_path = config.model.architecture.checkpoint_path,
                                                               freeze_params = config.model.parameters.freeze_params,
                                                               device = config.misc.device,
                                                               **config.model.architecture.kwargs)
    model.to(config.misc.device)

# %% Load model

checkpoint_path = MODEL_LOGS_PATH / config.model.architecture.model_to_load

if config.model.architecture.load_model:

    model = maping_model[config.model.architecture.model_class](config.model.architecture.model_name,
                                                                pretrained = False,
                                                                checkpoint_path = None,
                                                                freeze_params = config.model.parameters.freeze_params,
                                                                device = config.misc.device,
                                                                **config.model.architecture.kwargs)
    model.to(config.misc.device)

    state_dict = torch.load(checkpoint_path)

    model.load_state_dict(state_dict)

# %% optimizer and criterion

criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.model.parameters.learning_rate, 
                       weight_decay=config.model.parameters.weight_decay)

df_train = df[df.kfold != fold].reset_index(drop=True)
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
                            num_workers=0, shuffle=True, pin_memory=False, drop_last=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.data.parameters.valid_batch_size, 
                            num_workers=0, shuffle=False, pin_memory=False)

# %% lr scheduler

if config.model.parameters.scheduler == 'CosineAnnealingLR':

    T_max = df.shape[0] * (config.data.sampling.n_fold-1) * config.model.parameters.epochs // config.data.parameters.train_batch_size // config.data.sampling.n_fold # decaimiento suave a lo largo de todo el entrenamiento
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=config.model.parameters.min_lr)

elif config.model.parameters.scheduler == 'OneCycle':

    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=config.model.parameters.learning_rate, epochs=config.model.parameters.epochs, steps_per_epoch=int(np.floor(len(train_dataset)/config.data.parameters.train_batch_size)))

else:

    scheduler = None

# %% Training

if config.model.mode.train_model:

    if config.model.mode.save_checkpoints:
        os.makedirs(MODEL_LOGS_PATH, exist_ok=True)
        save_path = MODEL_LOGS_PATH
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
                        from_auroc= config.model.parameters.retrain_from_auroc,
                        save_path = save_path)

# %% predictions

torch.cuda.empty_cache()
gc.collect()

df_test = [file.name for file in TEST_IMAGES_PATH.iterdir() if file.is_file()]
df_test = [file.split(sep='.')[0] for file in df_test]
df_test = pd.DataFrame({'id':df_test,'label':np.ones(len(df_test))*np.nan})

test_dataset = HCD_Dataset(TEST_IMAGES_PATH, df_test, transforms=data_transforms["valid"])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config.data.parameters.test_batch_size)

soft_preds = predict_model(model, test_loader)

submission = pd.DataFrame({'id': df_test['id']})
submission['label'] = soft_preds.cpu()

# %% plottings

plot_loss(history)
plot_auroc(history)
plot_lr(history)

# %% Save history

hist_name = 'history.json' if config.model.parameters.retrain_from_auroc is None else 'ft_history.json'

history_path = MODEL_LOGS_PATH / hist_name

if config.misc.save_history:

    with open(history_path, 'w', encoding='utf-8') as file:
        json.dump(history, file, ensure_ascii=False, indent=4)

# %% Submission

if config.misc.save_submission:
    submission.to_csv(SUBMISSION_PATH / config.misc.submission_name, index=False)

# %%
