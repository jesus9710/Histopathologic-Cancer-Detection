# %% libs

import numpy as np
import pandas as pd

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedGroupKFold

from pathlib import Path
import yaml

from utils import *
from model_definition import *

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

# %% Parameters

n_folds = 5

epochs = 15
data_size = 500
batch_size = 50

learning_rate = 1e-4
weight_decay = 1e-5

# %% load data

df = pd.read_csv(LABELS_PATH)
id_wsi_df = pd.read_csv(ID_WSI_PATH)

# df = df.merge(id_wsi_df, on='id', how='left')
df = pd.merge(df, id_wsi_df, on='id')

# %% K-folds

sgkf = StratifiedGroupKFold(n_splits=n_folds)

for fold, ( _, val_) in enumerate(sgkf.split(df, df.label, groups=df.wsi)):
      df.loc[val_ , "kfold"] = int(fold)

df.kfold = df.kfold.astype('int')

# %% Augmentations

data_transforms_1 = {
    "train": A.Compose([
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.HorizontalFlip(p=0.2),
        A.HueSaturationValue(
                hue_shift_limit=0.1, 
                sat_shift_limit=0.1, 
                val_shift_limit=0.1, 
                p=0.3
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.3
            ),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(5.0, 10.0)),
            ], p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.7),
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

data_transforms_2 = {
    "train": A.Compose([
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.2),
        A.HorizontalFlip(p=0.2),
        A.HueSaturationValue(
                hue_shift_limit=0.1, 
                sat_shift_limit=0.1, 
                val_shift_limit=0.1, 
                p=0.3
            ),
        A.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.3
            ),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(5.0, 10.0)),
            ], p=0.3),
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

# %% functions

def create_model():

    model = HCD_Simple_Model(model_name='simple_model')
    model.to(config.misc.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, 
                        weight_decay= weight_decay)

    return model, optimizer

def create_loaders(train_data, val_data, data_size, batch_size):

    train_dataset_1 = HCD_Dataset_for_training(TRAIN_IMAGES_PATH, train_data, data_size=data_size, transforms=data_transforms_1["train"])
    valid_dataset_1 = HCD_Dataset_for_training(TRAIN_IMAGES_PATH, val_data, data_size=data_size, transforms=data_transforms_1["valid"])

    train_dataset_2 = HCD_Dataset_for_training(TRAIN_IMAGES_PATH, train_data, data_size=data_size, transforms=data_transforms_2["train"])
    valid_dataset_2 = HCD_Dataset_for_training(TRAIN_IMAGES_PATH, val_data, data_size=data_size, transforms=data_transforms_2["valid"])


    train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=batch_size)
    valid_loader_1 = torch.utils.data.DataLoader(valid_dataset_1, batch_size=batch_size)

    train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=batch_size)
    valid_loader_2 = torch.utils.data.DataLoader(valid_dataset_2, batch_size=batch_size)

    return train_loader_1, valid_loader_1, train_loader_2, valid_loader_2

# %% Training

criterion = torch.nn.BCELoss()

scores_1 = []
scores_2 = []

for fold in range(n_folds):

    

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    train_loader_1, valid_loader_1, train_loader_2, valid_loader_2 = create_loaders(df_train, df_valid, data_size=data_size, batch_size= batch_size)

    model, optimizer = create_model()

    print("Training model_1:\n")

    model_1, history_1 = train(model, epochs = epochs, criterion = criterion, optimizer = optimizer, train_dataloader = train_loader_1, val_dataloader = valid_loader_1, early_stopping = 10)

    print('\n')

    model, optimizer = create_model()

    print("Training model_2:\n")

    model_2, history_2 = train(model, epochs = epochs, criterion = criterion, optimizer = optimizer, train_dataloader = train_loader_2, val_dataloader = valid_loader_2, early_stopping = 10)

    print('\n')

    scores_1.append(max(history_1['Valid AUROC']))
    scores_2.append(max(history_2['Valid AUROC']))

score_1 = np.mean(scores_1)
score_2 = np.mean(scores_2)

print("Mean CV auroc score with transformation set 1: {:.4f}".format(score_1))
print("Mean CV auroc score with transformation set 2: {:.4f}".format(score_2))
# %%
