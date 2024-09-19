#%% libs

import numpy as np
import pandas as pd
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path
import yaml

from model_definition import *
from utils import *

import gc

# %% paths

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'src/Configuration/Config.yml'

MODEL_LOGS_PATH = ROOT_PATH / 'Models/'

TEST_IMAGE_PATH = ROOT_PATH / 'Data/test-images'

SUBMISSION_PATH = ROOT_PATH / 'Submissions/TTA'

with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

config = Dotenv(config)
config.misc.device = torch.device('cuda') if config.misc.device == 'cuda' else torch.device('cpu')

# %% load data

test_files = [file.name.split(sep='.')[0] for file in TEST_IMAGE_PATH.iterdir()]
test_df = pd.DataFrame({'id':test_files, 'label':np.ones(len(test_files))*np.nan})

# %% Model to TTA

model = maping_model[config.model.architecture.model_class](model_name=config.model.architecture.model_name, pretrained=False)

save_state = torch.load(MODEL_LOGS_PATH / config.model.architecture.model_name / config.model.tta.file)

model.load_state_dict(save_state)
model.to(config.misc.device)

# %% Transformations

tta_transforms = [

    A.Compose([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
               A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
               A.Normalize(
                   mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225], 
                   max_pixel_value=255.0, 
                   p=1.0),
               ToTensorV2()], p=1.),
    
    A.Compose([A.HorizontalFlip(p=1.0),
              A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
              A.Normalize(
                   mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225], 
                   max_pixel_value=255.0, 
                   p=1.0),
               ToTensorV2()], p=1.),
    
    A.Compose([A.VerticalFlip(p=1.0),
               A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
               A.Normalize(
                   mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225], 
                   max_pixel_value=255.0, 
                   p=1.0),
               ToTensorV2()], p=1.),
    
    A.Compose([A.RandomRotate90(p=1),
               A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
               A.Normalize(
                   mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225], 
                   max_pixel_value=255.0, 
                   p=1.0),
               ToTensorV2()], p=1.)
]

# %% predictions

model.eval()

submission_df = pd.DataFrame({'id':test_df['id']})

for ix, transform in enumerate(tta_transforms):

    test_dataset = HCD_Dataset(TEST_IMAGE_PATH, test_df, transforms=transform, device=config.misc.device)
    test_loader = torch.utils.data.DataLoader(test_dataset, config.data.parameters.test_batch_size)

    predictions = []

    with torch.no_grad():
        for data in test_loader:

            output = model(data['image']).squeeze()
            predictions.append(output)
    
        soft_preds = torch.cat(predictions).cpu()
        submission_df["label_"+str(ix+1)] = soft_preds

    gc.collect()

    print(f"test images predicted. Process: {ix+1}/{len(tta_transforms)}")

# %% submission

submission_name = SUBMISSION_PATH / ('tta_' + config.misc.submission_name)

submission_df['label'] = submission_df.drop('id',axis=1).mean(axis=1)
submission = submission_df[['id','label']]

submission.to_csv(submission_name, index=False)
# %%
