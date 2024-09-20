# %% libs

import numpy as np
import pandas as pd

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path
import yaml

from utils import *
from model_definition import *

import gc

# %% paths and configuration

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'src/Configuration/Config.yml'

with open(CONFIG_PATH, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

config = Dotenv(config)
config.misc.device = torch.device('cuda') if config.misc.device == 'cuda' else torch.device('cpu')

TEST_IMAGES_PATH = ROOT_PATH / 'Data/test-images'

MODEL_LOGS_PATH = ROOT_PATH / 'Models/' / (config.model.architecture.model_name + '/')

SUBMISSION_PATH = ROOT_PATH / 'Submissions'

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

# %% Transformations

data_transforms = {
    "test": A.Compose([
        A.Resize(config.data.parameters.img_size, config.data.parameters.img_size),
        A.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
        ToTensorV2()], p=1.)
}

# %% predictions

torch.cuda.empty_cache()
gc.collect()

df_test = [file.name for file in TEST_IMAGES_PATH.iterdir() if file.is_file()]
df_test = [file.split(sep='.')[0] for file in df_test]
df_test = pd.DataFrame({'id':df_test,'label':np.ones(len(df_test))*np.nan})

test_dataset = HCD_Dataset(TEST_IMAGES_PATH, df_test, transforms=data_transforms["test"])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config.data.parameters.test_batch_size)

soft_preds = predict_model(model, test_loader)

submission = pd.DataFrame({'id': df_test['id']})
submission['label'] = soft_preds.cpu()

# %% Submission

if config.misc.save_submission:
    submission.to_csv(SUBMISSION_PATH / config.misc.submission_name, index=False)

# %%
