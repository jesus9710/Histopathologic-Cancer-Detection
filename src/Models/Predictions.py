#%% libs

import numpy as np
import pandas as pd
import torch

from pathlib import Path

from model_definition import *
from transformations import *
from utils import *

import argparse
import sys

# %% defaults paths

ROOT_PATH = Path(__file__).parents[2]
CONFIG_PATH = ROOT_PATH / 'config.yml'
TEST_IMAGE_PATH = ROOT_PATH / 'Data/test-images'
PREDICTION_PATH = ROOT_PATH / 'Predictions'

# %% argparse config

# Uncomment for debugging
# sys.argv = ['Predictions.py']

parser = argparse.ArgumentParser(description="Script for predictions")

parser.add_argument('--model-type', type=str, help='Model architecture class name')
parser.add_argument('--model-name', type=str, help='Model name')
parser.add_argument('--model-file', type=str, help='File containing models weights')
parser.add_argument('--input', type=str, default=str(TEST_IMAGE_PATH), help='Test image directory')
parser.add_argument('--output', type=str, default=str(PREDICTION_PATH), help='Save predictions directory')
parser.add_argument('--config', type=str, default=str(CONFIG_PATH), help='Config file')
parser.add_argument('--submission-name', type=str, help='Submission file name')
parser.add_argument('--tta', type=int, help='Enables tta predictions. set 1 for tta predictions or 0 for regular prediction')

args = parser.parse_args()

config = load_config(CONFIG_PATH)

model_type = args.model_type if args.model_type else config.model.predictions.model_class
model_name = args.model_name if args.model_name else config.model.predictions.model_name
model_file = args.model_file if args.model_file else config.model.predictions.file
submission_name = args.submission_name if args.submission_name else config.model.predictions.submission_name
tta = args.tta if not(args.tta is None) else config.model.predictions.tta

TEST_IMAGE_PATH = Path(args.input)
PREDICTION_PATH = Path(args.output)
MODEL_LOGS_PATH = ROOT_PATH / 'Models/' / model_name / model_file

# %% load data

test_files = [file.name.split(sep='.')[0] for file in TEST_IMAGE_PATH.iterdir()]
test_df = pd.DataFrame({'id':test_files, 'label':np.ones(len(test_files))*np.nan})

# %% Transformations

if tta:
    transforms = get_tta_transforms(config.data.parameters.img_size)

else:
    transforms = [get_transforms(config.data.parameters.img_size)['valid']]

# %% Predictions

if __name__ == '__main__':

    # Load Model

    model = maping_model[model_type](model_name=model_name, pretrained=False, device = config.misc.device)

    save_state = torch.load(MODEL_LOGS_PATH)

    model.load_state_dict(save_state)
    model.to(config.misc.device)

    # predictions

    submission_df = pd.DataFrame({'id':test_df['id']})

    for ix, transform in enumerate(transforms):

        test_dataset = HCD_Dataset(TEST_IMAGE_PATH, test_df, transforms=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, config.data.parameters.test_batch_size,
                                                  num_workers=config.data.parameters.num_workers, pin_memory=config.data.parameters.pin_memory, shuffle=False, drop_last=False)

        predictions = predict_model(model, test_loader, device=config.misc.device)

        if tta:

            submission_df["label_"+str(ix+1)] = predictions
            
            if (ix+1) == len(transforms):

                submission_df['label'] = submission_df.drop('id',axis=1).mean(axis=1)

            print(f"test images predicted. Process: {ix+1}/{len(transforms)}")

        else:
            submission_df["label"] = predictions

    submission = submission_df[['id','label']]

    # submission

    submission_name = PREDICTION_PATH / submission_name

    submission.to_csv(submission_name, index=False)

# %%
