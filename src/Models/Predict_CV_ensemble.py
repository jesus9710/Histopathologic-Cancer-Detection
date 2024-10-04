# %% libs

import numpy as np
import pandas as pd

import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

from pathlib import Path

from utils import *
from transformations import *
from model_definition import *

import argparse
import sys

# %% default paths

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'Config.yml'
TEST_IMAGE_PATH = ROOT_PATH / 'Data/test-images'
PREDICTION_PATH = ROOT_PATH / 'Predictions'

# %% argparse config

config = load_config(CONFIG_PATH)

# Uncomment for debugging
# sys.argv = ['Predict_CV_ensemble.py']

parser = argparse.ArgumentParser(description="Script for predictions")

parser.add_argument('--model-type', type=str, help='Model architecture class name')
parser.add_argument('--model-name', type=str, help='Model name')
parser.add_argument('--input', type=str, default=str(TEST_IMAGE_PATH), help='Test image directory')
parser.add_argument('--output', type=str, default=str(PREDICTION_PATH), help='Save predictions directory')
parser.add_argument('--config', type=str, default=str(CONFIG_PATH), help='Config file')
parser.add_argument('--submission-name', type=str, help='Submission file name')
parser.add_argument('--tta', type=int, help='Enables tta predictions. set 1 for tta predictions or 0 for regular prediction')

args = parser.parse_args()

config = load_config(CONFIG_PATH)

model_type = args.model_type if args.model_type else config.model.predictions.model_class
model_name = args.model_name if args.model_name else config.model.predictions.model_name
submission_name = args.submission_name if args.submission_name else config.model.predictions.submission_name
tta = args.tta if not(args.tta is None) else config.model.predictions.tta

TEST_IMAGE_PATH = Path(args.input)
PREDICTION_PATH = Path(args.output)
MODEL_CV_LOGS_PATH = ROOT_PATH / 'Models/Cross Val Ensemble/' / model_name

# %% Transformations

if config.model.predictions.tta:

    transforms = get_tta_transforms(config.data.parameters.img_size)

else:

    transforms = [get_transforms(config.data.parameters.img_size)['valid']]

# %% Load data

df_test = [file.name for file in TEST_IMAGE_PATH.iterdir() if file.is_file()]
df_test = [file.split(sep='.')[0] for file in df_test]
df_test = pd.DataFrame({'id':df_test,'label':np.ones(len(df_test))*np.nan})

# %% predictions

if __name__ == '__main__':

    submissions_dict = {'id': df_test['id']}

    for idx, fold in enumerate(range(config.data.sampling.n_fold-1)):

        print(f'\nPredicting fold {idx}\n')

        model_list = (MODEL_CV_LOGS_PATH / ('fold' + str(fold))).iterdir()
        model_list = [f.name for f in model_list if f.is_file()]
        best_model, best_auroc = get_best_auroc_scored_model(model_list)

        checkpoint_path = MODEL_CV_LOGS_PATH / ('fold' + str(idx)) / best_model

        model = maping_model[model_type](model_name = model_name, pretrained=False, device = config.misc.device)
        model.to(config.misc.device)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)

        submission_df_tta = pd.DataFrame({'id':df_test['id']})
        
        for ix, transform in enumerate(transforms):

            test_dataset = HCD_Dataset(TEST_IMAGE_PATH, df_test, transforms=transform)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config.data.parameters.test_batch_size,
                                                    num_workers=config.data.parameters.num_workers, pin_memory=config.data.parameters.pin_memory, shuffle=False, drop_last=False)

            predictions = predict_model(model, test_loader, device=config.misc.device)

            if tta:

                submission_df_tta["label_"+str(ix+1)] = predictions

                if (ix+1) == len(transforms):

                    predictions = submission_df_tta.drop('id',axis=1).mean(axis=1)

                print(f"test images predicted. Process: {ix+1}/{len(transforms)}")

        submissions_dict.update({'label_fold_'+str(idx) : predictions})

    submission = pd.DataFrame(submissions_dict)
    submission['label'] = submission.drop('id',axis=1).mean(axis=1)
    submission = submission[['id','label']]

    # Submission

    submission.to_csv(PREDICTION_PATH / submission_name, index=False)
