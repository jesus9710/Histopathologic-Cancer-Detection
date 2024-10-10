#%% libs
import numpy as np
import pandas as pd
from pathlib import Path

from utils import *

import argparse
import sys

# %% Paths

ROOT_PATH = Path(__file__).parents[2]

CONFIG_PATH = ROOT_PATH / 'config.yml'
PREDICTION_PATH = ROOT_PATH / 'Predictions'
TEST_IMAGES_PATH = ROOT_PATH / 'Data/test-images'
ENSEMBLE_SUBMISSION_PATH = ROOT_PATH / 'Submissions/Ensemble'

# %% argparse config

# Uncomment for debugging
# sys.argv = ['Ensemble_predictions.py']

parser = argparse.ArgumentParser(description="Script for ensembling")

parser.add_argument('--input', type=str, default=str(PREDICTION_PATH), help='Predictions directory')
parser.add_argument('--output', type=str, default=str(ENSEMBLE_SUBMISSION_PATH), help='Output ensemble directory')
parser.add_argument('--config', type=str, default=str(CONFIG_PATH), help='Config file')
parser.add_argument('--submission-name', type=str, help="Submission file name")
parser.add_argument('--weighted-avg', type=int, help='Indicates weather or not to use weights for avering ensemble')
parser.add_argument('--weights', type=str, help='Weights for wheigted average ensemble')

args = parser.parse_args()

PREDICTION_PATH = Path(args.input)
ENSEMBLE_SUBMISSION_PATH = Path(args.output)
CONFIG_PATH =  Path(args.config)

config = load_config(CONFIG_PATH)

weighted_avg = args.weighted_avg if args.weighted_avg else config.model.ensemble.weighted_avg
weights = [float(w) for w in args.weights.split(sep=' ')] if args.weights else config.model.ensemble.weights
submission_name = args.submission_name if args.submission_name else config.model.ensemble.submission_name

# %% submissions to ensemble

df_test = [file.name for file in TEST_IMAGES_PATH.iterdir() if file.is_file()]
df_test = [file.split(sep='.')[0] for file in df_test]
df_test = pd.DataFrame({'id':df_test})

submission_list = sorted([file for file in PREDICTION_PATH.glob('*.csv')])
submission_acr = [file.name.split(sep='.')[0] for file in submission_list] 

for submission, acr in zip(submission_list, submission_acr):
    df = pd.read_csv(submission, header=0, names=['id',acr+'_label'])
    df_test = df_test.merge(df, on='id',how='inner')

# %% Ensemble

if weighted_avg:
    df_test['label'] = df_test.drop('id', axis=1).apply(lambda x: np.average(x, weights= np.array(weights)), axis=1)

else:
    df_test['label'] = df_test.drop('id', axis=1).mean(axis=1)

df_test = df_test[['id','label']]

# %% submission

df_test.to_csv(ENSEMBLE_SUBMISSION_PATH / submission_name, index=False)

# %%
