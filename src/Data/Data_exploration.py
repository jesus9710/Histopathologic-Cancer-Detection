#%% libs

import numpy as np
import pandas as pd

from PIL import Image as PIL_Image
import h5py
from pathlib import Path


# %% paths and constants

ROOT_PATH = Path(__file__).parents[2]

TRAIN_METADATA_PATH = ROOT_PATH / 'Data/raw-metadata/camelyonpatch_level_2_split_train_meta.csv'
VALID_METADATA_PATH = ROOT_PATH / 'Data/raw-metadata/camelyonpatch_level_2_split_valid_meta.csv'
TEST_METADATA_PATH = ROOT_PATH / 'Data/raw-metadata/camelyonpatch_level_2_split_test_meta.csv'

TRAIN_IMAGE_X_PATH = ROOT_PATH / 'Data/raw-data/camelyonpatch_level_2_split_train_x.h5'
TRAIN_IMAGE_Y_PATH = ROOT_PATH / 'Data/raw-data/camelyonpatch_level_2_split_train_y.h5'

VALID_IMAGE_X_PATH = ROOT_PATH / 'Data/raw-data/camelyonpatch_level_2_split_valid_x.h5'
VALID_IMAGE_Y_PATH = ROOT_PATH / 'Data/raw-data/camelyonpatch_level_2_split_valid_y.h5'

TEST_IMAGE_X_PATH = ROOT_PATH / 'Data/raw-data/camelyonpatch_level_2_split_test_x.h5'
TEST_IMAGE_Y_PATH = ROOT_PATH / 'Data/raw-data/camelyonpatch_level_2_split_test_y.h5'

LABELS_PATH = ROOT_PATH / 'Data/raw-metadata/train_labels.csv'
PATCH_ID_WSI = ROOT_PATH / 'Data/raw-metadata/patch_id_wsi_full.csv'

# %% load data

train_metadata = pd.read_csv(TRAIN_METADATA_PATH, index_col=0)
valid_metadata = pd.read_csv(VALID_METADATA_PATH, index_col=0)
test_metadata = pd.read_csv(TEST_METADATA_PATH, index_col=0)
labels = pd.read_csv(LABELS_PATH)
wsi_to_id = pd.read_csv(PATCH_ID_WSI)

train_metadata['wsi'] = train_metadata['wsi'].astype('str')
wsi_to_id['wsi'] = wsi_to_id['wsi'].astype('str')

pd.merge(train_metadata, wsi_to_id[['id', 'wsi']], on='wsi', how='left')

# %% Get Images

X_train = h5py.File(TRAIN_IMAGE_X_PATH, 'r')

for group in X_train:

    array = X_train[group][0]

    image = PIL_Image.fromarray(array)

image

# %%
