#%%
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#%% parameters

submission_name = 'Ensemble_4xtta_ConvNextS_ConvNextv2T_EffNetb0_SeResNet50.csv'

# %% Paths

ROOT_PATH = Path(__file__).parents[2]

SUBMISSIONS_PATH = ROOT_PATH / 'Submissions/TTA'
TEST_IMAGES_PATH = ROOT_PATH / 'Data/test-images'
ENSEMBLE_SUBMISSION_PATH = ROOT_PATH / 'Submissions/Ensemble'

# %% submissions to ensemble

df_test = [file.name for file in TEST_IMAGES_PATH.iterdir() if file.is_file()]
df_test = [file.split(sep='.')[0] for file in df_test]
df_test = pd.DataFrame({'id':df_test})

submission_list = ['tta_convnext_small_01.csv',
                   'tta_convnextv2_tiny_01.csv',
                   'tta_efficientnet_b0_01.csv',
                   'tta_seresnet50_ra2_submission_03.csv']

submission_acr = ['convnext_small','conconvnext_tiny','EfficientNet_b0','SeresNet50']

submission_list = [SUBMISSIONS_PATH / file for file in submission_list]

for submission, acr in zip(submission_list, submission_acr):
    df = pd.read_csv(submission, header=0, names=['id',acr+'_label'])
    df_test = df_test.merge(df, on='id',how='inner')

# %% Correlation

fig, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df_test.drop('id', axis=1).corr(), cmap='plasma', annot=True, ax=ax)

# %% Ensemble

df_test['label'] = df_test.drop('id', axis=1).mean(axis=1)
df_test = df_test[['id','label']]

# %% submission

df_test.to_csv(ENSEMBLE_SUBMISSION_PATH / submission_name, index=False)

# %%
