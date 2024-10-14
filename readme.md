# Histopathologic Cancer Detection LB private

**Private LB Score: 0.9785**,
**Public LB Score: 0.9763**

This project implements a machine learning solution to predict histopathologic cancer from digital pathology scan images. The dataset was released as part of a Kaggle competition in 2019 and is available [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview).

## Model Architecture

The final solution is based on an ensemble of five different model architectures: `convnet_small`, `efficientnet_b0`, `se-resnet50`, `swin_s3_tiny` and `vit_small`. The ensemble was created using a simple mean of predictions from all models. Among these, the top-performing single model was `vit_small`, achieving a **private LB AUROC of 0.9761** and a **public LB AUROC of 0.9729**. Before training, the header layers of the five models were replaced by a Generalized Mean Pooling followed by a linear layer with sigmoid activation.

## Dataset

A stratified group k-fold cross-validation strategy was employed to split the dataset into training and validation, using the wsi code to get non overlaping wsi groups. WSI assignment is available [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection/discussion/85283).

Due to the large size of the dataset, training different model architectures was time-consuming. To mitigate this, random subsampling of the training set and downsampling of the validation set were applied. Additionally, a basic CNN model was trained to evaluate the performance of different augmentation techniques through cross-validation, allowing for a faster iteration process.

These minor augmentation techniques were employed not to expand the dataset, as it was already sufficiently large, but to enhance the model's generalization. Additionally, two transformations were applied to the training, validation, and test data: resizing the images to 224x224 pixels and performing a normalization operation. The values used for normalization were based on the mean and standard deviation of the Imagenet dataset.

## Training

### Transfer Learning
Pre-trained models were trained using transfer learning to adapt them for the cancer detection task. Initially, the backbone weights of the models were frozen, and only the final classification layers were trained. For this phase, an aggressive learning rate and a one-cycle learning rate scheduler were employed.

### Fine-Tuning
To further improve performance, the entire model (including the pre-trained backbone) was later re-trained using a lower learning rate, combined with a Cosine Annealing scheduler. To enchance the optimization process, the best model weights were loaded when no improvements were observed over several epochs.

### Swa Training
After fine-tuning the various models, they were re-trained using a stochastic weighted average (SWA) approach over 20 epochs. This method enhanced the models' generalization capabilities, leading to improved AUROC scores.

## Predictions

To enhance prediction robustness, a Test-Time Augmentation (TTA) strategy with 4 variations was applied. The augmentations were based on geometric transformations similar to those used during data augmentation. This technique improved both private and public leaderboard scores.

Once TTA predictions were generated for each of the five models, they were ensembled to produce the final predictions.

## What did not work:

Combining predictions from the models using a weighted averaging approach:
- Since the private leaderboard (LB) score is now publicly available on Kaggle, the ensemble weights were adjusted to place greater emphasis on the predictions from the models that performed best, improving the private LB score (0.9788). However, these results are not taken into account, as this lead to overfitting to the private test data. Moreover, this practice is deemed inappropriate, as it exploits information that should not be accessible in a competition.
- Weights were also adjusted according to public LB score from model predictions, which resulted in overfitting to the public test data and ultimately produced a lower private LB score.
- Weights were also adjusted based on the local AUROC score reported by the trained models on the validation set. This led to overfitting on the local validation data and resulted in lower performance on both the public and private leaderboard.

Other model architectures, including ResNet, DenseNet, VGG-16, Inception-v3, and EVA02, were also tested but produced inferior results. Additionally, attempts to improve performance by adding more fully connected layers with batch normalization and incorporating dropout layers did not led to any improvement.

A cross-validation ensemble approach was tested on SeResNet50 to increase robustness. The dataset was divided into six distinct folds, and five models were trained using fold i for training and fold i+1 for validation (for i in n-folds). Due to the extended times required for each model to train, only a few samples of each folds were used (by random-samplig train data and downsampling validation data). This experiment yielded an AUROC value around 0.9 for each model, resulting in a poor ensemble predictor. Increasing the sample size of each folds may produce better results, but was discarded due to the high training time required.

For prediction task, an 8-variation TTA was also tested but did not show any significant improvement over the 4-variation approach.

## Final Ensemble Model Pipeline

The prediction pipeline for the final ensemble model is implemented by the powershell automation script `predict_and_ensemble.ps1`. This script will make predictions with different models and then combine all the results in a single submission.

## Configuration Files

Two configurations file where used: `config.yml` for training and predictions scripts and `ensemble_config.json` for automation powershell script `predict_and_ensemble.ps1`. The options available in `ensemble_config.json` override the prediction and ensemble variables set in `config.yml`. For more information, see documentation:

- [config.yml documentation](./Docs/configuration.md)
- [ensemble_config.json documentation](./Docs/ensemble_configuration.md)
