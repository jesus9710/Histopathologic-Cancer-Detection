# Histopathologic Cancer Detection LB private

**LB Private Score: 0.9747**
**LB Public Score: 0.9737**

This project implements a machine learning solution to predict histopathologic cancer from digital pathology scan images. The dataset was released as part of a Kaggle competition in 2019 and is available [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection/overview).

## Ensemble of Models

The final solution is based on an ensemble of five different model architectures: `convnet_small`, `convnet2_tiny`, `efficientnet_b0`, `se-resnet50`, and `vit_small`. The ensemble was created using a simple mean of predictions from all models. Among these, the top-performing single model was `convnet_small`, achieving a **private LB AUROC of 0.9686** and a **public LB AUROC of 0.9690**.

Other model architectures, such as ResNet, DenseNet, Swin Transformers, and EVA02, were also tested but yielded inferior results.

## Dataset

A stratified group k-fold cross-validation strategy was employed to split the dataset into training and validation, using the wsi code to get non overlaping wsi groups. WSI assignment is available [here](https://www.kaggle.com/competitions/histopathologic-cancer-detection/discussion/85283).

Due to the large size of the dataset, training different model architectures was time-consuming. To mitigate this, random subsampling of the training set and downsampling of the validation set were applied. Additionally, a basic CNN model was trained to evaluate the performance of different augmentation techniques through cross-validation, allowing for a faster iteration process.

## Training

### Transfer Learning
Pre-trained models were trained using transfer learning to adapt them for the cancer detection task. Initially, the backbone weights of the models were frozen, and only the final classification layers were trained. For this phase, an aggressive learning rate and a one-cycle learning rate scheduler were employed.

### Fine-tuning
To further improve performance, the entire model (including the pre-trained backbone) was later re-trained using a lower learning rate, combined with a Cosine Annealing scheduler. To enchance the optimization process, the best model weights were loaded when no improvements were observed over several epochs

A cross-validation ensemble approach was considered to increase robustness, but it was not pursued due to the extended training times required for each model.

## Predictions

To enhance prediction robustness, a Test-Time Augmentation (TTA) strategy with 4 variations was applied. The augmentations were based on geometric transformations similar to those used during data augmentation. This technique improved both private and public leaderboard scores. An 8-variation TTA was also tested but did not show any significant improvement over the 4-variation approach.

Once TTA predictions were generated for each of the five models, they were ensembled to produce the final predictions.
