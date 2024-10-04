# Configuration file `config.yml`

This document describes the different configuration options that can be set in `config.yml`

## Data

Configuration related to data management

### Parameters

- **img_size**: size of the image to be rescaled
- **train_batch_size** : size of train batch used by train dataloader
- **valid_batch_size** : size of validation batch used by validation dataloader
- **test_batch_size** : size of test batch used by test dataloader
- **num_workers** : number of cpu cores used by dataloaders
- **pin_memory**: 

### Sampling

- **n_fold** : number of folds obtained by stratified group k-fold splitter. Only one of theese splits will be used for validation in a regular training. For cv ensembling a different split will be used for each model.
- **valid_downsample** : boolean value indicating if downsampling will be performed on the validation set.
- **downsample_quantity** : number of balanced samples in validation set if downsampling is performed.
- **Random_sampling** : boolean value indicating if random-sampling will be performed on the training set.
- **Rnd_sampling_q** : number of balanced samples in training set if random-sampling is performed.

## Model

Configuration related to model training and prediction 

### Architecture

- **model_class** : class used to instantiate the model (classes defined in [model_definition.py](../src/Models/model_definition.py)) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **model_name** : timm model name to load &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **load_model** : boolean value indicating whether or not the model weights are loaded from a file  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **model_to_load** : file name where model weights are saved. This file must be located in Models/<model_name>/  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **kwargs** : dictionary with additional arguments required for the instantiation of the model class &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`

### Parameters

- **freeze_params** : boolean value indicating whether or not the backbone weights are frozen &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **epochs** : number of epochs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **es_count** : number of epochs to stop training if there is no improvement in metric &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **es_reset** : number of epoch to restore the model best weights if there is no improvement in metric &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **min_eta** : minimum increase in the metric to be considered as an improvement &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **min_lr** : minimum learning rate for Cosine Annealing schedule &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **learning_rate** : learning rate. If One Cycle Scheduler is selected, then this correspond to the maximum learning rate &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **weight_decay** : weight decay parameter for ADAM optimizer &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **scheduler** : Learning rate scheduler. Options are: OneCycle, CosineAnnealing, or None &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **retrain_from_auroc** : if a value between 0 and 1 is indicated, the re-train mode will be enabled. In this mode, auroc validation metric must exceed the indicated value in order to save new weights &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **retrain_cv** : boolean value indicating if models will be re-trained. If true, each model will be re-trained from its best weights. Files must be located in Models/Cross Val Ensemble/<model_name>/<foldi> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for cv-ensemble training`
- **save_checkpoints** : boolean value indicating if model weights will be saved &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`
- **save_history** : boolean value indicating if train history will be saved in json format. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for training`

### Predictions

- **model_class** : class used to instantiate the model (classes defined in [model_definition.py](../src/Models/model_definition.py)) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for prediction`
- **model_name** : timm model name to load &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for prediction`
- **file** : name of the file containing the model weights. File must be located in Models/<model_name>/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for prediction`
- **cv_folds** : number of models of the same architecture trained with different folds. Files  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for cv-ensemble training`
- **tta** : boolean value indicating if TTA will be applied. TTA transformations are defined in [transformations.py](../src/Models/transformations.py) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for prediction`
- **submission_name** : submission name. Name must include the file extension (for example .csv) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; `for prediction`

## Misc

Additional configuration parameters

- **seed**: seed for reproducibility of experiments
- **device**: name of device. Options are: `cuda` for nvidia gpu accelerator or `cpu`