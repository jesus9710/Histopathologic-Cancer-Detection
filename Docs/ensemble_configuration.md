# Configuration file `ensemble_config.json`

When running the powershell script `predict_and_ensemble.ps1`, the options available in `ensemble_config.json` override the prediction and ensemble variables set in `config.yml`. This automation script provides the execution of several predictions from different models and then uses them to make an ensemble prediction.

## Parameters

- **do_ensemble**: boolean value indicating whether or not the ensembling of all predictions will be performed
- **tta**: integer value indicating if TTA will be applied. A value of 0 indicates that TTA will not apply, while a higher value indicates that it will. TTA transformations are defined in [transformations.py](../src/Models/transformations.py)
- **keep_submissions**: boolean value indicating if individual predictions will be kept. **If False, individual predictions will be removed**
- **submission_name** : submission name. Name must include the file extension (for example .csv)
- **input_dir**: directory where the test images are located. In this project, the location is `./Data/test-images`, but it can be modified for testing purposes
- **predictions_dir**: directory where predictions will be stored. In this project, the location is `./Predictions`, but it can be modified for testing purposes
- **output_dir** : directory where ensemble submission will be saved. In this project, the location is `./Submissions/Ensemble`, but it can be modified for testing purposes.
- **ensemble**: list of dictionary type objects containing information about each model. The structure of each dictionary should be as follows:

```
{
    "model_class": class used to instantiate the model

    "model_name": timm model name to load

    "file": name of the file containing the model weights. File must be located in Models/<model_name>/

    "swa_model": integer value indicating whether or not the model was trained with SWA. A value of 0 indicates that SWA was not applied, while a higher value indicates that it was

    "cv_ensemble": boolean value indicating whether the prediction will be performed with models trained with different folds (ensemble of predictions)
}
```

Model classes defined in [model_definition.py](../src/Models/model_definition.py)

If `cv_ensemble = true`, models should be organized under the directory: `Models/Cross Val Ensemble/<model_name>/`. Within this directory, each folder must correspond to a trained model from a different fold. The model weight files should be placed inside their respective folders. The system will automatically select the weights corresponding to the best AUROC score.

- **weighted_average** :  boolean value indicating whether or not weights will be used to average the predictions
- **weight_list** : list with weights for average the predictions. The weightings will be assigned to the files located in the predictions folder in alphabetical order