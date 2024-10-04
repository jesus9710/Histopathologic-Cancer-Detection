# Configuration file `ensemble_config.json`

The options available in `ensemble_config.json` override the prediction variables set in `config.yml`, when running the powershell script `predict_and_ensemble.ps1`. This automation script provides the execution of several predictions from different models and then uses them to make an ensemble prediction.

## Parameters

- **do_ensemble**: boolean value indicating whether or not the ensembling of all predictions will be performed
- **tta**: integer value indicating if TTA will be applied. A value of 0 indicates that TTA will not apply, otherwise it will. TTA transformations are defined in [transformations.py](../src/Models/transformations.py)
- **keep_submissions**: boolean value indicating if individual predictions will be kept. **If False, individual predictions will be removed**
- **submission_name** : submission name. Name must include the file extension (for example .csv)
- **input_dir**: Directory where the test images are located. In this project, the location is “./Data/test-images”, but it can be modified for testing purposes
- **predictions_dir**: Directory where predictions will be stored. In this project, the location is "./Predictions", but it can be modified for testing purposes
- **output_dir** : Directory where ensemble submission will be saved. In this project, the location is "./Submissions/Ensemble", but it can be modified for testing purposes.
- **ensemble**: list of dictionary type objects containing information about each model. The structure of each dictionary should be as follows:

```
{
    "model_class": class used to instantiate the model

    "model_name": timm model name to load

    "file": name of the file containing the model weights. File must be located in Models/<model_name>/

    "cv_ensemble": Boolean value indicating whether the prediction will be performed with models trained with different folds (ensemble of predictions).
}
```

Model classes defined in [model_definition.py](../src/Models/model_definition.py)

If cv_ensemble = true, the models must be located in Models/Cross Val Ensemble/<model_name>/. In this directory there should be as many folders as trained models with different folds. The files with the weights of the models should be inside these folders. The best auroc weights will be automatically selected.