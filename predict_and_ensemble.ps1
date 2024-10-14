# Load configuration from Json
$config = Get-Content "ensemble_config.json" | ConvertFrom-Json
$input_dir = Resolve-Path $config.input_dir
$predictions_dir = Resolve-Path $config.predictions_dir
$output_dir = Resolve-Path $config.output_dir
$weights = $config.weight_list -join ' '
$idx = 1

$env:NO_ALBUMENTATIONS_UPDATE = "1"

if ($config.do_ensemble) {

    # Prediction loop
    foreach ($model in $config.ensemble) {
        Write-Output "`nPredicting with model: $($model.model_name)`n"
        $submission_name = $model.model_name + "_$($idx)" + ".csv"

        # Cross validation ensemble predictions
        if ($model.cv_ensemble){
            python src/Models/Predict_CV_ensemble.py `
            --model-type $model.model_class `
            --model-name $model.model_name `
            --tta $config.tta `
            --input $input_dir `
            --output $predictions_dir `
            --submission-name $submission_name
        }

        # Model predictions
        else {
            python src/Models/Predictions.py `
            --model-type $model.model_class `
            --model-name $model.model_name `
            --model-file $model.file `
            --swa $model.swa_model `
            --tta $config.tta `
            --input $input_dir `
            --output $predictions_dir `
            --submission-name $submission_name
        }
        $idx ++
    }

    # Ensemble model
    Write-Output "`nEnsembling models"

    # Weighted average ensemble
    if ($config.weighted_average) {
        python src/Models/Ensemble_predictions.py `
        --weighted-avg 1 `
        --weights $weights `
        --input $predictions_dir `
        --output $output_dir `
        --submission-name $config.submission_name
    }
    
    # Simple average ensemble
    else {
        python src/Models/Ensemble_predictions.py `
        --input $predictions_dir `
        --output $output_dir `
        --submission-name $config.submission_name
    }

    Write-Output "`nEnsemble Submission saved in $output_dir"

    if ( -not ($config.keep_submissions)) {

        Remove-Item "$predictions_dir\*"
        Write-Output "`nRemoved single model predictions`n"
            
    }
}