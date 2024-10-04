# Cargar la configuración YAML en PowerShell (si necesitas leerla desde PowerShell)
$config = Get-Content "ensemble_config.json" | ConvertFrom-Json
$input_dir = Resolve-Path $config.input_dir
$predictions_dir = Resolve-Path $config.predictions_dir
$output_dir = Resolve-Path $config.output_dir
$idx = 1

$env:NO_ALBUMENTATIONS_UPDATE = "1"

if ($config.do_ensemble) {
    foreach ($model in $config.ensemble) {
        Write-Output "`nPredicting with model: $($model.model_name)`n"
        $submission_name = $model.model_name + "_$($idx)" + ".csv"

        if ($model.cv_ensemble){
            python src/Models/Predict_CV_ensemble.py `
            --model-type $model.model_class `
            --model-name $model.model_name `
            --tta $config.tta `
            --input $input_dir `
            --output $predictions_dir `
            --submission-name $submission_name
        }

        else {
            python src/Models/Predictions.py `
            --model-type $model.model_class `
            --model-name $model.model_name `
            --model-file $model.file `
            --tta $config.tta `
            --input $input_dir `
            --output $predictions_dir `
            --submission-name $submission_name
        }
        $idx ++
    }

    Write-Output "Ensembling models`n"

    python src/Models/Ensemble_predictions.py `
        --input $predictions_dir `
        --output $output_dir `
        --submission-name $config.submission_name

    if ( -not ($config.keep_submissions)) {

        Remove-Item "$predictions_dir\*"
        Write-Output "Removed single model predictions"
            
    }
}