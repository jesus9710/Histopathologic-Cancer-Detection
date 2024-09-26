# Cargar la configuración YAML en PowerShell (si necesitas leerla desde PowerShell)
$config = Get-Content "ensemble_config.json" | ConvertFrom-Json
$input_dir = Resolve-Path $config.input_dir
$predictions_dir = Resolve-Path $config.predictions_dir
$output_dir = Resolve-Path $config.output_dir
$idx = 1

if ($config.do_ensemble) {
    foreach ($model in $config.ensemble) {
        Write-Output "Predicting with model: $($model.model_name)`n:"
        $submission_name = $model.model_name + "_$($idx)" + ".csv"
        python src/Models/Predictions.py `
            --model-type $model.model_class `
            --model-name $model.model_name `
            --model-file $model.file `
            --input $input_dir `
            --output $predictions_dir `
            --submission-name $submission_name
        $idx ++
    }

    python src/Models/Ensemble_predictions.py `
        --input $predictions_dir `
        --output $output_dir

    if ( -not ($config.keep_submissions)) {

        Write-Output "Se eliminan los submissions individuales"
        Remove-Item $predictions_dir
            
    }
}