$ErrorActionPreference = "Continue"

$RunRoot = "I:\shaoxing_tts_remote\work\remote_runs"
Write-Host "RUN_ROOT=$RunRoot"

Get-ChildItem -LiteralPath $RunRoot -Directory | ForEach-Object {
    $Dir = $_.FullName
    $Name = $_.Name
    Write-Host ""
    Write-Host "== $Name =="

    $History = Join-Path $Dir "history.tsv"
    if (Test-Path $History) {
        $Rows = Import-Csv -LiteralPath $History -Delimiter "`t"
        if ($Rows.Count -gt 0) {
            $Last = $Rows[-1]
            Write-Host "last_epoch=$($Last.epoch)"
            if ($Rows[0].PSObject.Properties.Name -contains "val_cer") {
                $BestCer = $Rows | Sort-Object {[double]$_.val_cer} | Select-Object -First 1
                $BestExact = $Rows | Sort-Object {[double]$_.val_exact} -Descending | Select-Object -First 1
                Write-Host "best_val_cer_epoch=$($BestCer.epoch) val_cer=$($BestCer.val_cer) val_exact=$($BestCer.val_exact)"
                Write-Host "best_val_exact_epoch=$($BestExact.epoch) val_exact=$($BestExact.val_exact) val_cer=$($BestExact.val_cer)"
            }
            if ($Rows[0].PSObject.Properties.Name -contains "val_f1") {
                $BestF1 = $Rows | Sort-Object {[double]$_.val_f1} -Descending | Select-Object -First 1
                $BestAcc = $Rows | Sort-Object {[double]$_.val_accuracy} -Descending | Select-Object -First 1
                Write-Host "best_val_f1_epoch=$($BestF1.epoch) val_f1=$($BestF1.val_f1) val_acc=$($BestF1.val_accuracy)"
                Write-Host "best_val_acc_epoch=$($BestAcc.epoch) val_acc=$($BestAcc.val_accuracy) val_f1=$($BestAcc.val_f1)"
            }
        }
    }

    $Eval = Join-Path $Dir "checkpoint_eval.tsv"
    if (Test-Path $Eval) {
        Write-Host "checkpoint_eval:"
        Get-Content -LiteralPath $Eval
    }

    Get-ChildItem -LiteralPath $Dir -Filter "predictions_*.tsv" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "prediction_file=$($_.Name) bytes=$($_.Length)"
    }
}
