$ErrorActionPreference = "Continue"

Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like "*syllable_svtr_b160_e180*" -or
        $_.CommandLine -like "*train_crnn_ipa_digits.py*"
    } |
    ForEach-Object {
        Write-Host "stopping pid=$($_.ProcessId) $($_.Name)"
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }

nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
