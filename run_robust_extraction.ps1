Write-Host "=================================="
Write-Host "ROBUST EXTRACTION MONITOR STARTED"
Write-Host "=================================="

function Run-Extraction {
    param ([string]$dataset)
    $success = $false
    while (-not $success) {
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Starting/Resuming extract_features.py for dataset: $dataset"
        
        # Execute the python script
        $cmd = ".\venv_cuda\Scripts\python.exe extract_features.py --dataset $dataset"
        Invoke-Expression $cmd
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] >> SUCCESS: $dataset extraction completed normally."
            $success = $true
        } else {
            Write-Host "[WARNING] $dataset extraction crashed or was killed (Exit Code: $LASTEXITCODE)."
            Write-Host "Waiting 5 seconds before automatically resuming to recover from potential OOM..."
            Start-Sleep -Seconds 5
        }
    }
}

# Run DCSASS first until completely finished
Run-Extraction -dataset "dcass"

# Once DCSASS is completely finished safely, run UCF
Run-Extraction -dataset "ucf"

Write-Host "=================================="
Write-Host "ALL FEATURE EXTRACTIONS COMPLETED!"
Write-Host "=================================="
