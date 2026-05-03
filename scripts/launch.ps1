# NeuralForge Launcher
param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8080
)

Write-Host "⚡ Launching NeuralForge Studio..." -ForegroundColor Cyan

# Activate venv if exists
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

# Launch
python -m neural_forge.ui.app --host $Host --port $Port
