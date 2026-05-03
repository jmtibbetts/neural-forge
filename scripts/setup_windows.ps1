# NeuralForge Windows 11 Setup Script
# Optimized for: RTX 5090 | i9-284K | 128GB DDR5 | CUDA 12.x

Write-Host "⚡ NeuralForge Setup" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor DarkGray

# Check Python
$pythonVersion = python --version 2>&1
Write-Host "Python: $pythonVersion" -ForegroundColor Green

# Create venv
Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
Write-Host "`n[2/5] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 (RTX 5090 / Ada Lovelace / Blackwell)
Write-Host "`n[3/5] Installing PyTorch (CUDA 12.1)..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric
Write-Host "`n[4/5] Installing PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric
# Windows binaries for scatter/sparse
pip install pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Install remaining requirements
Write-Host "`n[5/5] Installing NeuralForge dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n✅ Setup complete!" -ForegroundColor Green
Write-Host "Launch with: python -m neural_forge.ui.app" -ForegroundColor Cyan
Write-Host "Then open:   http://localhost:8080" -ForegroundColor Cyan
