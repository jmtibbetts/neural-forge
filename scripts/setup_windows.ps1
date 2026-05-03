# ============================================================
# NeuralForge Windows 11 Setup Script
# Optimized for: RTX 5090 | i9-284K | 128GB DDR5 | CUDA 12.x
# Requires: Python 3.12
# Run from the neural-forge root directory
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  NeuralForge Windows Setup" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host ""

# Find Python 3.12 via the py launcher
$py312 = py -3.12 --version 2>&1
if ($py312 -match "3\.12") {
    Write-Host "  Using Python 3.12: $py312" -ForegroundColor Green
    $PY_CMD = "py -3.12"
} else {
    Write-Host "  WARNING: py -3.12 not found, trying python..." -ForegroundColor Yellow
    $PY_CMD = "python"
}

# Create venv
Write-Host ""
Write-Host "  [0/6] Creating virtual environment (.venv)..." -ForegroundColor Yellow
Invoke-Expression "$PY_CMD -m venv .venv"
.\.venv\Scripts\Activate.ps1
Write-Host "  OK - venv activated" -ForegroundColor Green

# Step 1
Write-Host ""
Write-Host "  [1/6] Upgrading pip and build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# Step 2
Write-Host ""
Write-Host "  [2/6] Installing PyTorch 2.3.1 + CUDA 12.1..." -ForegroundColor Yellow
Write-Host "        This is a large download, please wait." -ForegroundColor DarkGray
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

$torchCheck = python -c "import torch; cuda = torch.cuda.is_available(); gpu = torch.cuda.get_device_name(0) if cuda else 'N/A'; print('torch', torch.__version__, '| CUDA:', cuda, '| GPU:', gpu)" 2>&1
Write-Host "  OK - $torchCheck" -ForegroundColor Green

# Step 3
Write-Host ""
Write-Host "  [3/6] Installing PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric==2.5.3

# Step 4
Write-Host ""
Write-Host "  [4/6] Installing PyG binary extensions (prebuilt wheels)..." -ForegroundColor Yellow
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Step 5
Write-Host ""
Write-Host "  [5/6] Installing ONNX and ONNX Runtime GPU..." -ForegroundColor Yellow
pip install onnx onnxruntime-gpu onnxscript

# Step 6
Write-Host ""
Write-Host "  [6/6] Installing remaining dependencies..." -ForegroundColor Yellow
pip install anthropic flask flask-socketio flask-cors eventlet plotly matplotlib seaborn networkx pynvml psutil GPUtil numpy pandas scikit-learn tqdm rich click pyyaml python-dotenv requests aiohttp jupyter ipywidgets ipykernel pytest black ruff

# Done
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  To launch NeuralForge Studio:" -ForegroundColor Cyan
Write-Host "    python -m neural_forge.ui.app" -ForegroundColor White
Write-Host ""
Write-Host "  Then open: http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
