# ============================================================
# NeuralForge Windows 11 Setup Script
# Optimized for: RTX 5090 | i9-284K | 128GB DDR5 | CUDA 12.x
# Requires: Python 3.12
# Run from the neural-forge ROOT directory (not from scripts\)
# ============================================================

Write-Host ""
Write-Host "  NeuralForge Windows Setup" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host ""

# Find Python 3.12
$py312test = py -3.12 --version 2>&1
if ($py312test -match "3\.12") {
    $PY_CMD = "py -3.12"
    Write-Host "  Python: $py312test" -ForegroundColor Green
} else {
    $PY_CMD = "python"
    Write-Host "  py -3.12 not found, using default python" -ForegroundColor Yellow
}

# Create venv only if it does not already exist
Write-Host ""
Write-Host "  [0/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-Not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Invoke-Expression "$PY_CMD -m venv .venv"
    Write-Host "  venv created" -ForegroundColor Green
} else {
    Write-Host "  venv already exists, skipping creation" -ForegroundColor DarkGray
}
.\.venv\Scripts\Activate.ps1
Write-Host "  venv activated" -ForegroundColor Green

# Step 1
Write-Host ""
Write-Host "  [1/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel --quiet

# Step 2
Write-Host ""
Write-Host "  [2/6] Installing PyTorch 2.3.1 + CUDA 12.1 (large download)..." -ForegroundColor Yellow
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "  Verifying PyTorch..." -ForegroundColor DarkGray
$tv = python -c "import torch; print(torch.__version__)" 2>$null
$ca = python -c "import torch; print(torch.cuda.is_available())" 2>$null
$gn = python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>$null
Write-Host "  torch   : $tv" -ForegroundColor Green
Write-Host "  cuda    : $ca" -ForegroundColor Green
Write-Host "  gpu     : $gn" -ForegroundColor Green

# Step 3
Write-Host ""
Write-Host "  [3/6] Installing PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric==2.5.3

# Step 4
Write-Host ""
Write-Host "  [4/6] Installing PyG binary extensions..." -ForegroundColor Yellow
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# Step 5
Write-Host ""
Write-Host "  [5/6] Installing ONNX..." -ForegroundColor Yellow
pip install onnx onnxruntime-gpu onnxscript

# Step 6
Write-Host ""
Write-Host "  [6/6] Installing remaining dependencies..." -ForegroundColor Yellow
pip install anthropic flask flask-socketio flask-cors eventlet plotly matplotlib seaborn networkx pynvml psutil GPUtil numpy pandas scikit-learn tqdm rich click pyyaml python-dotenv requests aiohttp jupyter ipywidgets ipykernel pytest black ruff

# Final check
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host "  Final checks..." -ForegroundColor Yellow
Write-Host ""

$pkgs = @(
    @("torch",           "import torch; print(torch.__version__)"),
    @("torch-geometric", "import torch_geometric; print(torch_geometric.__version__)"),
    @("torch-scatter",   "import torch_scatter; print('ok')"),
    @("onnx",            "import onnx; print(onnx.__version__)"),
    @("onnxruntime",     "import onnxruntime; print(onnxruntime.__version__)"),
    @("flask",           "import flask; print(flask.__version__)"),
    @("anthropic",       "import anthropic; print(anthropic.__version__)")
)

foreach ($pkg in $pkgs) {
    $res = python -c $pkg[1] 2>$null
    if ($res) {
        Write-Host ("  [OK] {0,-20} {1}" -f $pkg[0], $res) -ForegroundColor Green
    } else {
        Write-Host ("  [!!] {0,-20} not found" -f $pkg[0]) -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host "  Done! To launch NeuralForge:" -ForegroundColor Green
Write-Host ""
Write-Host "    cd C:\Neural-Forge\neural-forge" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "    python -m neural_forge.ui.app" -ForegroundColor White
Write-Host ""
Write-Host "  Then open: http://localhost:8080" -ForegroundColor Cyan
Write-Host ""
