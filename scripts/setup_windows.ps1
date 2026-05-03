# ============================================================
# NeuralForge Windows 11 Setup Script
# Optimized for: RTX 5090 | i9-284K | 128GB DDR5 | CUDA 12.x
# Requires: Python 3.12 (recommended) or 3.11
# Run from the neural-forge root directory
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  ⚡ NeuralForge Windows Setup" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host ""

# ─── Pin to Python 3.12 ────────────────────────────────────
# Find py 3.12 launcher
$py = $null
try { $py = (Get-Command py -ErrorAction Stop).Source } catch {}

if ($py) {
    $pyVer = py -3.12 --version 2>&1
    if ($pyVer -match "3\.12") {
        Write-Host "  Using: py -3.12 ($pyVer)" -ForegroundColor Green
        $PY_CMD  = "py -3.12"
        $PIP_CMD = "py -3.12 -m pip"
    } else {
        Write-Host "  py -3.12 not found, falling back to py -3.13" -ForegroundColor Yellow
        $PY_CMD  = "py -3.13"
        $PIP_CMD = "py -3.13 -m pip"
    }
} else {
    $PY_CMD  = "python"
    $PIP_CMD = "python -m pip"
}

Write-Host "  Python cmd : $PY_CMD" -ForegroundColor DarkGray
Write-Host ""

# ─── Create venv with Python 3.12 ──────────────────────────
Write-Host "  [0/6] Creating virtual environment (.venv) with Python 3.12..." -ForegroundColor Yellow
Invoke-Expression "$PY_CMD -m venv .venv"
.\.venv\Scripts\Activate.ps1

# From here all commands use the venv's python/pip
Write-Host "  ✓ venv activated" -ForegroundColor Green

# ─── Step 1: Upgrade pip / build tools ─────────────────────
Write-Host "`n  [1/6] Upgrading pip and build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# ─── Step 2: PyTorch + CUDA 12.1 ───────────────────────────
Write-Host "`n  [2/6] Installing PyTorch 2.3 + CUDA 12.1 (RTX 5090)..." -ForegroundColor Yellow
Write-Host "        (~2.5 GB download — grab a coffee)" -ForegroundColor DarkGray
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Verify
$torchCheck = python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1
Write-Host "  ✓ $torchCheck" -ForegroundColor Green

# ─── Step 3: PyTorch Geometric core ────────────────────────
Write-Host "`n  [3/6] Installing PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric==2.5.3

# ─── Step 4: PyG binary extensions ─────────────────────────
# Prebuilt wheels for Python 3.12 + torch 2.3.1 + CUDA 12.1
Write-Host "`n  [4/6] Installing PyG binary extensions (prebuilt wheels, no compilation)..." -ForegroundColor Yellow
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv `
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# ─── Step 5: ONNX ──────────────────────────────────────────
Write-Host "`n  [5/6] Installing ONNX + ONNX Runtime GPU..." -ForegroundColor Yellow
pip install onnx>=1.16.0 onnxruntime-gpu>=1.18.0 onnxscript>=0.1.0

# ─── Step 6: All remaining dependencies ────────────────────
Write-Host "`n  [6/6] Installing remaining dependencies..." -ForegroundColor Yellow
pip install `
    anthropic `
    flask flask-socketio flask-cors eventlet `
    plotly matplotlib seaborn networkx `
    pynvml psutil GPUtil `
    numpy pandas scikit-learn `
    tqdm rich click `
    pyyaml python-dotenv requests aiohttp `
    jupyter ipywidgets ipykernel `
    pytest black ruff

# ─── Done ──────────────────────────────────────────────────
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host "  ✅  NeuralForge setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate venv next time:" -ForegroundColor DarkGray
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "  Launch the studio:" -ForegroundColor Cyan
Write-Host "    python -m neural_forge.ui.app" -ForegroundColor White
Write-Host ""
Write-Host "  Open browser:" -ForegroundColor Cyan
Write-Host "    http://localhost:8080" -ForegroundColor White
Write-Host ""
