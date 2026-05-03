# ============================================================
# NeuralForge Windows 11 Setup Script
# Optimized for: RTX 5090 | i9-284K | 128GB DDR5 | CUDA 12.x
# Run from the neural-forge root directory
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  ⚡ NeuralForge Windows Setup" -ForegroundColor Cyan
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host ""

# ─── Detect Python ─────────────────────────────────────────
$pyVer = python --version 2>&1
Write-Host "  Python: $pyVer" -ForegroundColor Green

# ─── Upgrade pip / build tools ─────────────────────────────
Write-Host "`n  [1/6] Upgrading pip and build tools..." -ForegroundColor Yellow
python -m pip install --upgrade pip setuptools wheel

# ─── Step 2: Install PyTorch with CUDA FIRST ───────────────
# Must happen before anything that depends on torch (scatter, sparse, etc.)
Write-Host "`n  [2/6] Installing PyTorch 2.x + CUDA 12.1 (RTX 5090)..." -ForegroundColor Yellow
Write-Host "        (this is the big download — ~2.5 GB, grab a coffee)" -ForegroundColor DarkGray
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify torch installed correctly
$torchCheck = python -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())" 2>&1
Write-Host "  ✓ $torchCheck" -ForegroundColor Green

# ─── Step 3: PyTorch Geometric core ────────────────────────
Write-Host "`n  [3/6] Installing PyTorch Geometric..." -ForegroundColor Yellow
pip install torch-geometric

# ─── Step 4: PyG binary extensions (scatter, sparse, etc.) ─
# These MUST come after torch and torch-geometric
# Using prebuilt wheels from pyg.org for torch 2.3 + CUDA 12.1
Write-Host "`n  [4/6] Installing PyG binary extensions (scatter, sparse, cluster)..." -ForegroundColor Yellow
Write-Host "        (using prebuilt wheels — no compilation needed)" -ForegroundColor DarkGray
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv `
    -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# ─── Step 5: ONNX + Runtime ────────────────────────────────
Write-Host "`n  [5/6] Installing ONNX and ONNX Runtime GPU..." -ForegroundColor Yellow
pip install onnx onnxruntime-gpu onnxscript

# ─── Step 6: All remaining dependencies ────────────────────
Write-Host "`n  [6/6] Installing remaining NeuralForge dependencies..." -ForegroundColor Yellow
pip install `
    anthropic `
    flask flask-socketio flask-cors eventlet `
    plotly matplotlib seaborn networkx `
    pynvml psutil GPUtil `
    numpy pandas scikit-learn `
    tqdm rich click `
    pyyaml python-dotenv requests aiohttp `
    jupyter ipywidgets ipykernel

# ─── Done ──────────────────────────────────────────────────
Write-Host ""
Write-Host "  ============================================" -ForegroundColor DarkGray
Write-Host "  ✅  NeuralForge setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Launch the studio:" -ForegroundColor Cyan
Write-Host "    python -m neural_forge.ui.app" -ForegroundColor White
Write-Host ""
Write-Host "  Then open your browser:" -ForegroundColor Cyan
Write-Host "    http://localhost:8080" -ForegroundColor White
Write-Host ""
