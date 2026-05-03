# NeuralForge - One-Click Install + Launch
# Double-click this file OR run from CMD:
#   powershell -ExecutionPolicy Bypass -File launch.ps1

$Host.UI.RawUI.WindowTitle = "NeuralForge Setup"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuralForge - Setup and Launch" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# ── Move to project root (one level up from scripts\) ─────
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host "  Project root: $projectRoot" -ForegroundColor DarkGray
Write-Host ""

# ── Find Python 3.12 ──────────────────────────────────────
Write-Host "  [CHECK] Looking for Python 3.12..." -ForegroundColor Yellow
$PY = $null
try {
    $ver = & py -3.12 --version 2>$null
    if ($ver -match "3\.12") { $PY = "py -3.12" }
} catch {}

if (-not $PY) {
    try {
        $ver = & python --version 2>$null
        if ($ver -match "3\.12") { $PY = "python" }
    } catch {}
}

if (-not $PY) {
    Write-Host "  ERROR: Python 3.12 not found." -ForegroundColor Red
    Write-Host "  Install from https://python.org and try again." -ForegroundColor Red
    pause
    exit 1
}
Write-Host "  Found: $ver ($PY)" -ForegroundColor Green
Write-Host ""

# ── Create venv if missing ────────────────────────────────
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  [SETUP] Creating virtual environment..." -ForegroundColor Yellow
    Invoke-Expression "$PY -m venv .venv"
    Write-Host "  venv created." -ForegroundColor Green
} else {
    Write-Host "  [SKIP] venv already exists." -ForegroundColor DarkGray
}

$venvPy  = "$projectRoot\.venv\Scripts\python.exe"
$venvPip = "$projectRoot\.venv\Scripts\pip.exe"

# ── Helper: check if a package is installed ───────────────
function Is-Installed($pkg) {
    $result = & $venvPip show $pkg 2>$null
    return ($result -ne $null -and $result -ne "")
}

# ── Step 1: pip upgrade ───────────────────────────────────
Write-Host ""
Write-Host "  [1/7] Upgrading pip..." -ForegroundColor Yellow
& $venvPip install --upgrade pip setuptools wheel --quiet
Write-Host "  Done." -ForegroundColor Green

# ── Step 2: PyTorch ──────────────────────────────────────
Write-Host ""
if (Is-Installed "torch") {
    $tv = & $venvPip show torch 2>$null | Select-String "Version"
    Write-Host "  [SKIP] PyTorch already installed. ($tv)" -ForegroundColor DarkGray
} else {
    Write-Host "  [2/7] Installing PyTorch 2.3.1 + CUDA 12.1 (large download)..." -ForegroundColor Yellow
    & $venvPip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    Write-Host "  PyTorch installed." -ForegroundColor Green
}

# ── Step 3: Verify torch + CUDA ──────────────────────────
Write-Host ""
Write-Host "  [CHECK] Verifying PyTorch + CUDA..." -ForegroundColor Yellow
$torchVer  = & $venvPy -c "import torch; print(torch.__version__)"        2>$null
$cudaOk    = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
$gpuName   = & $venvPy -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" 2>$null
Write-Host "  torch   : $torchVer" -ForegroundColor Green
Write-Host "  CUDA ok : $cudaOk"   -ForegroundColor Green
Write-Host "  GPU     : $gpuName"  -ForegroundColor Green

# ── Step 4: torch-geometric ──────────────────────────────
Write-Host ""
if (Is-Installed "torch-geometric") {
    Write-Host "  [SKIP] torch-geometric already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [3/7] Installing torch-geometric..." -ForegroundColor Yellow
    & $venvPip install torch-geometric==2.5.3
    Write-Host "  torch-geometric installed." -ForegroundColor Green
}

# ── Step 5: PyG binary extensions ────────────────────────
Write-Host ""
if (Is-Installed "torch-scatter") {
    Write-Host "  [SKIP] PyG extensions already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [4/7] Installing PyG binary extensions..." -ForegroundColor Yellow
    & $venvPip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
    Write-Host "  PyG extensions installed." -ForegroundColor Green
}

# ── Step 6: ONNX ─────────────────────────────────────────
Write-Host ""
if (Is-Installed "onnx") {
    Write-Host "  [SKIP] ONNX already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [5/7] Installing ONNX..." -ForegroundColor Yellow
    & $venvPip install onnx onnxruntime-gpu onnxscript
    Write-Host "  ONNX installed." -ForegroundColor Green
}

# ── Step 7: All other deps ────────────────────────────────
Write-Host ""
if (Is-Installed "flask") {
    Write-Host "  [SKIP] App dependencies already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [6/7] Installing app dependencies..." -ForegroundColor Yellow
    & $venvPip install yfinance anthropic flask flask-socketio flask-cors eventlet plotly matplotlib seaborn pynvml psutil GPUtil numpy pandas scikit-learn tqdm rich click pyyaml python-dotenv requests aiohttp jupyter ipywidgets ipykernel pytest black ruff
    Write-Host "  Dependencies installed." -ForegroundColor Green
}

# ── Final check table ─────────────────────────────────────
Write-Host ""
Write-Host "  [7/7] Final package check..." -ForegroundColor Yellow
Write-Host ""
$checks = @(
    @("torch",           "import torch; print(torch.__version__)"),
    @("torch-geometric", "import torch_geometric; print(torch_geometric.__version__)"),
    @("torch-scatter",   "import torch_scatter; print('ok')"),
    @("onnxruntime",     "import onnxruntime; print(onnxruntime.__version__)"),
    @("flask",           "import flask; print(flask.__version__)"),
    @("yfinance",        "import yfinance; print(yfinance.__version__)"),
    @("sklearn",         "import sklearn; print(sklearn.__version__)")
)
foreach ($c in $checks) {
    $res = & $venvPy -c $c[1] 2>$null
    if ($res) {
        Write-Host ("  [OK] {0,-20} {1}" -f $c[0], $res) -ForegroundColor Green
    } else {
        Write-Host ("  [!!] {0,-20} NOT FOUND" -f $c[0]) -ForegroundColor Red
    }
}

# ── Launch ────────────────────────────────────────────────
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup complete! Launching studio..." -ForegroundColor Green
Write-Host "  Open your browser: http://localhost:8080" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Start-Process "http://localhost:8080"
& $venvPy -m neural_forge.ui.app

pause
