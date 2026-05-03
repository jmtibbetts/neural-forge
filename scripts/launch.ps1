# NeuralForge - One-Click Install + Launch
# Double-click this file OR run from CMD:
#   powershell -ExecutionPolicy Bypass -File launch.ps1

$Host.UI.RawUI.WindowTitle = "NeuralForge Setup"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  NeuralForge - Setup and Launch" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Move to project root (one level up from scripts\)
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host "  Project root: $projectRoot" -ForegroundColor DarkGray
Write-Host ""

# Find Python 3.12
Write-Host "  [CHECK] Looking for Python 3.12..." -ForegroundColor Yellow
$PY = $null
try { $ver = & py -3.12 --version 2>$null; if ($ver -match "3\.12") { $PY = "py -3.12" } } catch {}
if (-not $PY) {
    try { $ver = & python --version 2>$null; if ($ver -match "3\.12") { $PY = "python" } } catch {}
}
if (-not $PY) {
    Write-Host "  ERROR: Python 3.12 not found. Install from https://python.org" -ForegroundColor Red
    pause; exit 1
}
Write-Host "  Found: $ver ($PY)" -ForegroundColor Green
Write-Host ""

# Create venv if missing
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  [SETUP] Creating virtual environment..." -ForegroundColor Yellow
    Invoke-Expression "$PY -m venv .venv"
    Write-Host "  venv created." -ForegroundColor Green
} else {
    Write-Host "  [SKIP] venv already exists." -ForegroundColor DarkGray
}

$venvPy  = "$projectRoot\.venv\Scripts\python.exe"
$venvPip = "$projectRoot\.venv\Scripts\pip.exe"

function Is-Installed($pkg) {
    $result = & $venvPip show $pkg 2>$null
    return ($null -ne $result -and $result -ne "")
}

# Step 1: pip upgrade
Write-Host ""
Write-Host "  [1/7] Upgrading pip..." -ForegroundColor Yellow
& $venvPip install --upgrade pip setuptools wheel --quiet
Write-Host "  Done." -ForegroundColor Green

# Step 2: PyTorch nightly for RTX 5090 Blackwell (sm_100)
Write-Host ""
$torchVer    = & $venvPy -c "import torch; print(torch.__version__)" 2>$null
$blackwellOk = $false
if ($torchVer) {
    $smCheck = & $venvPy -c "import torch; caps=[torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]; print(any(c[0]>=10 for c in caps))" 2>$null
    if ($smCheck -eq "True") { $blackwellOk = $true }
}

if ($blackwellOk) {
    Write-Host "  [SKIP] PyTorch nightly already installed with Blackwell support. ($torchVer)" -ForegroundColor DarkGray
} else {
    if ($torchVer) {
        Write-Host "  [UPGRADE] Existing PyTorch has no Blackwell kernel. Reinstalling nightly..." -ForegroundColor Yellow
        & $venvPip uninstall torch torchvision torchaudio -y --quiet 2>$null
    } else {
        Write-Host "  [2/7] Installing PyTorch nightly + CUDA 12.8 for RTX 5090 Blackwell..." -ForegroundColor Yellow
    }
    Write-Host "  NOTE: ~2-3 GB download. Please wait..." -ForegroundColor DarkGray
    & $venvPip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    Write-Host "  PyTorch nightly installed." -ForegroundColor Green
}

# Step 3: Verify
Write-Host ""
Write-Host "  [CHECK] Verifying PyTorch + CUDA..." -ForegroundColor Yellow
$torchVer = & $venvPy -c "import torch; print(torch.__version__)"        2>$null
$cudaOk   = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
$gpuName  = & $venvPy -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')" 2>$null
$smVer    = & $venvPy -c "import torch; c=torch.cuda.get_device_capability(0); print('sm_'+str(c[0])+str(c[1]))" 2>$null
Write-Host "  torch   : $torchVer" -ForegroundColor Green
Write-Host "  CUDA ok : $cudaOk"   -ForegroundColor Green
Write-Host "  GPU     : $gpuName"  -ForegroundColor Green
Write-Host "  SM arch : $smVer"    -ForegroundColor Green

# Step 4: torch-geometric
Write-Host ""
if (Is-Installed "torch-geometric") {
    Write-Host "  [SKIP] torch-geometric already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [3/7] Installing torch-geometric..." -ForegroundColor Yellow
    & $venvPip install torch-geometric==2.5.3
    Write-Host "  torch-geometric installed." -ForegroundColor Green
}

# Step 5: PyG binary extensions (best-effort for nightly)
Write-Host ""
if (Is-Installed "torch-scatter") {
    Write-Host "  [SKIP] PyG extensions already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [4/7] Attempting PyG binary extensions..." -ForegroundColor Yellow
    & $venvPip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html 2>$null
    if (Is-Installed "torch-scatter") {
        Write-Host "  PyG extensions installed." -ForegroundColor Green
    } else {
        Write-Host "  [WARN] No prebuilt PyG wheels for nightly yet - graph models disabled. Financial Brain unaffected." -ForegroundColor Yellow
    }
}

# Step 6: ONNX
Write-Host ""
if (Is-Installed "onnx") {
    Write-Host "  [SKIP] ONNX already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [5/7] Installing ONNX..." -ForegroundColor Yellow
    & $venvPip install onnx onnxruntime-gpu onnxscript
    Write-Host "  ONNX installed." -ForegroundColor Green
}

# Step 7: App dependencies
Write-Host ""
if (Is-Installed "flask") {
    Write-Host "  [SKIP] App dependencies already installed." -ForegroundColor DarkGray
} else {
    Write-Host "  [6/7] Installing app dependencies..." -ForegroundColor Yellow
    & $venvPip install yfinance anthropic flask flask-socketio flask-cors eventlet plotly matplotlib seaborn nvidia-ml-py psutil GPUtil numpy pandas scikit-learn tqdm rich click pyyaml python-dotenv requests aiohttp jupyter ipywidgets ipykernel pytest black ruff
    Write-Host "  Done." -ForegroundColor Green
}

# Final check
Write-Host ""
Write-Host "  [7/7] Final package check..." -ForegroundColor Yellow
Write-Host ""
$checks = @(
    @("torch",           "import torch; print(torch.__version__)"),
    @("SM arch",         "import torch; c=torch.cuda.get_device_capability(0); print('sm_'+str(c[0])+str(c[1]))"),
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
        Write-Host ("  [OK] {0,-22} {1}" -f $c[0], $res) -ForegroundColor Green
    } else {
        Write-Host ("  [--] {0,-22} not available" -f $c[0]) -ForegroundColor DarkGray
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  All done! Launching NeuralForge..." -ForegroundColor Green
Write-Host "  Open: http://localhost:8080" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Start-Process "http://localhost:8080"
& $venvPy -m neural_forge.ui.app

pause
