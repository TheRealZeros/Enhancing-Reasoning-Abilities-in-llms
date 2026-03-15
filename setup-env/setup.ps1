$ErrorActionPreference = "Stop"

$envName = "enhancing-reasoning-mi"
$envFile = "setup-env/environment.yml"
$verifyScript = "setup-env/setup-scripts/verify_env.py"

Write-Host "=== Thesis Environment Setup ===" -ForegroundColor Cyan

# Check conda exists
try {
    conda --version | Out-Null
    Write-Host "Conda found." -ForegroundColor Green
}
catch {
    Write-Host "Conda is not available in this terminal." -ForegroundColor Red
    Write-Host "Open Anaconda Prompt / Miniconda Prompt, or ensure conda is on PATH." -ForegroundColor Yellow
    exit 1
}

# Initialise conda for this PowerShell session
& conda shell.powershell hook | Out-String | Invoke-Expression

# Create or update environment
Write-Host "Creating/updating conda environment from $envFile..." -ForegroundColor Cyan
conda env create -f $envFile
if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment may already exist. Attempting update instead..." -ForegroundColor Yellow
    conda env update -f $envFile --prune
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create or update environment." -ForegroundColor Red
        exit 1
    }
}

# Activate environment
Write-Host "Activating environment: $envName" -ForegroundColor Cyan
conda activate $envName

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to activate environment." -ForegroundColor Red
    exit 1
}

# Install PyTorch CUDA build explicitly from the PyTorch index
Write-Host "Installing PyTorch CUDA build..." -ForegroundColor Cyan
python -m pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install PyTorch CUDA build." -ForegroundColor Red
    exit 1
}

# Re-pin numpy for TransformerLens compatibility
Write-Host "Re-pinning numpy for TransformerLens compatibility..." -ForegroundColor Cyan
python -m pip install --upgrade --force-reinstall "numpy<2"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to re-pin numpy." -ForegroundColor Red
    exit 1
}

# Quick torch sanity check before full verify
Write-Host "Checking PyTorch and CUDA..." -ForegroundColor Cyan
python -c "import torch, numpy; print('torch:', torch.__version__); print('cuda:', torch.cuda.is_available()); print('numpy:', numpy.__version__)"

if ($LASTEXITCODE -ne 0) {
    Write-Host "PyTorch verification failed." -ForegroundColor Red
    exit 1
}

# Run verification
Write-Host "Running environment verification..." -ForegroundColor Cyan
python $verifyScript

if ($LASTEXITCODE -ne 0) {
    Write-Host "Environment verification failed." -ForegroundColor Red
    exit 1
}

Write-Host "Environment setup complete and verified." -ForegroundColor Green