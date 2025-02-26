# install.ps1
# PowerShell installation script for Windows

Write-Host "Starting Quantum Molecular System Installation..." -ForegroundColor Blue

# Check Python version
$pythonVersion = python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
$requiredVersion = "3.8"

if ([version]$pythonVersion -lt [version]$requiredVersion) {
    Write-Host "Error: Python $requiredVersion or higher is required" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host "Creating virtual environment..." -ForegroundColor Blue
python -m venv quantum_env

# Activate virtual environment
.\quantum_env\Scripts\Activate

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Blue
python -m pip install --upgrade pip

# Install Python packages
Write-Host "Installing Python requirements..." -ForegroundColor Blue
pip install -e .

# Check for CUDA
try {
    nvidia-smi
    Write-Host "CUDA detected, installing GPU requirements..." -ForegroundColor Blue
    pip install -e .[gpu]
} catch {
    Write-Host "CUDA not detected, skipping GPU requirements" -ForegroundColor Yellow
}

# Install visualization requirements
Write-Host "Installing visualization requirements..." -ForegroundColor Blue
pip install -e .[viz]

# Run tests
Write-Host "Running tests..." -ForegroundColor Blue
python -m pytest tests/

if ($LASTEXITCODE -eq 0) {
    Write-Host "Installation completed successfully!" -ForegroundColor Green
    Write-Host "To activate the environment, run: .\quantum_env\Scripts\Activate" -ForegroundColor Blue
} else {
    Write-Host "Installation completed with test failures" -ForegroundColor Red
}
