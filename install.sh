#!/bin/bash
# install.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Starting Quantum Molecular System Installation...${NC}"

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python $required_version or higher is required${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv quantum_env

# Activate virtual environment
source quantum_env/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install required system packages
echo -e "${BLUE}Installing system requirements...${NC}"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        libopenblas-dev \
        liblapack-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install openblas lapack
fi

# Install Python packages
echo -e "${BLUE}Installing Python requirements...${NC}"
pip install -e .

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}CUDA detected, installing GPU requirements...${NC}"
    pip install -e .[gpu]
fi

# Install visualization requirements
echo -e "${BLUE}Installing visualization requirements...${NC}"
pip install -e .[viz]

# Run tests
echo -e "${BLUE}Running tests...${NC}"
python -m pytest tests/

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Installation completed successfully!${NC}"
    echo -e "To activate the environment, run: ${BLUE}source quantum_env/bin/activate${NC}"
else
    echo -e "${RED}Installation completed with test failures${NC}"
fi
