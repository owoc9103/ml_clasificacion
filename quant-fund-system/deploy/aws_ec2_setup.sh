#!/bin/bash
# AWS EC2 Setup Script
# Run this on a fresh Ubuntu 22.04 LTS instance

set -e

echo "=========================================="
echo "Setting up Quant Fund Trading System"
echo "=========================================="

# Update system
echo "1. Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# Install Docker
echo "2. Installing Docker..."
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
echo "3. Installing Docker Compose..."
sudo apt-get install -y docker-compose

# Install Python (for running outside Docker if needed)
echo "4. Installing Python 3.11..."
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3-pip

# Install Git
echo "5. Installing Git..."
sudo apt-get install -y git

# Create project directory
echo "6. Creating project directory..."
mkdir -p ~/quantfund
cd ~/quantfund

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Clone your repository:"
echo "   git clone <your-repo-url> ."
echo ""
echo "2. Create .env file with your API keys:"
echo "   nano .env"
echo ""
echo "3. Build and run the Docker container:"
echo "   docker-compose build"
echo "   docker-compose up -d"
echo ""
echo "4. Set up cron job:"
echo "   ./deploy/setup_cron.sh"
echo ""
echo "5. Test the trading script:"
echo "   docker-compose exec trading-bot python scripts/run_live.py --mode paper"
echo ""
