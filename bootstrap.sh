#!/usr/bin/env bash
# Bootstrap script to install and start the LLM server

set -e

cd /var/home/srgtuszy/gpu-setup

echo "Installing llama-hermes.service..."
sudo cp llama-hermes.service /etc/systemd/system/

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo "Enabling and starting llama-hermes..."
sudo systemctl enable --now llama-hermes

echo "Checking status..."
sudo systemctl status llama-hermes

echo ""
echo "Server should be available at http://localhost:8002"