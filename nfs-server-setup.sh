#!/usr/bin/env bash
set -xeuo pipefail

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get -y upgrade

sudo apt install -y python3-pip python3-venv build-essential zip unzip net-tools aria2 nfs-common rsync

# --- DOCA-Host repo (Ubuntu 22.04/24.04) ---
wget https://www.mellanox.com/downloads/DOCA/DOCA_v3.0.0/host/doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb
sudo dpkg -i doca-host_3.0.0-058000-25.04-ubuntu2404_amd64.deb

# --- DOCA ---
sudo apt-get update
sudo apt-get -y install doca-networking