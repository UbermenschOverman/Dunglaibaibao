#!/bin/bash
# create.sh
# Script to create project directories for DW_CNN_ECG_Denoising
# Run from the project root directory

echo "Creating project directories..."

# Raw data folders (user will populate manually)
mkdir -p data/raw/MITDB
mkdir -p data/raw/NSTDB

# Processed data folders
mkdir -p data/processed/noisy_0dB
mkdir -p data/processed/noisy_1.25dB
mkdir -p data/processed/noisy_5dB

# Experiments folders
mkdir -p experiments/ablation_bw
mkdir -p experiments/ablation_em
mkdir -p experiments/ablation_ma
mkdir -p experiments/comparison

# Checkpoints and results
mkdir -p checkpoints
mkdir -p results

echo "All directories created successfully."
