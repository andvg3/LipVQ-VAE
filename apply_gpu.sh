#!/bin/bash
#SBATCH --job-name=gpu4 # Job name
#SBATCH --output=log/gpu4.txt # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=64G # Total RAM to be used
#SBATCH --cpus-per-task=32 # Number of CPU cores
#SBATCH --gres=gpu:2 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=72:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs
 
tail -f /dev/null