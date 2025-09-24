#!/bin/sh -l

#SBATCH --nodes=1
#SBATCH --gpus=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=4G        # total requested cpu memory
#SBATCH --time=1:00:00
#SBATCH --output=./inference_aifs_single-v1_4-3g40.txt

source ../.venv/bin/activate
uv run inference_aifs_single-v1.py