#!/bin/bash
#SBATCH -J charge-gcn
#SBATCH -o slurm-%j.out
#SBATCH -t 14:00:00
#SBATCH -A cis250151p
#SBATCH -p GPU-shared
#SBATCH --qos=gpu
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /jet/home/kchekuru/dl-proj2
source .venv/bin/activate

export POSCAR_DIR=${POSCAR_DIR:-/ocean/projects/cis250151p/shared/Data/PASCAR}
export CHARGE_DIR=${CHARGE_DIR:-/ocean/projects/cis250151p/shared/Data/CHARGESSS}
export NUM_EPOCHS=${NUM_EPOCHS:-400}
export BATCH_SIZE=${BATCH_SIZE:-16}
export HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-192}
export NUM_LAYERS=${NUM_LAYERS:-4}
export DROPOUT=${DROPOUT:-0.1}
export LR=${LR:-3e-4}
export WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
export LR_PATIENCE=${LR_PATIENCE:-12}
export EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-40}
export RESUME=${RESUME:-0}
export REQUIRE_GPU=${REQUIRE_GPU:-1}

python train.py
