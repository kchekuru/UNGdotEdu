#!/bin/bash
#SBATCH -J gpu-sanity
#SBATCH -o slurm-gpu-sanity-%j.out
#SBATCH -t 00:05:00
#SBATCH -A cis250151p
#SBATCH -p GPU-shared
#SBATCH --qos=gpu
#SBATCH --export=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:1

set -euo pipefail

cd /jet/home/kchekuru/dl-proj2
source .venv/bin/activate

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | head -n 20
fi

python - <<'PY'
import torch
import torch_geometric

print('torch_version', torch.__version__)
print('torch_cuda_build', torch.version.cuda)
print('pyg_version', torch_geometric.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable on allocated GPU node.')

name = torch.cuda.get_device_name(0)
print('gpu_name', name)

x = torch.randn(2048, 2048, device='cuda')
y = torch.randn(2048, 2048, device='cuda')
z = x @ y
print('sanity_matmul_mean', float(z.mean().item()))
print('GPU sanity check passed.')
PY
