#!/usr/bin/bash
#SBATCH -t 48:00:00
#SBATCH -J "lstm"
#SBATCH -N 6
#SBATCH -A hpc-prf-fessllm
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=lstm_%j.out
#SBATCH --error=lstm_%j.err
set -euo pipefail

ml lang
ml Miniforge3

export PYTHONUSERBASE=/pc2/groups/hpc-prf-fessllm/laxmanvj/.conda
export PATH=/scratch/hpc-prf-fessllm/laxmanvj/.conda/george_straggle_env/bin:$PATH


# --- minimal config (can be overridden by env) ---
IFACE="${IFACE:-enp226s0f0}"                 # network interface to read IP from
BACKEND="${BACKEND:-gloo}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29500}"

export GLOO_SOCKET_IFNAME=$IFACE
export GLOO_LOG_LEVEL=DEBUG

NCCL_DEBUG=INFO 
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_SOCKET_IFNAME="$IFACE" 
NCCL_IB_HCA=mlx5_0,mlx5_1

srun python -u $HOME/straggle-ml-experiments/models/lstm.py \
  --slurm_setup \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend gloo \
  --epochs 12 \
  --batch_size 32 \
  --workers 8 \
  --prefetch_factor 8 \
  --deterministic \
  --json $HOME/straggle-ml-experiments/models/lstm.json