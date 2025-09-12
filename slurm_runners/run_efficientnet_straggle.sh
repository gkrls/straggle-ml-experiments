#!/usr/bin/bash
#SBATCH -t 48:00:00
#SBATCH -J "efficientnet-straggle"
#SBATCH -N 6
#SBATCH -A hpc-prf-fessllm
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=efficientnet_straggle_%j.out
#SBATCH --error=efficientnet_straggle_%j.err
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

srun python -u $HOME/straggle-ml-experiments/models/efficientnet_2.py \
  --slurm_setup \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend gloo \
  --data /scratch/hpc-prf-fessllm/laxmanvj/ddp_experiments/src/data/imagenet/imagenet \
  --model efficientnet_b0 \
  --epochs 90 \
  --batch_size 128 \
  --deterministic \
  --drop_last_val \
  --prefetch_factor 6 \
  --workers 8 \
  --straggle_points 3 \
  --straggle_prob 2 \
  --straggle_ranks 1 \
  --straggle_amount 0.92 \
  --straggle_multiply 0.5 2 \
  --straggle_verbose \
  --json $HOME/straggle-ml-experiments/models/efficientnet_straggle.json