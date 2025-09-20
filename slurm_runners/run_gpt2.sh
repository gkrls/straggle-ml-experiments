#!/usr/bin/bash
#SBATCH -t 48:00:00
#SBATCH -J "gpt"
#SBATCH -N 6
#SBATCH -A hpc-prf-fessllm
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=gpt_%j.out
#SBATCH --error=gpt_%j.err
set -euo pipefail

ml lang
ml Miniforge3

export PYTHONUSERBASE=/pc2/groups/hpc-prf-fessllm/laxmanvj/.conda
export PATH=/scratch/hpc-prf-fessllm/laxmanvj/.conda/george_straggle_env/bin:$PATH


# --- minimal config (can be overridden by env) ---
IFACE="${IFACE:-enp226s0f0}"                 # network interface to read IP from
BACKEND="${BACKEND:-gloo}"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT="${MASTER_PORT:-29670}"

export GLOO_SOCKET_IFNAME=$IFACE
export GLOO_LOG_LEVEL=DEBUG

NCCL_DEBUG=INFO 
NCCL_DEBUG_SUBSYS=INIT,NET
NCCL_SOCKET_IFNAME="$IFACE" 
NCCL_IB_HCA=mlx5_0,mlx5_1

srun python -u ../models/gpt2.py \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend gloo \
  --workers 8 \
  --epochs 12 \
  --steps_per_epoch 6000 \
  --mini_val_every_steps 300 \
  --gradient_accumulation_steps 5 \
  --batch_size 12 \
  --seq_len 1024 \
  --amp \
  --prefetch_factor 4 \
  --log_every_steps 50 \
  --json ../models/gpt2.json \
  --data /scratch/hpc-prf-fessllm/laxmanvj/openwebtext/parquet \
  --cache_dir /scratch/hpc-prf-fessllm/laxmanvj/openwebtext/parquet/cache \
  --slurm_setup