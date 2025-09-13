#!/usr/bin/bash
#SBATCH -t 48:00:00
#SBATCH -J "bert-straggle"
#SBATCH -N 6
#SBATCH -A hpc-prf-fessllm
#SBATCH -p gpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=bert_straggle_%j.out
#SBATCH --error=bert_straggle_%j.err
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

srun python -u ../models/bert_finetune.py \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --backend gloo \
  --data /scratch/hpc-prf-fessllm/laxmanvj/squad/squad \
  --squad_version v1 \
  --epochs 5 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --warmup_ratio 0.1 \
  --deterministic \
  --workers 8 \
  --prefetch_factor 4 \
  --straggle_points 3 \
  --straggle_prob 2 \
  --straggle_ranks 1 \
  --straggle_amount 1.35 \
  --straggle_multiply 0.5 2 \
  --straggle_verbose \
  --json ../models/bert_finetune_straggle.json \
  --slurm_setup