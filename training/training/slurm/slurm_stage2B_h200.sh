#!/bin/bash
#SBATCH --job-name=sam3_stg2b
#SBATCH --account=ACD114147
#SBATCH --output=/work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization/logs/train_%j.out
#SBATCH --error=/work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=normal2

echo "--- Preparing Environment ---"
module load gcc/12.5.0 cuda/12.4

# Setup paths
: "${HF_TOKEN:?Set HF_TOKEN before submitting this job}"
export HF_TOKEN
export PYTHONPATH=/work/nthujerry123/sam3:/work/nthujerry123/sam3/sam3:$PYTHONPATH

# Create log directory
mkdir -p /work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization/logs

echo "--- Starting SAM3 Stage 2B (Generalization) Training ---"
# Start background disk monitor to prevent disk full issues
python3.11 /work/nthujerry123/sam3/sam3_v2/utils/disk_monitor.py \
    --dir /work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization/checkpoints \
    --max_checkpoints 3 > /dev/null 2>&1 &
MONITOR_PID=$!

cd /work/nthujerry123/sam3/sam3/sam3

# Use Python 3.11 which has PyTorch 2.4+ installed
python3.11 train/train.py \
    -c configs/stage2B_generalization.yaml \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1

echo "--- Training Completed ---"
kill $MONITOR_PID
