#!/bin/bash
#SBATCH --job-name=sam3_stage2A
#SBATCH --output=/work/nthujerry123/sam3/sam3_v2/experiments/stage2A_language_recovery/logs/train_%j.out
#SBATCH --error=/work/nthujerry123/sam3/sam3_v2/experiments/stage2A_language_recovery/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=normal2
#SBATCH --account=ACD114147
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

echo "=============================="
echo "SAM3-I Stage 2A Training (Language Recovery)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================="

# ── Environment Setup ──────────────────────────────────────────
echo "--- Loading Modules ---"
module load gcc/12.5.0 cuda/12.4

# Authenticate before submission with `hf auth login`, or export HF_TOKEN in
# the shell that invokes sbatch. Never store access tokens in this script.
: "${HF_TOKEN:?Set HF_TOKEN before submitting this job}"
export HF_TOKEN
export PYTHONPATH=/work/nthujerry123/sam3:/work/nthujerry123/sam3/sam3:$PYTHONPATH
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# ── Paths ──────────────────────────────────────────────────────
SAM3_V2_DIR=/work/nthujerry123/sam3/sam3_v2
SAM3_TRAIN=/work/nthujerry123/sam3/sam3/sam3/train/train.py
STAGE1_CKPT=/work/nthujerry123/sam3/experiments/merged_finetune_2xh200/checkpoints/checkpoint_40.pt
EXP_DIR=/work/nthujerry123/sam3/sam3_v2/experiments/stage2A_language_recovery

# ── Disk Safety Daemon ─────────────────────────────────────────
# Keep only the 1 most recent numbered checkpoint to save disk space
CKPT_DIR=$EXP_DIR/checkpoints
bash /work/nthujerry123/sam3/auto_clean_checkpoints.sh $CKPT_DIR 1 &
CLEANER_PID=$!
echo "Disk cleaner PID: $CLEANER_PID"

# ── Create Required Directories ───────────────────────────────
mkdir -p $CKPT_DIR
mkdir -p $EXP_DIR/dumps/multidomain
mkdir -p $EXP_DIR/tensorboard
mkdir -p $EXP_DIR/logs

# ── Training ──────────────────────────────────────────────────
cd /work/nthujerry123/sam3

echo ""
echo "Starting Stage 2A Training on 1x H200 GPU..."
echo "Config: configs/stage2A_language_recovery.yaml"
echo ""

python3.11 sam3/sam3/train/train.py \
    -c configs/stage2A_language_recovery.yaml \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1

TRAIN_EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────
kill $CLEANER_PID 2>/dev/null
echo ""
echo "=============================="
echo "Training finished: $(date)"
echo "Exit code: $TRAIN_EXIT"
echo "=============================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "SUCCESS: Stage 2A training completed!"
    ls -lh $CKPT_DIR/
else
    echo "FAILED: Training exited with code $TRAIN_EXIT"
fi
