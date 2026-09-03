#!/bin/bash
#SBATCH --job-name=sam3_stage2
#SBATCH --output=/work/nthujerry123/sam3/sam3_v2/experiments/stage2_uav_multidomain/logs/train_%j.out
#SBATCH --error=/work/nthujerry123/sam3/sam3_v2/experiments/stage2_uav_multidomain/logs/train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=normal2
#SBATCH --account=ACD114147
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL

echo "=============================="
echo "SAM3-I Stage 2 Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================="

# ── Environment Setup ──────────────────────────────────────────
echo "--- Loading CUDA 12.4 ---"
module load cuda/12.4

export PYTHONPATH=/work/nthujerry123/sam3/sam3:$PYTHONPATH
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# ── Paths ──────────────────────────────────────────────────────
SAM3_V2_DIR=/work/nthujerry123/sam3/sam3_v2
SAM3_TRAIN=/work/nthujerry123/sam3/sam3/sam3/train/train.py
STAGE1_CKPT=/work/nthujerry123/sam3/experiments/merged_finetune_2xh200/checkpoints/checkpoint_40.pt
EXP_DIR=/work/nthujerry123/sam3/sam3_v2/experiments/stage2_uav_multidomain

# ── Disk Safety Daemon ─────────────────────────────────────────
# Keep only the 1 most recent numbered checkpoint to save disk space
CKPT_DIR=$EXP_DIR/checkpoints
bash /work/nthujerry123/sam3/auto_clean_checkpoints.sh $CKPT_DIR 1 &
CLEANER_PID=$!
echo "Disk cleaner PID: $CLEANER_PID"

# ── Verify Stage 1 Checkpoint ─────────────────────────────────
if [ ! -f "$STAGE1_CKPT" ]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    exit 1
fi
echo "Stage 1 checkpoint: $STAGE1_CKPT"
echo "Stage 1 checkpoint size: $(du -sh $STAGE1_CKPT | cut -f1)"

# ── Create Required Directories ───────────────────────────────
mkdir -p $CKPT_DIR
mkdir -p $EXP_DIR/dumps/multidomain
mkdir -p $EXP_DIR/tensorboard

# ── Training ──────────────────────────────────────────────────
echo ""
echo "Starting Stage 2 Training..."
echo "Config: $SAM3_V2_DIR/configs/training/stage2_multidomain.yaml"
echo ""

python $SAM3_TRAIN \
    --config-path $SAM3_V2_DIR/configs/training \
    --config-name stage2_multidomain \
    launcher.experiment_log_dir=$EXP_DIR \
    trainer.checkpoint.resume_from=$STAGE1_CKPT \
    launcher.gpus_per_node=1 \
    launcher.num_nodes=1

TRAIN_EXIT=$?

# ── Cleanup ───────────────────────────────────────────────────
kill $CLEANER_PID 2>/dev/null
echo ""
echo "=============================="
echo "Training finished: $(date)"
echo "Exit code: $TRAIN_EXIT"
echo "=============================="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "SUCCESS: Stage 2 training completed!"
    ls -lh $CKPT_DIR/
else
    echo "FAILED: Training exited with code $TRAIN_EXIT"
fi
