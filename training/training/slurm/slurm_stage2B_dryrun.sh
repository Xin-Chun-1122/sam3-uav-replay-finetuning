#!/bin/bash
#SBATCH --job-name=sam3_dry
#SBATCH --account=ACD114147
#SBATCH --output=/work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization_frozen/logs/dryrun_%j.out
#SBATCH --error=/work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization_frozen/logs/dryrun_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --partition=normal2

echo "--- Preparing Environment ---"
module load gcc/12.5.0 cuda/12.4

# Setup paths
: "${HF_TOKEN:?Set HF_TOKEN before submitting this job}"
export HF_TOKEN
export PYTHONPATH=/work/nthujerry123/sam3:/work/nthujerry123/sam3/sam3:$PYTHONPATH

# Create log directory
mkdir -p /work/nthujerry123/sam3/sam3_v2/experiments/stage2B_generalization_frozen/logs

echo "--- Starting SAM3 Stage 2B Dry-run ---"
cd /work/nthujerry123/sam3/sam3/sam3

python3.11 train/train.py \
    -c configs/stage2B_generalization_dryrun.yaml \
    --use-cluster 0 \
    --num-gpus 1 \
    --num-nodes 1
