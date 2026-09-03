# SAM 3 Evaluation Report — FASDD

**Date**: 2026-06-20 14:25:02
**Dataset**: fasdd
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| fasdd | fire | original_sam3 | 30 | 6 | 47 | 24 | 0.113 | 0.200 | 0.145 | 0.010 | 0.004 | 0.4 |
| fasdd | smoke | original_sam3 | 11 | 8 | 5 | 3 | 0.615 | 0.727 | 0.667 | 0.010 | 0.008 | 0.4 |
| fasdd | fire | finetuned_sam3 | 30 | 6 | 25 | 24 | 0.194 | 0.200 | 0.197 | 0.010 | 0.009 | 0.6 |
| fasdd | smoke | finetuned_sam3 | 11 | 10 | 3 | 1 | 0.769 | 0.909 | 0.833 | 0.010 | 0.009 | 0.6 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| fasdd | original_sam3 | 10 | 4 | 6 | 0 | 0 | 0.400 | 0.000 |
| fasdd | finetuned_sam3 | 10 | 4 | 6 | 0 | 0 | 0.400 | 0.000 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 4 | 0 | 6 | 0 | 0 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
