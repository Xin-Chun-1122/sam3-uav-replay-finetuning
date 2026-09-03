# SAM 3 Evaluation Report — FASDD

**Date**: 2026-06-20 14:54:58
**Dataset**: fasdd
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| fasdd | fire | original_sam3 | 153 | 50 | 250 | 103 | 0.167 | 0.327 | 0.221 | 0.010 | 0.006 | 0.6 |
| fasdd | smoke | original_sam3 | 71 | 41 | 54 | 30 | 0.432 | 0.577 | 0.494 | 0.018 | 0.012 | 0.6 |
| fasdd | fire | finetuned_sam3 | 153 | 67 | 111 | 86 | 0.376 | 0.438 | 0.405 | 0.015 | 0.010 | 0.7 |
| fasdd | smoke | finetuned_sam3 | 71 | 57 | 19 | 14 | 0.750 | 0.803 | 0.776 | 0.020 | 0.016 | 0.7 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| fasdd | original_sam3 | 100 | 48 | 35 | 10 | 7 | 0.480 | 0.137 |
| fasdd | finetuned_sam3 | 100 | 59 | 33 | 6 | 2 | 0.590 | 0.039 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 46 | 13 | 39 | 2 | 11 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
