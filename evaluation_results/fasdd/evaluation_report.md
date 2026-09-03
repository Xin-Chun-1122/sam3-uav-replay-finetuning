# SAM 3 Evaluation Report — FASDD

**Date**: 2026-06-20 18:25:39
**Dataset**: fasdd
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| fasdd | fire | original_sam3 | 6049 | 1907 | 11097 | 4142 | 0.147 | 0.315 | 0.200 | 0.125 | 0.045 | 0.7 |
| fasdd | smoke | original_sam3 | 2877 | 1804 | 2302 | 1073 | 0.439 | 0.627 | 0.517 | 0.501 | 0.273 | 0.7 |
| fasdd | fire | finetuned_sam3 | 6049 | 3386 | 4804 | 2663 | 0.413 | 0.560 | 0.476 | 0.455 | 0.211 | 0.8 |
| fasdd | smoke | finetuned_sam3 | 2877 | 2290 | 834 | 587 | 0.733 | 0.796 | 0.763 | 0.766 | 0.503 | 0.8 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| fasdd | original_sam3 | 4181 | 1972 | 1447 | 469 | 293 | 0.472 | 0.147 |
| fasdd | finetuned_sam3 | 4181 | 2630 | 1267 | 263 | 21 | 0.629 | 0.011 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 1913 | 717 | 1492 | 59 | 658 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
