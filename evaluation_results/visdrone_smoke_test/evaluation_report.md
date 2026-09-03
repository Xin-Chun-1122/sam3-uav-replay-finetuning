# SAM 3 Evaluation Report — VISDRONE

**Date**: 2026-06-20 14:28:51
**Dataset**: visdrone
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| visdrone | car | original_sam3 | 178 | 141 | 51 | 37 | 0.734 | 0.792 | 0.762 | 0.010 | 0.009 | 0.2 |
| visdrone | person | original_sam3 | 391 | 68 | 88 | 323 | 0.436 | 0.174 | 0.249 | 0.010 | 0.006 | 0.2 |
| visdrone | car | finetuned_sam3 | 178 | 119 | 14 | 59 | 0.895 | 0.669 | 0.765 | 0.010 | 0.009 | 0.5 |
| visdrone | person | finetuned_sam3 | 391 | 0 | 0 | 391 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.5 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| visdrone | original_sam3 | 10 | 1 | 9 | 0 | 0 | 0.100 | 0.000 |
| visdrone | finetuned_sam3 | 10 | 0 | 10 | 0 | 0 | 0.000 | 0.000 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 0 | 0 | 9 | 1 | -1 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
