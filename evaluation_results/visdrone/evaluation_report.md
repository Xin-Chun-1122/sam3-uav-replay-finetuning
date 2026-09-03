# SAM 3 Evaluation Report — VISDRONE

**Date**: 2026-06-22 14:46:53
**Dataset**: visdrone
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| visdrone | car | original_sam3 | 28074 | 20473 | 14271 | 7601 | 0.589 | 0.729 | 0.652 | 0.670 | 0.410 | 0.4 |
| visdrone | person | original_sam3 | 27382 | 7352 | 8105 | 20030 | 0.476 | 0.268 | 0.343 | 0.225 | 0.084 | 0.4 |
| visdrone | car | finetuned_sam3 | 28074 | 16028 | 4881 | 12046 | 0.767 | 0.571 | 0.654 | 0.508 | 0.285 | 0.6 |
| visdrone | person | finetuned_sam3 | 27382 | 36 | 10 | 27346 | 0.783 | 0.001 | 0.003 | 0.010 | 0.009 | 0.6 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| visdrone | original_sam3 | 1610 | 15 | 1589 | 6 | 0 | 0.009 | 0.000 |
| visdrone | finetuned_sam3 | 1610 | 21 | 1461 | 128 | 0 | 0.013 | 0.000 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 9 | 12 | 1583 | 6 | 6 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
