# SAM 3 Evaluation Report — VISDRONE

**Date**: 2026-06-20 15:06:23
**Dataset**: visdrone
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/final_best/sam3_frozen_best.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| visdrone | car | original_sam3 | 1694 | 1249 | 825 | 445 | 0.602 | 0.737 | 0.663 | 0.047 | 0.032 | 0.3 |
| visdrone | person | original_sam3 | 1903 | 465 | 516 | 1438 | 0.474 | 0.244 | 0.322 | 0.019 | 0.010 | 0.3 |
| visdrone | car | finetuned_sam3 | 1694 | 983 | 257 | 711 | 0.793 | 0.580 | 0.670 | 0.037 | 0.025 | 0.6 |
| visdrone | person | finetuned_sam3 | 1903 | 0 | 0 | 1903 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.6 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| visdrone | original_sam3 | 100 | 1 | 99 | 0 | 0 | 0.010 | 0.000 |
| visdrone | finetuned_sam3 | 100 | 0 | 94 | 6 | 0 | 0.000 | 0.000 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 0 | 0 | 99 | 1 | -1 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
