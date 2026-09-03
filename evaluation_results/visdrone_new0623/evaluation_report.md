# SAM 3 Evaluation Report — VISDRONE

**Date**: 2026-06-23 18:03:49
**Dataset**: visdrone
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/best0623/checkpoint_25_converted.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| visdrone | car | original_sam3 | 28074 | 20473 | 14271 | 7601 | 0.589 | 0.729 | 0.652 | 0.670 | 0.410 | 0.4 |
| visdrone | person | original_sam3 | 27382 | 7352 | 8105 | 20030 | 0.476 | 0.268 | 0.343 | 0.225 | 0.084 | 0.4 |
| visdrone | car | finetuned_sam3 | 28074 | 0 | 1388 | 28074 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.7 |
| visdrone | person | finetuned_sam3 | 27382 | 0 | 1328 | 27382 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.7 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| visdrone | original_sam3 | 1610 | 15 | 1589 | 6 | 0 | 0.009 | 0.000 |
| visdrone | finetuned_sam3 | 1610 | 0 | 0 | 1603 | 7 | 0.000 | 1.000 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 0 | 0 | 1595 | 15 | -15 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
