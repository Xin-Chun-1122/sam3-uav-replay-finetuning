# SAM 3 Evaluation Report — FASDD

**Date**: 2026-06-23 17:27:27
**Dataset**: fasdd
**Confidence threshold**: 0.3
**IoU threshold**: 0.5
**NMS threshold**: 0.5
**Original checkpoint**: /home/alien/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt
**Fine-tuned checkpoint**: /home/alien/sam3/finetune_sam3/best0623/checkpoint_25_converted.pt

## Object-level Metrics

| Dataset | Class | Model | GT | TP | FP | FN | Precision | Recall | F1 | AP50 | mAP50:95 | FPS |
|---------|-------|-------|----|----|----|----|-----------|--------|----|------|----------|-----|
| fasdd | fire | original_sam3 | 6049 | 1907 | 11097 | 4142 | 0.147 | 0.315 | 0.200 | 0.125 | 0.045 | 0.7 |
| fasdd | smoke | original_sam3 | 2877 | 1804 | 2302 | 1073 | 0.439 | 0.627 | 0.517 | 0.501 | 0.273 | 0.7 |
| fasdd | fire | finetuned_sam3 | 6049 | 197 | 2578 | 5852 | 0.071 | 0.033 | 0.045 | 0.010 | 0.003 | 0.8 |
| fasdd | smoke | finetuned_sam3 | 2877 | 1193 | 2783 | 1684 | 0.300 | 0.415 | 0.348 | 0.369 | 0.169 | 0.8 |

## Image-level Success/Failure

| Dataset | Model | Total | Success | Partial Fail | Complete Fail | False Alarm | Success Rate | FA Rate |
|---------|-------|-------|---------|--------------|---------------|-------------|--------------|---------||
| fasdd | original_sam3 | 4181 | 1972 | 1447 | 469 | 293 | 0.472 | 0.147 |
| fasdd | finetuned_sam3 | 4181 | 213 | 1101 | 958 | 1909 | 0.051 | 0.956 |

## Fine-tuning Comparison

| Both Success | Improved by FT | Both Failed | FT Regression | Net Improvement |
|-------------|----------------|-------------|---------------|-----------------|
| 100 | 113 | 2096 | 1872 | -1759 |

## Notes

- Threshold was set via CLI, not tuned on test set.
- AP computed with pycocotools COCOeval.
- TP/FP/FN use IoU ≥ 0.50 with confidence threshold from CLI.
