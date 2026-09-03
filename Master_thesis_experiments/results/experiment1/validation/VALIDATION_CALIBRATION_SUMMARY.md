# Experiment 1 validation calibration

Checkpoint selection was completed on the frozen 120-image screening subset. The selected models were then evaluated on all 548 VisDrone-DET validation images solely to calibrate one operating threshold per model. These values are not final test results.

| Baseline | Frozen threshold | Overall F1 | Tiny recall | Small recall | Regular recall | Macro AP50 | Micro AP50 | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8x | 0.0602799207 | 0.5458 | 0.3756 | 0.6627 | 0.8093 | 0.3447 | 0.4663 | 17.883 |
| YOLO-World | 0.0525619350 | 0.5540 | 0.3829 | 0.7493 | 0.8770 | 0.3895 | 0.4999 | 21.445 |
| Original SAM 3 | 0.4894396067 | 0.7160 | 0.5779 | 0.8558 | 0.9182 | **0.6606** | 0.6953 | **13.199** |
| Adapted SAM 3 without replay | 2.187018e-10 | 0.0388 | 0.0607 | 0.1746 | 0.1632 | 0.0083 | 0.0064 | 250.323 |
| Adapted SAM 3 with replay (epoch 25) | 0.4773983955 | **0.7252** | **0.5953** | **0.8665** | **0.9300** | 0.6511 | **0.7148** | 13.403 |

The with-replay model improves overall F1, micro AP50, and recall in all three scale bins relative to Original SAM 3. Original SAM 3 remains better in macro AP50 and slightly better in FP/image on the complete validation split. The without-replay model's optimal threshold is extremely small and its performance remains poor, quantitatively indicating catastrophic forgetting of the six general/UAV categories after FASDD-only fine-tuning. All models use the official VisDrone `maxDets=500` limit per image.

Test-dev evaluation must use the thresholds above unchanged. Checkpoint selection, threshold optimization, and prompt tuning on test-dev are prohibited.
