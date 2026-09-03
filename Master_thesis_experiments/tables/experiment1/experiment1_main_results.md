# Experiment 1 — Object Scale on VisDrone-DET test-dev

## Dataset scale statistics

| Scale | Definition | Images containing scale | Evaluated GT instances |
|---|---|---:|---:|
| Tiny | $s < 32$ | 1,496 | 48,859 |
| Small | $32 \le s < 64$ | 1,523 | 16,361 |
| Regular | $s \ge 64$ | 1,164 | 7,092 |

Image sets overlap because one image can contain objects from multiple scale bins.

## Final test results

| Model | Tiny F1 | Tiny Recall | Small F1 | Small Recall | Regular F1 | Regular Recall | AP50 | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO | 0.2630 | 0.2232 | 0.3476 | 0.5056 | 0.2486 | 0.5955 | 0.1965 | 14.7323 |
| YOLO-World | 0.2930 | 0.2652 | 0.3880 | 0.6301 | 0.2690 | 0.7300 | 0.2724 | 16.8950 |
| Original SAM 3 | 0.4993 | 0.4434 | **0.5881** | 0.8298 | **0.4308** | 0.9016 | **0.5943** | **10.1553** |
| Adapted SAM 3 without replay | 0.0167 | 0.0634 | 0.0223 | 0.2368 | 0.0076 | 0.1729 | 0.0118 | 210.7248 |
| Adapted SAM 3 with replay | **0.4999** | **0.4608** | 0.5654 | **0.8438** | 0.4036 | **0.9162** | 0.5906 | 11.6907 |

AP50 is the 101-point macro AP at IoU 0.50 over car, person, bus, van, truck, and motorcycle. FP/image is total false positives divided by all 1,610 test images. F1, Recall, and FP/image use per-model confidence thresholds frozen on the complete 548-image validation split. All methods use class-aware matching at IoU 0.50, official VisDrone ignored-region filtering, and maxDets=500/image.

Supplementary all-instance micro AP50: YOLO 0.2896, YOLO-World 0.3526, Original SAM 3 0.5918, without replay 0.0237, with replay 0.5958.
