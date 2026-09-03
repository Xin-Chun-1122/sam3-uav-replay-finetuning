# Model registry (must be frozen before evaluation)

| Baseline / candidate | Checkpoint / package | Verified status |
|---|---|---|
| YOLO | `/home/alien/sam3/yolov8x.pt` | Found; YOLOv8x closed-vocabulary COCO baseline |
| YOLO-World | `Master_thesis_experiments/models/yolov8x-worldv2.pt` | Downloaded from Ultralytics official model integration; open vocabulary |
| Original SAM 3 | Hugging Face cached `sam3.pt` | Found; original detector + native tracker |
| Adapted SAM 3 without replay | `models/sam3_fasdd_no_replay_merged.pt` | Epoch 40 from `experiments/fasdd_finetune`; the matching config documents FASDD_UAV only (25,097 images; fire and smoke). It was merged with the original SAM 3 checkpoint, preserving 309 tracker tensors. |
| Adapted SAM 3 with replay (selected) | `finetune_sam3/checkpoint_25_merged.pt` | Epoch 25 frozen-backbone Stage-2C model, trained with VisDrone + FASDD + COCO/RefCOCO replay. Selected before test evaluation using a fixed 120-image validation screening set. |
| Adapted SAM 3 with replay (former primary) | `finetune_sam3/final_best/sam3_frozen_best.pt` | Inference-ready merged frozen Stage-2B. Its config/report uses `stage2B_mix` and records language replay; validation screening was weaker than epoch 25. |
| Adapted SAM 3 with replay (secondary ablation) | `models/sam3_stage2b_unfrozen_merged.pt` | Epoch 65 unfrozen Stage-2B detector merged into original SAM 3; 1,106 detector tensors replaced and 309 native tracker tensors preserved. |
| Adapted SAM 3 with replay (frozen alternate) | `finetune_sam3/checkpoint_25_merged.pt` | Epoch 25 from `stage2_pure_frozen_official`; its `stage2C_ultimate_mix` contains VisDrone + FASDD + COCO/RefCOCO. Inference smoke test passed and 309 native SAM 3 tracker tensors are present. |

## Important interpretation

The two checkpoints supplied as “best” and “second best” are useful, but they are not the two sides of the replay ablation: both Stage-2B configurations use mixed replay data. They support a separate **frozen versus unfrozen** ablation. The thesis replay ablation must compare otherwise matched training runs whose only intended difference is replay data.

Do not substitute an arbitrary checkpoint for a missing baseline. A checkpoint is accepted only when its training configuration/data mixture proves whether replay was enabled. Checkpoint selection and operating thresholds are performed on VisDrone validation data; the test-dev split is not used to choose a model or threshold.

`best0623/checkpoint_25_converted.pt` is not suitable for tracking evaluation: it contains only 1,106 `detector.*` tensors and no `tracker.*` tensors. It also must not be labelled “without replay”. For this run, use the complete `checkpoint_25_merged.pt` only as an additional **with-replay, frozen-backbone** candidate. Here, “pure frozen” describes which model parameters were frozen; it does not describe the replay data composition.
