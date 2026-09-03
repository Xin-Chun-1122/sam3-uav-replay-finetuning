# Frozen experiment protocol

## Experiment 1 — Object Scale (VisDrone-DET test)

Target classes and VisDrone IDs:

| Thesis class | VisDrone category ID |
|---|---:|
| person | 1 (pedestrian), 2 (people) |
| car | 4 |
| van | 5 |
| truck | 6 |
| bus | 9 |
| motorcycle | 10 (motor) |

For every valid target GT box, compute `s = sqrt(w*h)` in original-image pixels. Use tiny `<32`, small `[32,64)`, and regular `>=64`. Scale-specific Recall and F1 use GT scale ranges; GT from the other two ranges is ignored during matching. Report micro-averaged TP/FP/FN over all six categories and also retain per-class values.

Baselines: YOLO, YOLO-World, Original SAM 3, Adapted SAM 3 without replay, Adapted SAM 3 with replay. Every model must save raw per-image predictions (`image_id, category, score, xyxy`) before evaluation.

Report:

- Tiny/small/regular: Precision (audit value), Recall, F1.
- All target objects and all test images: AP50.
- All target objects and all test images: `FP/image = total FP / 1610` (or the validated test image count).

Use the same threshold-selection policy for every baseline: optimize one confidence threshold for micro F1 on the complete 548-image VisDrone validation split, then freeze it before test-dev inference. Use class-aware IoU matching for every baseline. AP50 uses ranked, unrounded confidence scores rather than the operating threshold used by F1/Recall/FP-image. Follow the official VisDrone rule that removes GT and detections whose box area is at least 50% covered by category-0 ignored regions; score-0 target GT remains available as an ignored match.

## Experiment 2 — Tracking Evaluation (VisDrone-MOT)

Event conditions:

1. Partial occlusion (target remains partially visible).
2. Temporary complete disappearance (target has no visible pixels, then reappears).
3. Scale decrease and recovery (target crosses from reliable scale to tiny and later returns to small/regular).
4. Camera motion / blur.

An event is one continuous affected interval for one GT target. A target can contribute multiple events. Dataset summary reports: number of events, number of distinct source sequences, median unique affected targets per selected tracking clip/sequence, and median event duration in frames (plus seconds).

Natural events must be selected before synthetic supplementation. If a requested quota is not met, synthetic cloud occlusion may only be added to unobstructed clips and must be reported separately (`source_type=synthetic_cloud`); it must never be silently mixed into a natural-only result.

Required event annotation fields are defined in `annotations/experiment2/events.schema.json`. Frame ranges are inclusive and zero-based.

### Track ID invariant

All submitted tracks must contain `track_id_source = "sam3_native"`. The ID is the `obj_id/out_obj_ids` returned by the SAM 3 video session. A re-identification or recovery method may link SAM 3 track segments, but must preserve the segment IDs and separately record the derived global ID. ByteTrack/SORT IDs are not valid SAM 3 native IDs.

Baselines:

- Original SAM 3 native tracker.
- ID recovery/linking algorithm over SAM 3 native track segments.
- Adapted SAM 3 with replay, using its SAM 3 native tracker.

Metrics:

- Frame Recall: visible GT target-frames matched at IoU >= 0.50 / evaluable visible GT target-frames.
- Recovery Rate: events for which the same target is correctly reacquired within a frozen recovery window after the event / eligible events.
- AP50: frame-level detection AP at IoU 0.50.
- Reliable → low-reliability → reliable analysis: report recovery and ID continuity separately; detection recovery does not imply identity recovery.

## Experiment 3 — Prompt Specificity (RefDrone test)

Categories: person, car, truck, bus, motorcycle.

1. Standard category prompts: report Precision, Recall, F1, AP50.
2. Synonyms: person→human, car→automobile, truck→lorry, motorcycle→motorbike. Bus has no predeclared synonym and is excluded from synonym retention unless a synonym is frozen before evaluation. Report the same metrics and `Retention = synonym F1 / standard-name F1` per category and macro average.
3. Select 50 test examples with exactly three annotated prompt layers: category → attribute → attribute plus spatial/relational description. Report each layer's Precision, Recall, F1, AP50, mean GT count, and mean predicted-target count.

The 50 examples and all three prompts/GT referents must be frozen in a manifest before running model comparisons.

RefDrone test uses byte-identical images from VisDrone2019-DET test-dev but its
referring-expression annotations do not exhaustively label every object of a broad
category. Therefore, standard-name prompts, synonym prompts, and specificity Layer 1
use the exhaustive VisDrone evaluation GT for the same images. Layers 2 and 3 use
RefDrone attribute/referent GT. This prevents correctly detected but unmentioned
same-class objects from being incorrectly counted as false positives. The 50-example
manifest remains fixed before model comparison.
