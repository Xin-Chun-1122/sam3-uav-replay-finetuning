# SAM 3 checkpoint screening on VisDrone-DET validation

All rows use the same deterministic 120-image subset (seed 42), six standard category prompts, class-aware IoU 0.50 matching, and the official VisDrone ignored-region rule. Operating thresholds maximize micro F1 on this validation subset. These are model-selection results, not final test results.

| Candidate | Replay/training role | Threshold | Overall F1 | Tiny recall | Small recall | Regular recall | Macro AP50 | Micro AP50 | FP/image |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original SAM 3 | Original | 0.487744 | 0.7080 | 0.5699 | 0.8334 | 0.9270 | 0.6543 | 0.6950 | 13.175 |
| FASDD epoch 40 | Without replay | 6.414e-9 | 0.0323 | 0.0286 | 0.0839 | 0.0295 | 0.0059 | 0.0082 | 93.492 |
| Epoch 25 pure-frozen | **With replay — selected** | 0.482556 | **0.7173** | **0.5828** | **0.8432** | **0.9340** | 0.6494 | **0.7142** | 12.967 |
| `sam3_frozen_best.pt` | With replay | 0.274946 | 0.4588 | 0.1781 | 0.6065 | 0.8890 | 0.4767 | 0.3278 | 11.017 |
| Stage-2B epoch 65 unfrozen | With replay | 0.231396 | 0.4467 | 0.1847 | 0.6094 | 0.8497 | 0.4727 | 0.3178 | 13.708 |
| Stage-2C epoch 75 | With replay | 0.465999 | 0.6293 | 0.4062 | 0.8082 | **0.9368** | 0.5501 | 0.5996 | 12.567 |

Selection conclusion: epoch 25 is the strongest adapted candidate for the thesis's all-instance and scale-focused objectives. Compared with Original SAM 3, it improves overall F1, micro AP50, recall in all three scale bins, and FP/image on this screening subset. Original SAM 3 retains a small macro-AP50 advantage (0.6543 versus 0.6494), which must be reported rather than hidden.
