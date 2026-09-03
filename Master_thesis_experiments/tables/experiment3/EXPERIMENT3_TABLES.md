# Experiment 3 — Prompt Specificity Evaluation

All values below are measured on the frozen RefDrone test protocol. IoU threshold is 0.50; operating thresholds were frozen on the Experiment 1 validation split.

## Table 1. Standard category prompts

| Model | Category | Prompt | Precision | Recall | F1 | AP50 |
|---|---|---|---:|---:|---:|---:|
| Original SAM 3 | person | person | 0.7443 | 0.3319 | 0.4591 | 0.3906 |
| Original SAM 3 | car | car | 0.7356 | 0.8202 | 0.7756 | 0.7889 |
| Original SAM 3 | truck | truck | 0.5777 | 0.6418 | 0.6081 | 0.6173 |
| Original SAM 3 | bus | bus | 0.7487 | 0.7045 | 0.7260 | 0.7262 |
| Original SAM 3 | motorcycle | motorcycle | 0.6350 | 0.4208 | 0.5062 | 0.4521 |
| Original SAM 3 | Macro average | — | 0.6883 | 0.5839 | 0.6150 | 0.5950 |
| Adapted SAM 3 with replay | person | person | 0.7111 | 0.3442 | 0.4639 | 0.3922 |
| Adapted SAM 3 with replay | car | car | 0.7673 | 0.8217 | 0.7936 | 0.8054 |
| Adapted SAM 3 with replay | truck | truck | 0.4643 | 0.6798 | 0.5518 | 0.6000 |
| Adapted SAM 3 with replay | bus | bus | 0.5841 | 0.7404 | 0.6530 | 0.7015 |
| Adapted SAM 3 with replay | motorcycle | motorcycle | 0.5870 | 0.4791 | 0.5276 | 0.4654 |
| Adapted SAM 3 with replay | Macro average | — | 0.6228 | 0.6130 | 0.5980 | 0.5929 |

## Table 2. Synonym prompts

| Model | Category | Synonym | Precision | Recall | F1 | AP50 | Retention |
|---|---|---|---:|---:|---:|---:|---:|
| Original SAM 3 | person | human | 0.9008 | 0.1069 | 0.1912 | 0.3239 | 0.4164 |
| Original SAM 3 | car | automobile | 0.7194 | 0.8266 | 0.7693 | 0.7870 | 0.9919 |
| Original SAM 3 | truck | lorry | 0.6327 | 0.6027 | 0.6173 | 0.6132 | 1.0152 |
| Original SAM 3 | motorcycle | motorbike | 0.6247 | 0.4129 | 0.4972 | 0.4366 | 0.9822 |
| Original SAM 3 | Macro average | — | 0.7194 | 0.4873 | 0.5187 | 0.5402 | 0.8514 |
| Adapted SAM 3 with replay | person | human | 0.7697 | 0.0918 | 0.1640 | 0.2889 | 0.3536 |
| Adapted SAM 3 with replay | car | automobile | 0.7632 | 0.8184 | 0.7898 | 0.8011 | 0.9953 |
| Adapted SAM 3 with replay | truck | lorry | 0.4929 | 0.6612 | 0.5648 | 0.6109 | 1.0236 |
| Adapted SAM 3 with replay | motorcycle | motorbike | 0.5839 | 0.4748 | 0.5237 | 0.4600 | 0.9927 |
| Adapted SAM 3 with replay | Macro average | — | 0.6524 | 0.5115 | 0.5106 | 0.5402 | 0.8413 |

Bus is excluded from synonym retention because no bus synonym was specified in the experiment.

## Table 3. Fixed three-layer specificity subset (50 images)

| Model | Layer | Prompt scope | Precision | Recall | F1 | AP50 | Mean GT | Mean predicted |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Original SAM 3 | 1 | Category | 0.7884 | 0.7617 | 0.7748 | 0.7799 | 18.8800 | 18.2400 |
| Original SAM 3 | 2 | Category + attribute | 0.5800 | 0.6444 | 0.6105 | 0.5408 | 3.6000 | 4.0000 |
| Original SAM 3 | 3 | Attribute + spatial/relational description | 0.3614 | 0.1987 | 0.2564 | 0.2647 | 3.0200 | 1.6600 |
| Adapted SAM 3 with replay | 1 | Category | 0.8300 | 0.7860 | 0.8074 | 0.8168 | 18.8800 | 17.8800 |
| Adapted SAM 3 with replay | 2 | Category + attribute | 0.5769 | 0.3333 | 0.4225 | 0.4637 | 3.6000 | 2.0800 |
| Adapted SAM 3 with replay | 3 | Attribute + spatial/relational description | 0.3750 | 0.0993 | 0.1571 | 0.0957 | 3.0200 | 0.8000 |
