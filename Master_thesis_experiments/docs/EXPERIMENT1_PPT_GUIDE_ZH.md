# 實驗一 PPT 與論文呈現指南

## 必須先修正的兩件事

1. 舊圖 `qualitative_adapted_strict_wins_all_baselines_all_scales.png` 是單一
   GT 的局部比較，只能放附錄，不能作為實驗一的主要定性結果。
2. 舊簡報中的 with/without replay 數字是預期值，不是正式 test 實測值。
   正文必須改用本文件的正式結果。

## 評估定義

- Dataset：VisDrone2019-DET test-dev，共 1,610 張影像。
- Categories：car、person、bus、van、truck、motorcycle。
- 物件尺度：\(s=\sqrt{wh}\)。
- Tiny：\(s<32\)。
- Small：\(32\le s<64\)。
- Regular：\(s\ge64\)。
- 尺度是每個 GT instance 的屬性，不是整張影像的唯一屬性。
- 同一張影像可以同時包含 Tiny、Small、Regular，因此三組影像數會重疊。
- IoU threshold：0.50。
- Confidence threshold：每個模型先在 548 張 validation images 上決定，
  再固定使用於 test。
- 使用 class-aware one-to-one matching、VisDrone ignored-region filtering、
  maxDets=500/image。
- AP50 是六類別的 macro mAP50；FP/image 是全部 FP 除以 1,610。

## 建議頁數

- 口試／完整報告：7 頁。
- 時間很短時：6 頁，將 Slide 1 與 Slide 2 合併。
- 舊的「單一目標 strict-win 圖」只放附錄，不算在 7 頁內。

## Slide 1

### Title

**Experiment 1 — Object Scale Evaluation: Experimental Design**

### 版面

- 左上：研究問題與 Dataset。
- 右上：尺度公式和三個區間。
- 中間：Baselines 與 Metrics。
- 下半部放 `experiment1_scale_examples_all_gt.png`。

### 中文講稿

「實驗一評估模型在不同物件尺度下的語言引導偵測能力。我使用
VisDrone2019-DET test-dev 的 1,610 張影像，評估六個 UAV 常見類別。
每個物件依照平方根面積 \(s=\sqrt{wh}\) 分成 Tiny、Small 和 Regular。
請注意尺度是物件層級的屬性，所以同一張影像可能同時屬於多個尺度組。
所有模型使用相同 IoU 0.5 matching protocol，而 confidence threshold
都先在 validation set 決定，沒有使用 test set 調整。」

## Slide 2

### Title

**Experiment 1 — Object Scale Distribution on VisDrone-DET**

### 表格

| Scale | Definition | Images containing scale | Evaluated GT instances | GT ratio |
|---|---|---:|---:|---:|
| Tiny | \(s<32\) | 1,496 | 48,859 | 67.6% |
| Small | \(32\le s<64\) | 1,523 | 16,361 | 22.6% |
| Regular | \(s\ge64\) | 1,164 | 7,092 | 9.8% |

表格下方必須註明：

> Image counts overlap because one image may contain objects from multiple scale bins.

### 中文講稿

「測試集中共有 72,312 個可評估目標，其中 67.6% 是 Tiny，代表 UAV
影像主要困難來自小尺度目標。這裡的影像數不是互斥切分，而是包含該尺度
至少一個目標的影像數，所以三列相加會超過 1,610。」

## Slide 3

### Title

**Experiment 1 — Quantitative Results across Object Scales**

### 正式 test 結果

| Model | Tiny F1 | Tiny R | Small F1 | Small R | Regular F1 | Regular R | mAP50 | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8x | 0.2630 | 0.2232 | 0.3476 | 0.5056 | 0.2486 | 0.5955 | 0.1965 | 14.7323 |
| YOLOv8x-WorldV2 | 0.2930 | 0.2652 | 0.3880 | 0.6301 | 0.2690 | 0.7300 | 0.2724 | 16.8950 |
| Original SAM 3 | 0.4993 | 0.4434 | **0.5881** | 0.8298 | **0.4308** | 0.9016 | **0.5943** | **10.1553** |
| Adapted SAM 3 without replay | 0.0167 | 0.0634 | 0.0223 | 0.2368 | 0.0076 | 0.1729 | 0.0118 | 210.7248 |
| Adapted SAM 3 with replay (Ours) | **0.4999** | **0.4608** | 0.5654 | **0.8438** | 0.4036 | **0.9162** | 0.5906 | 11.6907 |

### 版面規則

- Ours 放最後一列，以淡紫色或淡橘色底標示。
- 每欄最佳值才用粗體，不要把 Ours 的所有數字全部粗體。
- F1、Recall、mAP50 標註 ↑；FP/image 標註 ↓。
- 表格下方放一行 protocol，不把 protocol 塞進表格。

### 中文講稿

「量化結果顯示，with replay 模型在 Tiny F1，以及 Tiny、Small、
Regular 三個尺度的 Recall 都是最高，表示微調後提高了跨尺度的目標覆蓋率。
相較 without replay，replay 明顯避免模型崩潰與大量 false positives。
不過 Original SAM 3 在 Small F1、Regular F1、macro mAP50 和 FP/image
仍略優，因此本實驗支持的是 recall-oriented robustness improvement，
而不是所有指標全面超越。」

## Slide 4

### Title

**Experiment 1 — Full-scene Comparison with Detection Baselines**

### 圖片

放 `qualitative_full_scene_detection_baselines.png`，圖片盡量滿版。

### 中文講稿

「這張圖改用完整影像呈現，而不是只框一個挑選目標。每個 panel 都保留
六類別的所有可評估 GT。綠色虛線是 GT、藍色是 IoU 大於等於 0.5 的
true positive、紅色是 false positive。三列分別是 Tiny、Small 和
Regular-dominant 場景。在這些代表性影像中，Ours 找到的目標數較多，
尤其 Tiny 和 Small 場景的差異最明顯。」

## Slide 5

### Title

**Experiment 1 — Ablation Study: Effect of Replay**

### 圖片

放 `qualitative_full_scene_sam_ablation.png`，圖片盡量滿版。

### 中文講稿

「這一頁比較 Original SAM 3、without replay 與 with replay。
Without replay 在完整影像中產生大量 false positives，顯示只使用新任務
資料會破壞原有語言 grounding 能力。加入 replay 後，模型在三種尺度都能
恢復穩定偵測，而且代表性場景的 TP 和 Recall 高於 Original。這是 replay
在本研究中的主要貢獻。」

## Slide 6

### Title

**Experiment 1 — Annotated Moving-video Comparison**

### 影片

放 `experiment1_annotated_detection_comparison_6s.mp4`，影片佔頁面
約 85%，不再另外放靜態圖。

- 左側：Original SAM 3。
- 右側：Adapted SAM 3 with replay (Ours)。
- 每一張影格皆以文字 prompt `car` 獨立推論，未使用 temporal tracker。
- Tiny／Small／Regular 的虛線框是該影格的全部有效 car GT。
- 青色實線是 TP；紅色實線是 FP；每格顯示 TP、FP、FN。
- 播放設定：Automatically、Mute、Rewind after Playing，不要循環。
- 頁面角落註明：
  `Qualitative VisDrone-MOT demo; official metrics use VisDrone-DET.`

### 中文講稿

「這段影片不是照片輪播，也不是人工畫框。左右兩個模型都在每張連續影格
上重新使用 car prompt 做語言引導偵測，沒有沿用上一格的追蹤結果。虛線
顯示所有有效 car GT，顏色代表 Tiny、Small 和 Regular；青色是 TP，紅色
是 FP。這個代表性片段中 Ours 找到更多有效車輛且誤檢較少。影片屬於
VisDrone-MOT 的動態定性展示，正式實驗一結論仍以完整 VisDrone-DET
test-dev 數據為準。」

## Slide 7

### Title

**Experiment 1 — Key Findings**

### 重點

- Ours achieves the highest recall across Tiny, Small, and Regular objects.
- Replay prevents the severe degradation observed without replay.
- The gain is strongest in target coverage, while precision/calibration remains a limitation.
- Future work：confidence calibration、hard-negative replay、scale-aware loss。

### 中文講稿

「總結來說，微調加 replay 的主要價值是提高不同尺度的目標覆蓋率，
並避免 without replay 的 catastrophic forgetting。Tiny F1 與三個尺度的
Recall 都有所提升，但 precision 與 macro mAP50 仍有改善空間，因此後續
可以加入 confidence calibration、hard-negative replay 或 scale-aware loss。」

## 論文圖片

- 正文主圖：`qualitative_full_scene_all_models_all_scales.pdf`
- 若版面太擠，可拆成：
  - `qualitative_full_scene_detection_baselines.pdf`
  - `qualitative_full_scene_sam_ablation.pdf`
- 單尺度圖與舊單目標局部圖放附錄。

### 論文圖說

> Full-scene qualitative comparison on representative Tiny, Small, and
> Regular-dominant VisDrone-DET test images. All evaluable instances from the
> six target categories are shown. Green dashed boxes denote ground truth,
> blue boxes denote true-positive predictions at IoU >= 0.50, and red boxes
> denote false positives. Predictions use validation-frozen model-specific
> confidence thresholds.

## 影片

正式推薦：
`experiment1_annotated_detection_comparison_6s.mp4`。這是兩個模型在
40 張連續影格上的實際 per-frame inference，使用與實驗一相同的 prompt
及 validation-frozen confidence thresholds。最後一秒的 TP／FP／FN 是
這個代表性短片的描述性統計，不是 test-dev benchmark 指標。

來源是 VisDrone-MOT，因此投影片必須標註「Qualitative demo」；正式
Experiment 1 quantitative evidence 仍是 VisDrone-DET test-dev。論文正文
使用靜態 PDF，不嵌入影片，可在補充材料或 QR code／連結提供影片。

不推薦 `uav_moving_vehicles_8s.mp4`（沒有模型標註）及
`experiment1_full_scene_scale_comparison.mp4`（靜態圖片輪播）。
