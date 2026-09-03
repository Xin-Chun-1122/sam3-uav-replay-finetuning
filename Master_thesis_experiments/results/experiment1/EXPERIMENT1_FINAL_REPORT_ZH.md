# 實驗一：無人機影像物件尺度評估

## 實驗目的

本實驗評估語言引導物件偵測模型在 VisDrone-DET test-dev 中面對不同物件尺度時的偵測能力。評估類別為 car、person、bus、van、truck 與 motorcycle。

對每一個有效 GT bounding box，以原始影像像素計算：

\[
s=\sqrt{wh}.
\]

為避免邊界樣本被遺漏，尺度採互斥定義：tiny 為 $s<32$、small 為 $32\le s<64$、regular 為 $s\ge64$。

## 測試資料尺度統計

| 尺度 | 定義 | 包含該尺度的影像數 | 正式評估 GT 數 |
|---|---|---:|---:|
| Tiny | $s<32$ | 1,496 | 48,859 |
| Small | $32\le s<64$ | 1,523 | 16,361 |
| Regular | $s\ge64$ | 1,164 | 7,092 |

三組影像數會重疊，因為同一張影像可以同時包含不同尺度的物件；正式指標是在 GT instance 層級分組。

## 評估設定

- Test set：VisDrone2019-DET test-dev，共 1,610 張影像。
- IoU threshold：0.50，採 class-aware greedy matching。
- Confidence threshold：每個模型分別在完整 548 張 VisDrone validation set 上以 micro F1 校正，之後固定套用於 test-dev。
- AP50：六類別 101-point interpolated macro AP50。
- FP/image：全部 test false positives 除以 1,610。
- 每張影像最多保留 500 個最高分 detections。
- 依官方 VisDrone 規則排除至少 50% box 面積落在 category-0 ignored region 的 GT 與 detection。
- YOLO 與 YOLO-World 使用 1024 輸入解析度；SAM 3 使用其原生 1008 解析度。

## 正式 test-dev 結果

| Model | Tiny F1 | Tiny Recall | Small F1 | Small Recall | Regular F1 | Regular Recall | AP50 | FP/image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO | 0.2630 | 0.2232 | 0.3476 | 0.5056 | 0.2486 | 0.5955 | 0.1965 | 14.7323 |
| YOLO-World | 0.2930 | 0.2652 | 0.3880 | 0.6301 | 0.2690 | 0.7300 | 0.2724 | 16.8950 |
| Original SAM 3 | 0.4993 | 0.4434 | **0.5881** | 0.8298 | **0.4308** | 0.9016 | **0.5943** | **10.1553** |
| Adapted SAM 3 without replay | 0.0167 | 0.0634 | 0.0223 | 0.2368 | 0.0076 | 0.1729 | 0.0118 | 210.7248 |
| Adapted SAM 3 with replay | **0.4999** | **0.4608** | 0.5654 | **0.8438** | 0.4036 | **0.9162** | 0.5906 | 11.6907 |

補充的 all-instance micro AP50 分別為：YOLO 0.2896、YOLO-World 0.3526、Original SAM 3 0.5918、without replay 0.0237、with replay 0.5958。

## 結果分析

與 Original SAM 3 相比，with-replay 模型的 Tiny、Small 與 Regular Recall 分別提高 0.0174、0.0140 與 0.0147，且 Tiny F1 由 0.4993 小幅提高至 0.4999。這表示 replay 微調提高了不同尺度物件的偵測敏感度，核心尺度召回指標均為五個 baseline 中最高。

另一方面，with-replay 的 Small F1 與 Regular F1 分別降低 0.0227 與 0.0273，macro AP50 降低 0.0037，FP/image 增加 1.5354。這顯示召回率提升伴隨 precision 與 false-positive 的取捨。因此適合在論文中主張「with-replay 提升多尺度召回能力，尤其改善 tiny 物件偵測」，但不應宣稱所有指標全面優於 Original SAM 3。

Without-replay 模型的 macro AP50 僅 0.0118，FP/image 達 210.7248。即使使用 validation 校正後的模型專屬 threshold，其一般 UAV 類別能力仍明顯崩解，呈現 FASDD-only 微調造成的災難性遺忘。加入 VisDrone 與一般語言資料 replay 後，with-replay 模型的 macro AP50 回升至 0.5906，證明 replay 對保留既有語意類別能力具有必要性。

## 建議論文結論文字

「實驗結果顯示，採用 replay 的適應式 SAM 3 在 tiny、small 與 regular 三種尺度均取得最高召回率，其中 tiny recall 由原始 SAM 3 的 0.4434 提升至 0.4608。相較之下，未使用 replay 的 FASDD-only 模型出現顯著災難性遺忘。雖然 replay 模型在 small/regular F1 與整體 macro AP50 上未超越原始模型，但其結果呈現出明確的 recall–precision trade-off，證明 replay 策略能在保留一般語意能力的同時提升 UAV 多尺度目標的偵測敏感度。」
