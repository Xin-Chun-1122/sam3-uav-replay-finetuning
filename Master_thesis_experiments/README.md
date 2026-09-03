# 無人機影像之語言引導物件偵測與追蹤

這個目錄保存論文實驗的設定、資料清單、標註、模型預測、評估結果與論文表格。原始大型資料集不複製進來；每次執行會保存資料來源、檔案清單與統計，確保結果可重現。

## 固定實驗規則

- 隨機種子：`42`。
- Detection IoU：`0.50`。
- VisDrone 的 `pedestrian (1)` 與 `people (2)` 合併為 `person`。
- 實驗一類別：`car, person, bus, van, truck, motorcycle`。
- 尺度定義：`s = sqrt(width * height)`；`tiny < 32`、`small 32 <= s < 64`、`regular >= 64`。
- 尺度是 **ground-truth instance 層級** 屬性。一張圖可同時出現在多個尺度 manifest；尺度評估時，其他尺度 GT 必須設為 ignore，不能當成 false positive。
- 實驗二的每個 prediction track ID 必須來自 **SAM 3 native tracker**。ByteTrack、SORT 或自行逐幀編號不得冒充 SAM 3 track ID。
- confidence threshold 不可用 test split 調參；應先在 validation split 固定，再一次性評估 test。

完整凍結規格見 [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md)。

## 目錄

```text
configs/                 固定設定與資料/模型路徑
data_manifests/          可重現的影像、影片與事件清單
annotations/             人工事件與 prompt 標註
predictions/             各 baseline 原始預測
results/                 JSON/CSV 評估結果
tables/                  論文可直接引用的表格
logs/                    執行紀錄
scripts/                 統計與評估程式
docs/                    實驗定義、標註手冊
```

## 第一步：實驗一尺度統計

```bash
cd /home/alien/sam3/Master_thesis_experiments
../.venv/bin/python scripts/analyze_visdrone_scales.py \
  --dataset-root /home/alien/Downloads/VisDrone2019-DET-test-dev \
  --output-dir results/experiment1/scale_statistics \
  --manifest-dir data_manifests/experiment1
```

輸出包括 `summary.json`、`scale_counts.csv`、`class_scale_counts.csv`、`image_scale_membership.csv`，以及三個尺度的影像 manifest。

## 實驗一完成狀態（2026-07-21）

五個 baseline 已在完整 1,610 張 VisDrone-DET test-dev 上完成推論與評估：YOLO、YOLO-World、Original SAM 3、Adapted SAM 3 without replay、Adapted SAM 3 with replay。所有 operating thresholds 均在完整 548 張 validation split 上先行鎖定，test 結果的 `threshold_source` 皆為 `frozen_external`。

- 中文完整報告：[`results/experiment1/EXPERIMENT1_FINAL_REPORT_ZH.md`](results/experiment1/EXPERIMENT1_FINAL_REPORT_ZH.md)
- 教授主表（Markdown）：[`tables/experiment1/experiment1_main_results.md`](tables/experiment1/experiment1_main_results.md)
- 論文表格（CSV）：[`tables/experiment1/experiment1_main_results.csv`](tables/experiment1/experiment1_main_results.csv)
- 論文表格（LaTeX）：[`tables/experiment1/experiment1_main_results.tex`](tables/experiment1/experiment1_main_results.tex)
- Scale statistics：[`tables/experiment1/dataset_scale_statistics.csv`](tables/experiment1/dataset_scale_statistics.csv)
- Raw/result SHA-256 manifest：[`tables/experiment1/artifact_manifest.json`](tables/experiment1/artifact_manifest.json)
