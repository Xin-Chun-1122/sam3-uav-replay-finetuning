# 實驗二：VisDrone-MOT Tracking Evaluation

## 評估設定

- 測試資料集：VisDrone2019-MOT test-dev
- 評估事件：160 筆，全部來自 VisDrone-MOT 自然影像，未使用合成雲動畫
- IoU 判定門檻：0.5
- Recovery window：遮蔽或低可靠事件結束後的前 10 個可見影格
- Track ID：三種方法皆使用 SAM 3 原生 `out_obj_ids`，未使用 ByteTrack、DeepSORT 或其他外部 ID linker
- Original SAM 3 與 Adapted SAM 3 with replay 使用文字提示初始化
- ID tracking baseline 使用 Original SAM 3，先以文字建立 SAM 3 原生 ID，再以事件前 GT 中心點選定及修正既有原生 ID；後續追蹤完全由 SAM 3 原生 tracker 執行

Frame Recall 定義為同一個固定 SAM 3 原生 ID 在可見 GT 影格中達到 IoU ≥ 0.5 的比例。Recovery Rate 定義為事件結束後前 10 個可見影格內，同一原生 ID 是否重新達到 IoU ≥ 0.5。AP50 為以固定原生 track ID 對事件目標計算的 101-point track AP50。

## 表一：四種追蹤情境統計

| 情境 | Events | Sequences | 每支序列 Unique targets 中位數 | Event duration 中位數（frames） | 來源 |
|---|---:|---:|---:|---:|---|
| Partial occlusion | 50 | 17 | 3 | 15.0 | Natural VisDrone-MOT |
| Temporary complete disappearance | 50 | 15 | 4 | 32.0 | Natural VisDrone-MOT |
| Scale decrease and recovery | 50 | 15 | 3 | 6.5 | Natural VisDrone-MOT |
| Camera motion / blur | 10 | 10 | 31 | 4.5 | Natural VisDrone-MOT |

## 表二：整體追蹤結果

| Model | Frame Recall | Recovery Rate | AP50 |
|---|---:|---:|---:|
| Original SAM 3 | 0.2991 | 0.3438 | 0.1248 |
| Original SAM 3 + native-ID refinement | **0.4284** | **0.5125** | **0.4257** |
| Adapted SAM 3 with replay | 0.2644 | 0.3312 | 0.1116 |

## 表三：各情境追蹤結果

| Model | Condition | Frame Recall | Recovery Rate | AP50 |
|---|---|---:|---:|---:|
| Original SAM 3 | Partial occlusion | 0.3525 | 0.4400 | 0.2024 |
| Original SAM 3 | Temporary complete disappearance | 0.2867 | 0.2400 | 0.1209 |
| Original SAM 3 | Scale decrease and recovery | 0.1897 | 0.2600 | 0.0465 |
| Original SAM 3 | Camera motion / blur | **0.8086** | **0.8000** | **0.7726** |
| Original SAM 3 + native-ID refinement | Partial occlusion | **0.4663** | **0.5200** | **0.4610** |
| Original SAM 3 + native-ID refinement | Temporary complete disappearance | **0.4517** | **0.4000** | **0.4505** |
| Original SAM 3 + native-ID refinement | Scale decrease and recovery | **0.3417** | **0.5600** | **0.3409** |
| Original SAM 3 + native-ID refinement | Camera motion / blur | 0.7486 | **0.8000** | 0.7426 |
| Adapted SAM 3 with replay | Partial occlusion | 0.2627 | 0.3600 | 0.1130 |
| Adapted SAM 3 with replay | Temporary complete disappearance | 0.3056 | 0.2800 | 0.1256 |
| Adapted SAM 3 with replay | Scale decrease and recovery | 0.1935 | 0.2800 | 0.0683 |
| Adapted SAM 3 with replay | Camera motion / blur | 0.7143 | 0.7000 | 0.6570 |

## 表四：由可靠尺度進入 tiny，再回到可靠尺度

| Model | 進入 tiny 前 Frame Recall | tiny 階段 Frame Recall | 回到 small/regular 後 Frame Recall | Scale-event Recovery Rate |
|---|---:|---:|---:|---:|
| Original SAM 3 | 0.2587 | 0.1383 | 0.2373 | 0.2600 |
| Original SAM 3 + native-ID refinement | **0.6168** | **0.1803** | **0.4256** | **0.5600** |
| Adapted SAM 3 with replay | 0.2699 | 0.1357 | 0.2484 | 0.2800 |

## 結果說明

Original SAM 3 + native-ID refinement 在整體 Frame Recall、Recovery Rate 與 AP50 均為最佳，並在 partial occlusion、temporary complete disappearance 與 scale decrease/recovery 三種情境取得最佳結果。這表示在事件前明確選定並修正正確的 SAM 3 原生 ID，能顯著提升後續追蹤穩定度。

尺度分析顯示三種方法在物件進入 tiny 階段時都明顯下降。回到 small/regular 後，Original SAM 3、native-ID refinement 與 Adapted SAM 3 with replay 的 Frame Recall 分別恢復至 0.2373、0.4256 與 0.2484；其中 native-ID refinement 的 scale-event Recovery Rate 最高，為 0.5600。

目前 Adapted SAM 3 with replay 並未優於 Original SAM 3：整體 Frame Recall、Recovery Rate 與 AP50 分別低 0.0347、0.0125 與 0.0133。因此這組實測結果不能宣稱 replay checkpoint 改善追蹤；可如實解釋為目前的微調主要未提升跨影格 ID 持續性，後續應加入具時間連續性的 replay 或 tracking loss 再驗證。
