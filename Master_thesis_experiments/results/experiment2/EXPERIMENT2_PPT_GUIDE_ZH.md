# 實驗二 PPT 最終配置與中文講稿

## 重要結論

目前磁碟中的正式實測結果顯示，整體最佳方法是 **Original SAM 3 +
native-ID refinement**，不是 Adapted SAM 3 with replay。Adapted 模型確實有成功且
優於部分基線的個別事件，可作為定性成功案例，但不能用這些案例取代 160 個事件的
整體量化結果。

## 圖例與追蹤設定

- Prompt：使用事件目標的類別名詞，例如 `car` 或 `truck`。
- 所有方法的 Track ID 都來自 SAM 3 原生 `out_obj_ids`，沒有使用
  ByteTrack、DeepSORT 或外部 ID linker。
- 綠色虛線：局部畫面內所有符合 prompt 的可見 GT。
- 黃色虛線：本事件用來計算 Frame Recall、Recovery Rate 與 AP50 的選定 GT。
- 細青色實線：模型針對同一 prompt 輸出的其他 SAM 3 原生 ID。
- 粗藍框 `ID n OK`：選定的原生 ID 與選定 GT 的 IoU ≥ 0.5。
- 粗紅框 `ID n ERR`：選定 ID 沒有正確對應選定 GT。
- 紫框 `ID n HIDDEN`：Temporary disappearance 期間選定 GT 不可見；不是錯誤判定，
  隱藏影格也不計入 Frame Recall。

同一個時間點的所有模型使用完全相同的局部裁切範圍。圖片不是只偵測一個物件；
所有同 prompt 目標都已畫出，黃色只是指出本事件分析的 target。Partial
occlusion 案例在 Before、During、After 分別以目標為中心裁切，但每一欄的三種
模型仍使用完全相同的 ROI。

---

## 第 1 頁：實驗設計與事件統計

### 英文標題

`Experiment 2: Tracking Evaluation — Experimental Design`

### PPT 放置內容

- Dataset: VisDrone2019-MOT test-dev
- 160 natural events; no synthetic cloud animation
- Prompt: target category name
- Track IDs: native SAM 3 `out_obj_ids`
- IoU threshold: 0.5
- Recovery window: first 10 visible frames after an event

| Condition | Events | Sequences | Median unique targets / sequence | Median duration (frames) |
|---|---:|---:|---:|---:|
| Partial occlusion | 50 | 17 | 3 | 15.0 |
| Temporary complete disappearance | 50 | 15 | 4 | 32.0 |
| Scale decrease and recovery | 50 | 15 | 3 | 6.5 |
| Camera motion / blur | 10 | 10 | 31 | 4.5 |

### 中文講稿

「實驗二使用 VisDrone-MOT test-dev，總共整理 160 個自然追蹤事件，因為樣本數
已經足夠，所以沒有加入合成雲動畫。Event 表示同一個目標完整進入並離開一次困難
情境；Sequences 表示事件來自多少支不同影片；Unique targets 是每支影片曾受該
情境影響的不同目標數中位數；Duration 是每個事件持續影格數的中位數。三種方法
全部使用 SAM 3 原生 track ID，確保比較的是同一個追蹤機制。」

---

## 第 2 頁：整體量化結果

### 英文標題

`Experiment 2: Overall Tracking Performance`

### PPT 放置內容

| Model | Frame Recall | Recovery Rate | AP50 |
|---|---:|---:|---:|
| Original SAM 3 | 0.2991 | 0.3438 | 0.1248 |
| Original SAM 3 + Native-ID Refinement | **0.4284** | **0.5125** | **0.4257** |
| Adapted SAM 3 with Replay | 0.2644 | 0.3312 | 0.1116 |

在表格下方放三個短定義：

- Frame Recall: fixed native ID detected with IoU ≥ 0.5 over visible GT frames
- Recovery Rate: same native ID recovered within 10 visible post-event frames
- AP50: 101-point track AP at IoU 0.5

### 中文講稿

「Frame Recall 衡量一路播放影片時，同一個固定 ID 在多少可見影格中仍正確對應
目標；Recovery Rate 衡量事件結束後十個可見影格內，同一 ID 是否重新找到目標；
AP50 則評估固定 track ID 的定位品質。這組正式實測中，Native-ID refinement 在
三項指標都最高。Adapted with replay 在個別事件有成功案例，但整體尚未提升跨影格
ID 持續性，因此這一頁不能宣稱 Ours 是整體最佳。」

---

## 第 3 頁：各困難情境結果

### 英文標題

`Experiment 2: Performance under Challenging Tracking Conditions`

### PPT 放置內容

| Model | Condition | Frame Recall | Recovery Rate | AP50 |
|---|---|---:|---:|---:|
| Original SAM 3 | Partial occlusion | 0.3525 | 0.4400 | 0.2024 |
| Original SAM 3 | Temporary disappearance | 0.2867 | 0.2400 | 0.1209 |
| Original SAM 3 | Scale decrease/recovery | 0.1897 | 0.2600 | 0.0465 |
| Original SAM 3 | Camera motion/blur | **0.8086** | **0.8000** | **0.7726** |
| Native-ID Refinement | Partial occlusion | **0.4663** | **0.5200** | **0.4610** |
| Native-ID Refinement | Temporary disappearance | **0.4517** | **0.4000** | **0.4505** |
| Native-ID Refinement | Scale decrease/recovery | **0.3417** | **0.5600** | **0.3409** |
| Native-ID Refinement | Camera motion/blur | 0.7486 | **0.8000** | 0.7426 |
| Adapted with Replay | Partial occlusion | 0.2627 | 0.3600 | 0.1130 |
| Adapted with Replay | Temporary disappearance | 0.3056 | 0.2800 | 0.1256 |
| Adapted with Replay | Scale decrease/recovery | 0.1935 | 0.2800 | 0.0683 |
| Adapted with Replay | Camera motion/blur | 0.7143 | 0.7000 | 0.6570 |

### 中文講稿

「分情境結果顯示，Native-ID refinement 在遮擋、暫時消失與尺度變化三類較穩定；
Original SAM 3 在 camera motion/blur 的平均表現最高。Adapted 模型在 temporary
disappearance 與 scale recovery 的部分個案可以保持或恢復原生 ID，但還沒有形成
整體優勢。這也說明追蹤能力除了單張影像辨識外，還受到初始化 ID 是否正確與跨影格
記憶的影響。」

---

## 第 4 頁：可靠尺度進入 Tiny 後的恢復

### 英文標題

`Experiment 2: Recovery after Entering the Tiny Scale Stage`

### PPT 放置內容

| Model | Before Tiny | Tiny Stage | After Returning to Small/Regular | Recovery Rate |
|---|---:|---:|---:|---:|
| Original SAM 3 | 0.2587 | 0.1383 | 0.2373 | 0.2600 |
| Native-ID Refinement | **0.6168** | **0.1803** | **0.4256** | **0.5600** |
| Adapted with Replay | 0.2699 | 0.1357 | 0.2484 | 0.2800 |

旁邊寫：`s = sqrt(w h); tiny: s < 32`

### 中文講稿

「這一頁專門回答教授要求的額外問題。當同一個物件由 small 或 regular 進入 tiny
時，三種方法的 Frame Recall 都下降。回到可靠尺度後，Native-ID refinement
恢復到 0.4256，Recovery Rate 為 0.56，是三者最高。Adapted 模型也有回升，但目前
回升幅度不大。」

---

## 第 5 頁：Partial Occlusion 定性案例

### 英文標題

`Experiment 2: Partial Occlusion — Prompt: "car"`

### 主簡報圖片

使用：

`figures/experiment2/ppt_local_tracking_cases/partial_occlusion/continuous_occlusion/continuous_partial_occlusion_ours_5frames.png`

這張圖只選取官方連續 partial-occlusion 區間 frame 213–229 中的五個影格：
213、217、220、224、229。五張的 selected target 官方 `occlusion=1`，沒有混入
遮擋前或遮擋解除後的影格。

Ours 對局部 ROI 中所有 car 的實際配對數：

- Frame 213：6/6
- Frame 217：6/6
- Frame 220：5/5
- Frame 224：4/4
- Frame 229：4/4

完整逐車 IoU 稽核紀錄：

`figures/experiment2/ppt_local_tracking_cases/partial_occlusion/continuous_occlusion/audit.json`

### 基線比較備份頁

做 3 列 × 3 欄：

- 欄：Before frame 205 / During frame 220 / After frame 237
- 列：Original SAM 3 / Native-ID Refinement / Ours

個別圖片位於：

`figures/experiment2/ppt_local_tracking_cases/partial_occlusion/`

檔名格式是：

- `before_sam3_original.png`
- `during_sam3_native_id_tracker.png`
- `after_sam3_adapted_with_replay.png`

其餘格子依相同規則放置。頁面角落寫：

`Natural event: partial_occlusion-018; selected GT ID 12; duration 17 frames`

### 中文講稿

「這個案例的 prompt 是 car。目標從 frame 213 到 229 連續被官方標記為 partial
occlusion 等級 1；投影片顯示其中五個時間點，全部都還在同一段遮擋區間，沒有混入
遮擋解除後的畫面。每一張局部 ROI 內看得到的所有 car 都有 Ours 的真實 prediction
配對，分別是 6/6、6/6、5/5、4/4 與 4/4。黃色虛線是 selected target 12，
藍色 ID 5 是 Ours 使用的 SAM 3 原生 ID。完整事件中，Ours 在遮擋階段 15/17 個
影格正確，事件後 15/15 個影格成功恢復。Original 也成功；Native-ID baseline
事件後為 0/15，所以此案例證明 Ours 優於 Native-ID baseline，但不是只有 Ours
成功。」

---

## 第 6 頁：Temporary Complete Disappearance 定性案例

### 英文標題

`Experiment 2: Temporary Complete Disappearance — Prompt: "car"`

### 圖片排列

做 3 列 × 3 欄：

- 欄：Before frame 103 / Hidden frame 115 / After frame 123
- 列：Original SAM 3 / Native-ID Refinement / Ours

圖片位於：

`figures/experiment2/ppt_local_tracking_cases/temporary_complete_disappearance/`

頁面角落寫：

`Natural event: temporary_complete_disappearance-031; target absent for 14 frames`

### 中文講稿

「這台車在 frame 103 還看得到，之後進入天橋下方，從 frame 109 到 122 完全沒有
GT，因此中間的紫框代表 target hidden，不是模型判錯。車輛在 frame 123 再出現時，
Original 的 ID 2 和 Native-ID baseline 的 ID 0 都沒有恢復到該目標；Ours 的
原生 ID 1 重新對應同一台車，事件後 15/15 個可見影格都正確。」

---

## 第 7 頁：Scale Decrease and Recovery 定性案例

### 英文標題

`Experiment 2: Scale Decrease and Recovery — Prompt: "car"`

### 圖片排列

做 3 列 × 3 欄：

- 欄：Before frame 216 / Tiny frame 219 / After frame 227
- 列：Original SAM 3 / Native-ID Refinement / Ours

圖片位於：

`figures/experiment2/ppt_local_tracking_cases/scale_decrease_and_recovery/`

頁面下方寫：

`s: 32.31 px → 30.30 px → 32.86 px`

### 中文講稿

「這個案例不是用肉眼猜 tiny，而是按照 s 等於根號 w 乘 h 計算。目標由
32.31 pixel 降到 30.30 pixel，因此確實進入 tiny，再回到 32.86 pixel。
Original 的固定 ID 沒有正確對應；Native-ID baseline 在回到可靠尺度後只有
5/15 個影格正確；Ours 的 ID 3 在三個階段分別為 15/15、5/5 與 15/15，這是一個
真實的成功恢復案例。」

---

## 第 8 頁：Camera Motion 定性案例

### 英文標題

`Experiment 2: Strong Camera Motion — Prompt: "truck"`

### 圖片排列

做 3 列 × 3 欄：

- 欄：Before frame 428 / During frame 435 / After frame 442
- 列：Original SAM 3 / Native-ID Refinement / Ours

圖片位於：

`figures/experiment2/ppt_local_tracking_cases/camera_motion_or_blur/`

頁面下方寫：

`Median global motion = 8.484 px; sequence threshold = 1.488 px; 5.70× threshold`

### 中文講稿

「這個代表案例屬於 camera motion 分支，不是 blur 分支。事件中的全域位移中位數
是 8.484 pixel，而該序列的門檻是 1.488 pixel，因此是門檻的 5.70 倍。Prompt 是
truck，所有貨車都會被畫出。Ours 的 ID 0 在事件前、中、後都正確；Native-ID
baseline 在事件前與相機運動期間對錯目標；Original 也成功，所以這張圖只能說
Ours 優於 Native-ID baseline，不能說只有 Ours 成功。」

---

## 第 9 頁：10 秒內追蹤影片

### 英文標題

`Experiment 2: Video Demonstration of ID Recovery`

### 最推薦影片

`videos/experiment2/experiment2_overpass_disappearance_ours_recovery_8s_h264.mp4`

- H.264 MP4
- 1280 × 900
- 7.33 秒
- Prompt: `car`
- Ours selected native SAM 3 ID: 1

備用影片：

`videos/experiment2/experiment2_partial_occlusion_ours_recovery_8s_h264.mp4`

- H.264 MP4
- 1280 × 900
- 8.29 秒
- Prompt: `car`
- Ours selected native SAM 3 ID: 8

PowerPoint 設定：插入 → 視訊 → 此裝置；播放選項設為「按一下時播放」或「自動」，
不要勾選循環播放。主簡報放天橋消失案例即可，Partial occlusion 影片放備份頁。

### 中文講稿

「影片中的 prompt 是 car。綠色虛線表示畫面內所有可見車輛 GT，青色實線表示
模型輸出的其他原生 ID；藍色 ID 1 是本事件追蹤的目標。車輛進入天橋下方時 GT
暫時消失，重新出現後 Ours 仍以同一個原生 ID 1 對應該車，這就是 Recovery Rate
所衡量的成功恢復。」

---

## 簡報中不可使用的說法

- 不可把 Adapted SAM 3 with replay 稱為「整體最佳模型」。
- 不可把 Camera motion 案例說成「模糊案例」。
- 不可把紫色 hidden 框說成模型正確或錯誤；該影格沒有可見 GT。
- 不可只用成功案例推論整體 160 個事件都較好。

如果要讓 Ours 成為量化最佳，必須使用新的 checkpoint 重新跑完三種方法與 160 個
事件，並以新產生的 JSON 實測結果替換表格，不能直接調整目前數字。
