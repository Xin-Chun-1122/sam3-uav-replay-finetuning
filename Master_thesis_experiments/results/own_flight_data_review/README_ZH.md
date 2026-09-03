# 自建無人機飛行資料測試說明

## 測試設定

- 影像來源：`Master_thesis_experiments/test_data`，共 10 張自建飛行影像。
- Ours：`finetune_sam3/checkpoint_25_merged.pt`。
- Original SAM 3：Hugging Face 快取中的官方 `sam3.pt`。
- Ours 固定操作閾值：0.4773983955。
- Original SAM 3 固定操作閾值：0.4894396067。
- person-only 場景使用 prompt `person`。
- 交通場景依內容使用 prompt `car`、`bus`、`person`。
- 所有方框皆由模型推論直接產生，未人工新增、刪除或移動。

## 論文建議主圖

`original_vs_ours/paper_self_collected_person_original_vs_ours.png`

這張圖比較三張自建操場空拍影像。在相同影像與固定操作閾值下，Original SAM 3 均未輸出 person 預測；Ours 分別輸出 4、3、3 個 person 預測，方框目視對應影像中的小型行人。

建議圖說：

> 自建無人機飛行影像之外部泛化定性比較（prompt: person）。在固定操作閾值下，Original SAM 3 未產生有效預測，而 Ours 可定位空拍視角下的小型行人。所有預測框皆為模型直接輸出，未經人工增刪。

## 輸出位置

- 完整 20 組 prompt–image 比較：`original_vs_ours/`
- 可自行排版的六張無文字分圖：`original_vs_ours/paper_panels/`
- 各組預測數量：`original_vs_ours/comparison_counts.csv`
- 可稽核設定：`original_vs_ours/comparison_audit.json`
- Ours 單模型完整結果：本資料夾的 `annotated_prompt_specific/` 與三張 contact sheet。
- 原始預測：`ours_raw_predictions.jsonl`、`original_raw_predictions.jsonl`。

## 論文使用限制

目前自建資料沒有人工 Ground Truth，因此本結果只能作為「外部資料定性泛化測試」，不能由此計算 Precision、Recall、F1 或 AP50，也不能單獨作為統計上證明 Ours 最佳的依據。若要形成正式的自建資料定量實驗，需先對這 10 張影像標註所有目標，再用相同評估程式計算兩個模型的指標。
