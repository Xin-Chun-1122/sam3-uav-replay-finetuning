# 實驗三：Prompt Specificity Evaluation（正式實測結果）

## 1. 實驗狀態

實驗已完成。所有表格數字均由保存的 raw bounding-box predictions 實際計算，
不是預期值或人工填入值。

- Dataset：RefDrone test split
- RefDrone 原始 queries：3,432
- 有 RefDrone query 的唯一影像：1,595
- 三層 specificity 子集：50 張不同影像
- 子集類別分布：person、car、truck、bus、motorcycle 各 10 張
- IoU threshold：0.50
- AP50：101-point interpolated AP
- Operating threshold：
  - Original SAM 3：0.4894396067
  - Adapted SAM 3 with replay：0.4773983955
- Threshold 來源：實驗一 validation split；未使用 RefDrone test 調參

## 2. Ground-truth protocol

RefDrone 與 VisDrone2019-DET test-dev 使用 byte-identical images，但 RefDrone
只標註 referring-expression 所指的物件，並未完整標註廣泛類別的所有物件。
因此：

- 標準類別、同義詞及 Layer 1：使用相同影像的完整 VisDrone evaluation GT。
- Layer 2：使用 RefDrone 中同屬性（本實驗為顏色）的 referent boxes。
- Layer 3：使用該 RefDrone expression 的原始 referent boxes。

這項處理可避免模型正確偵測到「未被 RefDrone expression 提及的同類物件」
時，被錯誤計為 false positive。

完整類別 GT 數：

| Category | GT instances |
|---|---:|
| person | 26,828 |
| car | 27,925 |
| truck | 2,630 |
| bus | 2,931 |
| motorcycle | 5,824 |

## 3. 標準類別名稱（macro average）

| Model | Precision | Recall | F1 | AP50 |
|---|---:|---:|---:|---:|
| Original SAM 3 | 0.6883 | 0.5839 | 0.6150 | 0.5950 |
| Adapted SAM 3 with replay | 0.6228 | 0.6130 | 0.5980 | 0.5929 |

Adapted 的 macro Recall 增加 0.0291，但 Precision 降低 0.0655；macro F1
降低 0.0170，AP50 降低 0.0021。逐類而言，Adapted 的 person、car、
motorcycle F1 較高，而 truck、bus 較低。

## 4. 同義詞（macro average；不含 bus）

| Model | Precision | Recall | F1 | AP50 | Retention |
|---|---:|---:|---:|---:|---:|
| Original SAM 3 | 0.7194 | 0.4873 | 0.5187 | 0.5402 | 0.8514 |
| Adapted SAM 3 with replay | 0.6524 | 0.5115 | 0.5106 | 0.5402 | 0.8413 |

教授指定的同義詞為 human、automobile、lorry、motorbike。因未指定 bus
同義詞，bus 不納入 Retention。兩模型對 automobile、lorry、motorbike
的 retention 接近或高於 0.98；主要弱點是 human：

- Original human retention：0.4164
- Adapted human retention：0.3536

## 5. 固定三層 specificity 子集（50 張）

| Model | Layer | Precision | Recall | F1 | AP50 | Mean GT | Mean predicted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original SAM 3 | 1 | 0.7884 | 0.7617 | 0.7748 | 0.7799 | 18.8800 | 18.2400 |
| Original SAM 3 | 2 | 0.5800 | 0.6444 | 0.6105 | 0.5408 | 3.6000 | 4.0000 |
| Original SAM 3 | 3 | 0.3614 | 0.1987 | 0.2564 | 0.2647 | 3.0200 | 1.6600 |
| Adapted SAM 3 with replay | 1 | 0.8300 | 0.7860 | 0.8074 | 0.8168 | 18.8800 | 17.8800 |
| Adapted SAM 3 with replay | 2 | 0.5769 | 0.3333 | 0.4225 | 0.4637 | 3.6000 | 2.0800 |
| Adapted SAM 3 with replay | 3 | 0.3750 | 0.0993 | 0.1571 | 0.0957 | 3.0200 | 0.8000 |

Layer 定義：

1. 標準類別名稱。
2. 顏色屬性 + 類別。
3. 屬性 + 空間／關係描述的簡短名詞片語。

## 6. 可放入論文的主要結論

Adapted SAM 3 with replay 在廣泛類別層（Layer 1）優於 Original SAM 3，
F1 提升 0.0326，AP50 提升 0.0369，顯示 UAV adaptation 對一般類別定位
有效。然而，加入屬性與空間語言後，Adapted 的 Recall 明顯下降：

- Layer 2 F1：0.4225（Original：0.6105）
- Layer 3 F1：0.1571（Original：0.2564）

因此，實驗支持「adaptation 提升 broad category grounding」，但不支持
「adaptation 全面提升 compositional prompt grounding」。Replay 對標準名稱
與大部分同義詞具有保留效果，但尚不足以保留細緻的屬性—空間組合語意。
此結果應作為方法限制，並可提出後續加入 compositional-language replay、
hard negative expressions 或 referring-expression supervision。

## 7. 正式產物

- 完整表格：`tables/experiment3/EXPERIMENT3_TABLES.md`
- 標準類別 CSV：`tables/experiment3/table1_standard_prompts.csv`
- 同義詞 CSV：`tables/experiment3/table2_synonym_prompts.csv`
- 三層 CSV：`tables/experiment3/table3_specificity_layers.csv`
- Original result JSON：`results/experiment3/sam3_original.json`
- Adapted result JSON：`results/experiment3/sam3_adapted_with_replay.json`
- 鎖定 protocol：`annotations/experiment3/experiment3_locked.json`
- 50 筆 manifest：`annotations/experiment3/specificity_50_manifest.json`

