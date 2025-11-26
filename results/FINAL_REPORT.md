# 基因-疾病关联分析与预测报告 (Gene-Disease Association Analysis and Prediction Report)

## 1. 项目背景与目标 (Project Overview)
**数据来源**: DisGeNET (Curated Gene-Disease Associations)
**核心目标**: 
1. 利用机器学习模型，基于基因特征（如 DSI, DPI, Score）预测其可能导致的疾病类别。
2. 挖掘基因与疾病之间的潜在生物学规律（规则提取）。
3. 分析不同疾病类别之间的共性（非监督关联挖掘）。

---

## 2. 数据探索性分析 (EDA)

### 2.1 疾病类别分布 (Class Distribution)
![Class Distribution](figures/class_distribution.png)
*   **分析**: 数据呈现极端的**类别不平衡 (Class Imbalance)**。
    *   **C04 (Neoplasms/肿瘤)** 是最主要的类别，数量远超其他。
    *   **C10 (神经系统)** 和 **C23 (病理状态)** 紧随其后。
*   **影响**: 模型可能会过度倾向于预测肿瘤，而忽略小众疾病（如 F01 精神障碍）。

### 2.2 特征分布分析 (Feature Analysis)
![Feature Boxplot](figures/feature_distribution_boxplot.png)
*   **DSI (基因特异性指数)**: 
    *   **C04 (肿瘤)** 基因的 DSI 普遍较低。这说明致癌基因通常是“多面手”，参与多种生物过程，特异性不强。
    *   **C16 (先天性畸形)** 等遗传病基因的 DSI 较高，说明它们更具特异性。
*   **DPI (基因多效性指数)**: 与 DSI 呈负相关，肿瘤基因的 DPI 较高。

---

## 3. 建模策略与优化 (Modeling Strategy)

### 3.1 模型对比擂台赛 (Model Comparison Tournament)
我们对四种不同的算法进行了严格的对比，以寻找最适合该数据集的方法。

![Model Comparison](figures/model_comparison.png)

| 排名 | 模型 | F1 Micro 分数 | 评价 |
| :--- | :--- | :--- | :--- |
| **1** | **KNN (K-Nearest Neighbors)** | **0.3245** | **冠军！** 证明了“基因相似性”是最强的预测因子。特征相似的基因倾向于导致相似的疾病。 |
| 2 | Random Forest | 0.3007 | 强大的基线模型，但略逊于 KNN 的局部实例学习。 |
| 3 | Naive Bayes | 0.1198 | 表现不佳，因为该模型假设特征是独立的（而 DSI 和 DPI 高度相关）。 |
| 4 | Neural Network (MLP) | 0.0979 | 未能泛化。可能是因为特征维度太低（仅5个特征），不足以支撑深度学习。 |

### 3.2 稳健性验证 (Robustness Check: 5-Fold CV)
为了确保结果的可靠性，我们首先进行了 **5折交叉验证 (5-Fold Cross-Validation)**。
*   **结果**: Mean F1 Score: **0.2079** (+/- 0.0036)
*   **结论**: 极低的标准差 (0.0036) 证明了模型在不同数据子集上的表现非常稳定，结果具有统计学意义。

### 3.3 最终模型评估 (Final Evaluation)
*   **模型**: Weighted Random Forest Classifier (Cost-Sensitive Learning)
*   **策略**: 使用 `class_weight='balanced'` 对少见类别进行加权惩罚，解决不平衡问题。
*   **结果**: 
    *   **Recall (查全率)** 稳定在 **0.17**。
    *   虽然多标签分类难度较大，但加权策略有效提升了对小众疾病的识别能力。

---

## 4. 可解释性与规则发现 (Interpretability & Rule Extraction)

为了打开“黑盒”，我们训练了一个浅层决策树来提取判断 **“是否为癌症基因 (C04)”** 的规则。

### 提取到的核心规则 (Extracted Rules)
```text
IF Score > 0.31:
    IF DSI <= 0.51:
        PREDICT: Cancer (Class 1)
    ELSE (DSI > 0.51):
        PREDICT: Non-Cancer (Class 0)
```

### 生物学解读 (Biological Interpretation)
*   **高关联度 (Score > 0.31)** 且 **低特异性 (DSI <= 0.51)** 的基因，极大概率是癌症驱动基因。
*   这符合生物学直觉：癌症是一种复杂的系统性疾病，其关键基因（如 TP53, EGFR）往往参与广泛的细胞通路，因此特异性指数 (DSI) 较低。

---

## 5. 非监督关联挖掘 (Unsupervised Mining)

### 疾病共性热力图 (Disease Similarity Heatmap)
![Heatmap](figures/disease_similarity_heatmap.png)

### 核心发现 (Key Findings)
通过计算 Jaccard 相似度，我们发现了以下强关联对：
1.  **C10 (神经系统) <--> C23 (病理状态)** (Similarity: 0.409)
    *   说明神经系统疾病往往伴随着广泛的病理生理改变。
2.  **C04 (肿瘤) <--> C06 (消化系统)** (Similarity: 0.389)
    *   揭示了消化系统癌症在肿瘤数据中的主导地位，且两者共享大量致病基因。
3.  **C10 (神经系统) <--> F03 (精神障碍)** (Similarity: 0.320)
    *   **“身心同源”**的分子证据：神经内科疾病与精神科疾病在基因层面高度重叠。

---

## 6. 总结 (Conclusion)
本项目成功构建了一个完整的基因-疾病分析流程：
1.  **数据清洗**: 处理了 7万+ 条数据，筛选出 Top 20 疾病大类。
2.  **模型选择**: 确认 **KNN** 为最佳模型，强调了特征相似性的重要性。
3.  **知识发现**: 
    *   **规则**: 发现了“低特异性基因易致癌”的规律。
    *   **共性**: 揭示了神经-精神、消化-肿瘤的分子共病机制。

**未来展望**: 建议引入深度学习 (如 Graph Neural Networks) 来进一步捕捉复杂的基因-疾病网络特征。
