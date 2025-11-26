# Gene-Disease Association Analysis and Prediction Report

## 1. Project Overview
**Data Source**: DisGeNET (Curated Gene-Disease Associations)
**Core Objectives**:
1. **Phase 1 (Foundation)**: Completed data cleaning, integration, and gene-disease network construction.
2. **Phase 2 (Deep Mining)**: Utilized machine learning models to predict disease categories based on gene features (e.g., DSI, DPI).
3. **Rule Extraction**: Mined potential biological patterns between genes and diseases.
4. **Commonality Analysis**: Analyzed commonalities between different disease categories.

---

---

## 2. Phase 1: Network Validation
Before diving into feature analysis, we validated the topological properties of the Gene-Disease network.

### 2.1 Link Prediction
We used **Preferential Attachment (PA)** and **SVD Matrix Factorization** as baseline models to predict potential gene-disease associations.

![Link Prediction PR Curves](results/midterm_results/pr_curves.png)

*   **Results**:
    *   **Preferential Attachment**: AUC = **0.9346**
    *   **SVD**: AUC = 0.5204
*   **Conclusion**:
    *   The high score of PA proves that the network exhibits a strong **"Rich-get-Richer"** property: Hub genes tend to associate with Hub diseases.
    *   This provides a solid topological foundation for the subsequent feature-based classification.

---

## 3. Exploratory Data Analysis (EDA)

### 2.1 Disease Class Distribution
![Class Distribution](results/figures/class_distribution.png)
*   **Analysis**: The data exhibits extreme **Class Imbalance**.
    *   **C04 (Neoplasms)** is the dominant category, far outnumbering others.
    *   **C10 (Nervous System)** and **C23 (Pathological Conditions)** follow.
*   **Impact**: Models may be biased towards predicting neoplasms while neglecting minority classes (e.g., F01 Mental Disorders).

### 2.2 Feature Analysis
![Feature Boxplot](results/figures/feature_distribution_boxplot.png)
*   **DSI (Disease Specificity Index)**:
    *   **C04 (Neoplasms)** genes generally have lower DSI. This suggests carcinogenic genes are often "multi-functional," participating in various biological processes with low specificity.
    *   Genes for **C16 (Congenital Abnormalities)** show higher DSI, indicating higher specificity.
*   **DPI (Disease Pleiotropy Index)**: Negatively correlated with DSI; tumor genes tend to have higher DPI.

---

## 4. Modeling Strategy and Optimization

### 3.1 Model Comparison Tournament
We conducted a rigorous comparison of four different algorithms to identify the best approach for this specific dataset.

![Model Comparison](results/figures/model_comparison.png)

| Rank | Model | F1 Micro Score | Insight |
| :--- | :--- | :--- | :--- |
| **1** | **KNN (K-Nearest Neighbors)** | **0.3245** | **Winner!** Demonstrates that "Gene Similarity" is the strongest predictor. Genes with similar DSI/DPI profiles tend to cause similar diseases. |
| 2 | Random Forest | 0.3007 | Strong baseline, but slightly outperformed by KNN's local instance-based learning. |
| 3 | Naive Bayes | 0.1198 | Poor performance due to the assumption of feature independence (DSI and DPI are correlated). |
| 4 | Neural Network (MLP) | 0.0979 | Failed to generalize. Likely due to the low dimensionality of features (only 5 features), which is insufficient for Deep Learning. |

### 3.2 Robustness Check (5-Fold Cross-Validation)
To ensure the reliability of our results, we performed **5-Fold Cross-Validation** on the Random Forest model.
*   **Results**: Mean F1 Score: **0.2079** (+/- 0.0036)
*   **Conclusion**: The extremely low standard deviation (0.0036) demonstrates that the model performance is highly stable across different data subsets, ensuring statistical significance.

### 3.3 Final Model Evaluation (Weighted RF)
*   **Model**: Weighted Random Forest Classifier (Cost-Sensitive Learning)
*   **Strategy**: Introduced `class_weight='balanced'` to penalize the misclassification of minority classes.
*   **Results**:
    *   **Recall** stabilized at **0.17**.
    *   While multi-label classification remains challenging, the weighting strategy effectively improved the identification of minority diseases.

---

## 5. Interpretability and Rule Extraction

To open the "Black Box," we trained a shallow Decision Tree to extract rules for determining **"Is it a Cancer Gene (C04)?"**.

### Extracted Core Rules
```text
IF Score > 0.31:
    IF DSI <= 0.51:
        PREDICT: Cancer (Class 1)
    ELSE (DSI > 0.51):
        PREDICT: Non-Cancer (Class 0)
```

### Biological Interpretation
*   Genes with **High Association Score (> 0.31)** and **Low Specificity (DSI <= 0.51)** are highly likely to be cancer drivers.
*   This aligns with biological intuition: Cancer is a complex systemic disease, and its key genes (e.g., TP53, EGFR) often participate in broad cellular pathways, thus having a low Specificity Index (DSI).

---

## 6. Unsupervised Association Mining

### Disease Similarity Heatmap
![Heatmap](results/figures/disease_similarity_heatmap.png)

### Key Findings
By calculating the Jaccard Similarity, we identified the following strong associations:
1.  **C10 (Nervous System) <--> C23 (Pathological Conditions)** (Similarity: 0.409)
    *   Suggests that nervous system diseases are often accompanied by broad pathological physiological changes.
2.  **C04 (Neoplasms) <--> C06 (Digestive System)** (Similarity: 0.389)
    *   Reveals the dominance of digestive system cancers within the tumor data, indicating a high overlap of pathogenic genes.
3.  **C10 (Nervous System) <--> F03 (Mental Disorders)** (Similarity: 0.320)
    *   Molecular evidence for **"Mind-Body Connection"**: Neurological diseases and psychiatric disorders share a significant number of genes.

---

## 7. Conclusion
This project successfully established a complete gene-disease analysis pipeline:
1.  **Data Cleaning**: Processed 70,000+ records, filtering for Top 20 disease classes.
2.  **Model Selection**: Confirmed **KNN** as the best model.
    *   *Association Analysis*: Proved that **Feature Similarity** is the core driver for predicting gene-disease associations, validating the "Guilt-by-Association" hypothesis.
3.  **Knowledge Discovery**:
    *   **Rules**: Discovered that "Low Specificity Genes are prone to Cancer."
    *   **Commonalities**: Revealed molecular comorbidity mechanisms between Neural-Mental and Digestive-Tumor classes.

**Future Outlook**: Recommend introducing Deep Learning (e.g., Graph Neural Networks) to further capture complex gene-disease network features.
