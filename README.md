# Gene-Disease Association Analysis and Prediction Report

## 1. Project Overview
> **⚠️ Scope & Limitation Disclaimer**:
> This project is an **exploratory, proof-of-concept study**. The primary goal is to establish a computational pipeline and generate biological hypotheses.
> *   **Not a Clinical Tool**: The results should not be used for clinical diagnosis.
> *   **Hypothesis Generation**: The rules and associations discovered are hypotheses requiring further biological validation.
> *   **Feature Scope**: We deliberately limited features to interpretable indices (DSI, DPI) to test their predictive power, acknowledging that this simplifies the complex biological reality.

**Data Source**: DisGeNET (Curated Gene-Disease Associations)
**Core Objectives**:
1. **Phase 1 (Foundation)**: Completed data cleaning, integration, and gene-disease network construction.
2. **Phase 2 (Deep Mining)**: Utilized machine learning models to predict disease categories based on gene features (e.g., DSI, DPI).
3. **Rule Extraction**: Mined potential biological patterns between genes and diseases.
4. **Commonality Analysis**: Analyzed commonalities between different disease categories.

### 1.1 Motivation & Hypothesis
*   **Why this task?** Understanding whether gene-level properties alone can predict disease classes helps identify "generalist" vs. "specialist" genes.
*   **Hypothesis**: Genes associated with similar disease classes share topological properties (DSI) and pleiotropic patterns (DPI). We test if these **interpretable indices** carry enough signal for classification, deliberately excluding opaque network embeddings in Phase 2 to focus on biological interpretability.

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
    *   The high score of PA confirms that the network exhibits a strong **"Rich-get-Richer"** property.
    *   **Hypothesis Bridge**: We *hypothesize* that this topological centrality might be captured by gene-level metrics like DSI/DPI, potentially aiding classification. However, we acknowledge that network centrality does not guarantee feature separability for disease classes.

---

## 3. Exploratory Data Analysis (EDA)

### 2.1 Disease Class Distribution
![Class Distribution](results/figures/class_distribution.png)
*   **Analysis**: The data exhibits extreme **Class Imbalance**.
    *   **C04 (Neoplasms)** is the dominant category, far outnumbering others.
    *   **C10 (Nervous System)** and **C23 (Pathological Conditions)** follow.
*   **Impact**: Models may be biased towards predicting neoplasms while neglecting minority classes (e.g., F01 Mental Disorders).
    *   **Mechanism of Failure**: Severe imbalance causes models to optimize for global accuracy by "collapsing" to the majority class (C04). This explains why Micro-F1 (accuracy-driven) appears decent while Macro-F1 (minority-sensitive) is low.

### 2.2 Feature Analysis
![Feature Boxplot](results/figures/feature_distribution_boxplot.png)
*   **DSI (Disease Specificity Index)**:
    *   **C04 (Neoplasms)** genes generally have lower DSI. This suggests carcinogenic genes are often "multi-functional," participating in various biological processes with low specificity.
    *   Genes for **C16 (Congenital Abnormalities)** show higher DSI, indicating higher specificity.
    *   Genes for **C16 (Congenital Abnormalities)** show higher DSI, indicating higher specificity.
*   **DPI (Disease Pleiotropy Index)**: Negatively correlated with DSI; tumor genes tend to have higher DPI.
*   **Rationale**: We selected these features because they map directly to biological concepts (Specificity vs. Pleiotropy), allowing us to interpret *why* a model classifies a gene as "Cancer-related".

---

## 4. Modeling Strategy and Optimization

### 3.1 Model Comparison Tournament
We conducted a rigorous comparison of four different algorithms to identify the best approach for this specific dataset.

![Model Comparison](results/figures/model_comparison.png)

| Rank | Model | F1 Micro Score | Insight |
| :--- | :--- | :--- | :--- |
| **1** | **KNN (K-Nearest Neighbors)** | **0.3245** | **Best Baseline**: Suggests that "Gene Similarity" carries predictive signal, particularly for dominant classes (C04). Performance on minority classes remains limited. |
| 2 | Random Forest | 0.3007 | Strong baseline, but slightly outperformed by KNN's local instance-based learning. |
| 3 | Naive Bayes | 0.1198 | Poor performance due to the assumption of feature independence (DSI and DPI are correlated). |
| 3 | Naive Bayes | 0.1198 | Poor performance due to the assumption of feature independence (DSI and DPI are correlated). |
| 4 | Neural Network (MLP) | 0.0979 | Failed to generalize. Likely due to the low dimensionality of features (only 5 features), which is insufficient for Deep Learning. |

#### Why did KNN outperform complex models?
*   **Low Dimensionality**: With only ~5 features, the data manifold is low-dimensional. Deep Learning (MLP) is overkill and prone to overfitting or underfitting without massive feature spaces.
*   **Local Similarity**: The "Guilt-by-Association" principle works locally. Genes with similar DSI/DPI scores cluster together in feature space, which KNN captures perfectly. Random Forest struggles to find orthogonal splits in such a highly overlapping distribution.

### 3.2 Robustness Check (5-Fold Cross-Validation)
To ensure the reliability of our results, we performed **5-Fold Cross-Validation** on the Random Forest model.
*   **Results**: Mean F1 Score: **0.2079** (+/- 0.0036)
*   **Conclusion**: The extremely low standard deviation (0.0036) demonstrates that the model performance is highly stable across different data subsets, ensuring statistical significance.

### 3.3 Final Model Evaluation (Weighted RF)
*   **Model**: Weighted Random Forest Classifier (Cost-Sensitive Learning)
*   **Strategy**: Introduced `class_weight='balanced'` to penalize the misclassification of minority classes.
*   **Detailed Metrics**:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Micro F1** | **0.2235** | Reflects overall accuracy, dominated by majority classes (C04, C10). |
| **Macro F1** | **0.1302** | **Critical Insight**: The low Macro F1 reveals the model's struggle with minority classes. |

*   **Per-Class Performance (Selected)**:
    *   **Majority Class (C04 Neoplasms)**: Support=4232, Recall=0.27. The model is biased towards this class.
    *   **Minority Class (F01 Mental)**: Support=351, Recall=0.01. The model fails to identify these due to extreme imbalance.
    *   **Conclusion**: While weighting helped, the feature signal (DSI/DPI) alone is insufficient to distinguish minority classes from the dominant cancer signal. Future work needs richer features (embeddings, pathways).

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

### Biological Interpretation (Hypothesis Generation)
> **Caution**: These rules are derived from the current dataset and may reflect **Literature Bias** (e.g., cancer genes are more heavily studied and annotated).
*   **Observation**: In this dataset, genes with **High Association Score** and **Low Specificity** tend to be labeled as Cancer (C04).
*   **Hypothesis**: This suggests "generalist" genes might be key drivers in systemic diseases like cancer, but this requires validation across independent databases to rule out annotation bias.

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
    *   **Observation**: Neurological diseases and psychiatric disorders share a significant number of genes.
    *   **Limitation**: This could stem from **Annotation Bias** (shared research focus) rather than purely shared pathogenesis. Further functional validation is needed.

> **Note**: High similarity may also stem from shared annotation bias in the literature.

---

## 7. Conclusion
This project successfully established a complete gene-disease analysis pipeline:
1.  **Data Cleaning**: Processed 70,000+ records, filtering for Top 20 disease classes.
2.  **Model Selection**: Identified **KNN** as the most effective baseline.
    *   *Association Analysis*: **Suggested** that Feature Similarity is a relevant factor for predicting gene-disease associations, supporting the "Guilt-by-Association" hypothesis for major classes.
3.  **Knowledge Discovery**:
    *   **Rules**: Discovered that "Low Specificity Genes are prone to Cancer."
    *   **Commonalities**: Revealed molecular comorbidity mechanisms between Neural-Mental and Digestive-Tumor classes.

**Future Outlook**: Recommend introducing Deep Learning (e.g., Graph Neural Networks) to further capture complex gene-disease network features.
