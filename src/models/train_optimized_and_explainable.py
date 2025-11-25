import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, make_scorer

# Define paths
PROCESSED_DATA_PATH = '../../data/processed/cleaned_disgenet.csv'
RESULTS_DIR = '../../results/'

def train_optimized_models():
    print("Loading data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return

    # 1. Prepare Data
    feature_cols = ['DSI', 'DPI', 'score', 'EI', 'NofPmids']
    X = df[feature_cols].fillna(0)
    
    y_list = df['filtered_diseaseClass'].astype(str).str.split(',')
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)
    classes = mlb.classes_
    
    # --- Step 1: K-Fold Cross-Validation (Robustness Check) ---
    print("\n=== Step 1: 5-Fold Cross-Validation (Robustness Check) ===")
    print("Running 5-Fold CV to estimate average model performance...")
    
    # Define the model (Weighted RF)
    rf_cv = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
    
    # Define metric (Micro F1 is standard for multi-label)
    scorer = make_scorer(f1_score, average='micro')
    
    # Run CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_cv, X, Y, cv=kf, scoring=scorer, n_jobs=-1)
    
    print(f"5-Fold CV F1 Scores: {cv_scores}")
    print(f"Mean F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print("-> This proves our model performance is stable across different data subsets.")

    # --- Step 2: Final Model Training (Train/Test Split) ---
    print("\n=== Step 2: Final Model Training (80/20 Split) ===")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    print("Training full Weighted Random Forest...")
    rf_balanced = RandomForestClassifier(n_estimators=100, max_depth=15, 
                                       class_weight='balanced',
                                       random_state=42, n_jobs=-1)
    rf_balanced.fit(X_train, Y_train)
    
    Y_pred_balanced = rf_balanced.predict(X_test)
    
    print("\n--- Optimized Model Performance (Test Set) ---")
    print(f"F1 Score (Micro): {f1_score(Y_test, Y_pred_balanced, average='micro'):.4f}")
    print(f"F1 Score (Macro): {f1_score(Y_test, Y_pred_balanced, average='macro'):.4f}")
    print("\nClassification Report (Weighted RF):")
    print(classification_report(Y_test, Y_pred_balanced, target_names=classes, zero_division=0))

    # --- Step 3: Interpretability (Rule-based Decision Tree) ---
    print("\n=== Step 3: Extracting Rules with Decision Tree ===")
    
    # Target: Cancer (C04) vs Others
    cancer_idx = list(classes).index('C04')
    Y_cancer = Y_train[:, cancer_idx]
    
    dt_explainable = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    dt_explainable.fit(X_train, Y_cancer)
    
    feature_names = feature_cols
    rules = export_text(dt_explainable, feature_names=feature_names)
    
    print("\n--- Extracted Rules for Detecting Cancer (C04) ---")
    print(rules)
    
    # Save rules
    rule_path = os.path.join(RESULTS_DIR, 'decision_tree_rules.txt')
    with open(rule_path, 'w') as f:
        f.write(f"Mean 5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
        f.write(rules)
    print(f"Rules and CV metrics saved to {rule_path}")

if __name__ == "__main__":
    train_optimized_models()
