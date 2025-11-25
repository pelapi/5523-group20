import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score

# Define paths
PROCESSED_DATA_PATH = '../../data/processed/cleaned_disgenet.csv'
MODEL_OUTPUT_PATH = '../../results/models/'
METRICS_OUTPUT_PATH = '../../results/metrics.txt'

def train_model():
    print("Loading data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return

    # 1. Prepare Features (X)
    # We use the numerical features available
    feature_cols = ['DSI', 'DPI', 'score', 'EI', 'NofPmids']
    X = df[feature_cols].fillna(0) # Fill any remaining NaNs with 0 just in case

    # 2. Prepare Target (Y)
    # The target is 'filtered_diseaseClass', which is a string like "C04,C06"
    # We need to split it into a list for MultiLabelBinarizer
    y_list = df['filtered_diseaseClass'].astype(str).str.split(',')
    
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)
    classes = mlb.classes_
    print(f"Target Classes ({len(classes)}): {classes}")

    # 3. Split Data
    print("Splitting data into Train and Test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4. Train Model
    print("Training Random Forest Classifier (this may take a moment)...")
    # Using a reasonable number of estimators and depth to prevent overfitting and ensure speed
    clf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    clf.fit(X_train, Y_train)

    # 5. Evaluate
    print("Evaluating model...")
    Y_pred = clf.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(Y_test, Y_pred)
    f1_micro = f1_score(Y_test, Y_pred, average='micro')
    f1_macro = f1_score(Y_test, Y_pred, average='macro')
    
    print(f"\n--- Model Performance ---")
    print(f"Accuracy (Exact Match): {accuracy:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    
    report = classification_report(Y_test, Y_pred, target_names=classes, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # 6. Save Model and Metrics
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_OUTPUT_PATH, 'rf_model.pkl')
    mlb_path = os.path.join(MODEL_OUTPUT_PATH, 'mlb.pkl')
    
    joblib.dump(clf, model_path)
    joblib.dump(mlb, mlb_path)
    
    with open(METRICS_OUTPUT_PATH, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1 Micro: {f1_micro:.4f}\n")
        f.write(f"F1 Macro: {f1_macro:.4f}\n\n")
        f.write(report)
        
    print(f"Model saved to {model_path}")
    print(f"Metrics saved to {METRICS_OUTPUT_PATH}")

    # 7. Feature Importance
    importances = clf.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    print("\n--- Feature Importance ---")
    print(feature_imp_df)

if __name__ == "__main__":
    train_model()
