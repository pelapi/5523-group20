import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score

# Define paths
PROCESSED_DATA_PATH = '../../data/processed/cleaned_disgenet.csv'
RESULTS_DIR = '../../results/'
FIGURES_DIR = '../../results/figures/'

def compare_models():
    print("Loading data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return

    # 1. Prepare Data
    feature_cols = ['DSI', 'DPI', 'score', 'EI', 'NofPmids']
    X = df[feature_cols].fillna(0)
    
    # Scale features (Important for KNN and MLP)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y_list = df['filtered_diseaseClass'].astype(str).str.split(',')
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_list)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    # 2. Define Models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, class_weight='balanced', random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        "Naive Bayes": OneVsRestClassifier(GaussianNB()) # NB doesn't support multi-label natively, so we wrap it
    }
    
    results = []
    
    print("\n=== Starting Model Comparison Tournament ===")
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        
        f1_micro = f1_score(Y_test, Y_pred, average='micro')
        f1_macro = f1_score(Y_test, Y_pred, average='macro')
        acc = accuracy_score(Y_test, Y_pred)
        
        print(f"  -> F1 Micro: {f1_micro:.4f}")
        results.append({
            "Model": name,
            "F1 Micro": f1_micro,
            "F1 Macro": f1_macro,
            "Accuracy": acc
        })

    # 3. Visualize Results
    results_df = pd.DataFrame(results)
    print("\n--- Final Leaderboard ---")
    print(results_df)
    
    # Save results table
    results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_results.csv'), index=False)
    
    # Plot Comparison
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y="F1 Micro", data=results_df, palette="magma")
    plt.title("Model Comparison: F1 Score (Micro)", fontsize=16)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 0.4) # Adjust based on expected range
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, 'model_comparison.png')
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    compare_models()
