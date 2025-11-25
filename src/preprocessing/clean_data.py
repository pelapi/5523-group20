import pandas as pd
import os

# Define paths
RAW_DATA_PATH = '../../data/raw/curated_Disgenet.csv'
PROCESSED_DATA_PATH = '../../data/processed/cleaned_disgenet.csv'

def clean_data():
    print("Loading raw data...")
    try:
        df = pd.read_csv(RAW_DATA_PATH, sep=';')
    except FileNotFoundError:
        print(f"Error: File not found at {RAW_DATA_PATH}")
        return

    print(f"Original shape: {df.shape}")

    # 1. Select relevant columns for prediction
    # We keep gene features and disease class labels
    cols_to_keep = ['geneSymbol', 'DSI', 'DPI', 'score', 'EI', 'NofPmids', 'diseaseClass', 'diseaseType']
    df = df[cols_to_keep]

    # 2. Drop rows with missing critical values
    # diseaseClass is our target, so we can't have it missing
    df = df.dropna(subset=['diseaseClass', 'DSI', 'DPI'])
    print(f"Shape after dropping NaNs: {df.shape}")

    # 3. Handle Disease Classes (Target Engineering)
    # The 'diseaseClass' column contains comma-separated values (e.g., "C04,C20").
    # For a robust "All Disease" prediction, we should treat this as a Multi-Label problem.
    # However, for initial cleaning, we will just ensure the format is consistent.
    
    # Let's filter out classes that are too rare (e.g., appear less than 50 times) to reduce noise
    # First, let's see the distribution of individual classes
    all_classes = df['diseaseClass'].str.split(',').explode()
    class_counts = all_classes.value_counts()
    
    # Keep top 20 classes (covering the vast majority of diseases)
    top_classes = class_counts.head(20).index.tolist()
    print(f"Top 20 Disease Classes to focus on: {top_classes}")

    # Function to filter classes in the row
    def filter_top_classes(class_str):
        classes = class_str.split(',')
        valid_classes = [c for c in classes if c in top_classes]
        return ','.join(valid_classes) if valid_classes else None

    df['filtered_diseaseClass'] = df['diseaseClass'].apply(filter_top_classes)
    df = df.dropna(subset=['filtered_diseaseClass'])
    
    print(f"Shape after filtering for top 20 classes: {df.shape}")

    # 4. Save processed data
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Cleaned data saved to {PROCESSED_DATA_PATH}")

if __name__ == "__main__":
    clean_data()
