import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths
PROCESSED_DATA_PATH = '../../data/processed/cleaned_disgenet.csv'
FIGURES_DIR = '../../results/figures/'

def plot_visualizations():
    print("Loading processed data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return

    # Ensure output directory exists
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Set style
    sns.set_theme(style="whitegrid")

    # --- 1. Disease Class Distribution ---
    print("Generating Class Distribution Plot...")
    plt.figure(figsize=(12, 6))
    
    # Explode the classes because one row can have multiple classes (e.g., "C04,C20")
    # We want to count each occurrence.
    df_exploded = df.assign(diseaseClass=df['filtered_diseaseClass'].str.split(',')).explode('diseaseClass')
    
    # Count top classes
    class_counts = df_exploded['diseaseClass'].value_counts().head(15)
    
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", legend=False)
    plt.title('Top 15 Disease Classes Distribution', fontsize=16)
    plt.xlabel('Disease Class (MeSH Code)', fontsize=12)
    plt.ylabel('Number of Associations', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'))
    plt.close()

    # --- 2. DSI and DPI Distribution by Class ---
    print("Generating Feature Boxplots...")
    # We focus on the top 5 classes to keep the plot readable
    top_5_classes = class_counts.head(5).index.tolist()
    df_top5 = df_exploded[df_exploded['diseaseClass'].isin(top_5_classes)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # DSI Boxplot
    sns.boxplot(ax=axes[0], data=df_top5, x='diseaseClass', y='DSI', hue='diseaseClass', palette="Set2", legend=False)
    axes[0].set_title('Gene Specificity (DSI) by Disease Class', fontsize=14)
    axes[0].set_ylabel('DSI (Lower = More Specific)', fontsize=12)

    # DPI Boxplot
    sns.boxplot(ax=axes[1], data=df_top5, x='diseaseClass', y='DPI', hue='diseaseClass', palette="Set2", legend=False)
    axes[1].set_title('Gene Pleiotropy (DPI) by Disease Class', fontsize=14)
    axes[1].set_ylabel('DPI (Higher = More Pleiotropic)', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'feature_distribution_boxplot.png'))
    plt.close()

    # --- 3. Score Distribution ---
    print("Generating Score Distribution Plot...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['score'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Gene-Disease Association Scores', fontsize=16)
    plt.xlabel('Score (Confidence)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'score_distribution.png'))
    plt.close()

    print(f"All figures saved to: {os.path.abspath(FIGURES_DIR)}")

if __name__ == "__main__":
    plot_visualizations()
