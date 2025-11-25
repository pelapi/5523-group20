import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from itertools import combinations

# Define paths
PROCESSED_DATA_PATH = '../../data/processed/cleaned_disgenet.csv'
FIGURES_DIR = '../../results/figures/'

def plot_disease_associations():
    print("Loading data...")
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {PROCESSED_DATA_PATH}")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Task: Find Shared Genes between Disease Classes ---
    # We want to see which disease classes share the most genes.
    # This is a proxy for "Comorbidity" or "Shared Molecular Mechanism".

    print("Processing disease-gene matrix...")
    
    # 1. Explode disease classes (one row per class)
    df_exploded = df.assign(diseaseClass=df['filtered_diseaseClass'].str.split(',')).explode('diseaseClass')
    
    # 2. Get unique genes for each class
    # Group by Class and get a set of genes
    class_genes = df_exploded.groupby('diseaseClass')['geneSymbol'].apply(set)
    
    # Filter to top 10 classes to keep the heatmap readable
    top_classes = df_exploded['diseaseClass'].value_counts().head(10).index.tolist()
    class_genes = class_genes[top_classes]
    
    print(f"Analyzing shared genes for top 10 classes: {top_classes}")

    # 3. Calculate Jaccard Similarity Matrix
    n = len(top_classes)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            class_a = top_classes[i]
            class_b = top_classes[j]
            
            genes_a = class_genes[class_a]
            genes_b = class_genes[class_b]
            
            # Jaccard Index = Intersection / Union
            intersection = len(genes_a.intersection(genes_b))
            union = len(genes_a.union(genes_b))
            
            jaccard = intersection / union if union > 0 else 0
            similarity_matrix[i, j] = jaccard

    # 4. Plot Heatmap
    print("Generating Heatmap...")
    plt.figure(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(similarity_matrix, dtype=bool)) # Mask upper triangle to avoid redundancy
    
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                xticklabels=top_classes, yticklabels=top_classes,
                mask=mask, square=True, linewidths=.5)
    
    plt.title('Genetic Similarity between Disease Classes (Jaccard Index)', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, 'disease_similarity_heatmap.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Heatmap saved to {output_path}")
    
    # 5. Print Top Associations for PPT
    print("\n--- Top Shared Mechanisms (Highest Jaccard Similarity) ---")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((similarity_matrix[i, j], top_classes[i], top_classes[j]))
    
    # Sort by similarity
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    for score, c1, c2 in pairs[:5]:
        print(f"Similarity {score:.3f}: {c1} <--> {c2}")

if __name__ == "__main__":
    plot_disease_associations()
