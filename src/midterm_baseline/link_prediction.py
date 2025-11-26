#!/usr/bin/env python3
import csv
import os
import sys
import random
import math
import json
from typing import List, Tuple, Dict, Set
import argparse

def check_and_install_packages():
    required_packages = {
        'networkx': 'networkx',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'numpy': 'numpy'
    }
    missing_packages = []
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Run: pip3 install --break-system-packages " + " ".join(missing_packages))
        return False
    return True

if not check_and_install_packages():
    sys.exit(1)

import networkx as nx
import numpy as np

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False



class DataCleaner:
    def __init__(self, csv_file: str):
        self.csv_file = csv_file
        self.data: List[Dict] = []

    def load_data(self):
        print(f"Loading data: {self.csv_file}")
        with open(self.csv_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                self.data.append(dict(row))
        print(f"Loaded {len(self.data)} raw records")
        return self

    def clean_data(self, min_score: float = 0.0):
        print("Cleaning data...")
        cleaned_data: List[Dict] = []
        for row in self.data:
            if not (row.get('geneId') and row.get('diseaseId') and row.get('score')):
                continue
            try:
                score = float(row['score'])
            except (ValueError, TypeError):
                continue
            if score >= min_score:
                cleaned_data.append({
                    'gene': f"gene_{row['geneId']}",
                    'disease': f"disease_{row['diseaseId']}",
                    'score': score,
                    'gene_symbol': row.get('geneSymbol', ''),
                    'disease_name': row.get('diseaseName', '')
                })
        seen = set()
        unique_data: List[Dict] = []
        for item in cleaned_data:
            key = (item['gene'], item['disease'])
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        self.data = unique_data
        print(f"Kept {len(self.data)} records after cleaning")
        return self

    def get_statistics(self):
        genes = set(item['gene'] for item in self.data)
        diseases = set(item['disease'] for item in self.data)
        return {
            'total_records': len(self.data),
            'unique_genes': len(genes),
            'unique_diseases': len(diseases),
            'avg_score': float(np.mean([item['score'] for item in self.data]))
        }


class BipartiteGraph:
    def __init__(self, data: List[Dict]):
        self.data = data
        self.graph: nx.Graph = nx.Graph()
        self.genes: Set[str] = set()
        self.diseases: Set[str] = set()

    def build_graph(self):
        print("Building bipartite graph...")
        for item in self.data:
            gene = item['gene']
            disease = item['disease']
            score = item['score']
            self.genes.add(gene)
            self.diseases.add(disease)
            self.graph.add_edge(gene, disease, weight=score)
        print(f"Graph built with {len(self.genes)} genes and {len(self.diseases)} diseases")
        print(f"Total edges: {self.graph.number_of_edges()}")
        return self

    def get_degree_distribution(self):
        gene_degrees = [self.graph.degree(gene) for gene in self.genes]
        disease_degrees = [self.graph.degree(disease) for disease in self.diseases]
        return {
            'gene_degrees': gene_degrees,
            'disease_degrees': disease_degrees
        }


class EdgeSplitter:
    def __init__(self, graph: BipartiteGraph, test_ratio: float = 0.2, val_ratio: float = 0.1):
        self.graph = graph
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.train_edges: List[Tuple[str, str]] = []
        self.val_edges: List[Tuple[str, str]] = []
        self.test_edges: List[Tuple[str, str]] = []
        self.negative_edges: List[Tuple[str, str]] = []

    def split_edges(self, random_seed: int = 42):
        print("Splitting edges...")
        random.seed(random_seed)
        np.random.seed(random_seed)
        edges = list(self.graph.graph.edges())
        random.shuffle(edges)
        n_test = int(len(edges) * self.test_ratio)
        n_val = int(len(edges) * self.val_ratio)
        self.test_edges = edges[:n_test]
        self.val_edges = edges[n_test:n_test + n_val]
        self.train_edges = edges[n_test + n_val:]
        print(f"Train: {len(self.train_edges)} edges")
        print(f"Validation: {len(self.val_edges)} edges")
        print(f"Test: {len(self.test_edges)} edges")
        return self

    def generate_negative_samples(self, method: str = 'random', ratio: float = 1.0):
        print(f"Generating negative samples (method: {method})...")
        positive_edges = set(self.graph.graph.edges())
        n_negative = int(len(self.test_edges) * ratio)
        negative_edges: List[Tuple[str, str]] = []
        genes = list(self.graph.genes)
        diseases = list(self.graph.diseases)
        if method == 'random':
            while len(negative_edges) < n_negative:
                gene = random.choice(genes)
                disease = random.choice(diseases)
                if (gene, disease) not in positive_edges and (disease, gene) not in positive_edges:
                    negative_edges.append((gene, disease))
        elif method == 'degree_based':
            gene_degrees = [(gene, self.graph.graph.degree(gene)) for gene in genes]
            disease_degrees = [(disease, self.graph.graph.degree(disease)) for disease in diseases]
            gene_degrees.sort(key=lambda x: x[1], reverse=True)
            disease_degrees.sort(key=lambda x: x[1], reverse=True)
            top_genes = [g for g, _ in gene_degrees[:len(gene_degrees)//2]]
            top_diseases = [d for d, _ in disease_degrees[:len(disease_degrees)//2]]
            while len(negative_edges) < n_negative:
                gene = random.choice(top_genes)
                disease = random.choice(top_diseases)
                if (gene, disease) not in positive_edges and (disease, gene) not in positive_edges:
                    negative_edges.append((gene, disease))
        self.negative_edges = negative_edges
        print(f"Generated {len(negative_edges)} negative samples")
        return self


class BipartiteGraphBaselines:
    def __init__(self, graph: nx.Graph, genes: Set[str], diseases: Set[str]):
        self.graph = graph
        self.genes = genes
        self.diseases = diseases

    def preferential_attachment(self, gene: str, disease: str) -> float:
        gene_degree = self.graph.degree(gene)
        disease_degree = self.graph.degree(disease)
        return float(gene_degree * disease_degree)

    def predict_links(self, test_edges: List[Tuple], negative_edges: List[Tuple], method: str = 'preferential_attachment'):
        print(f"Predicting links using {method}...")
        if method == 'preferential_attachment':
            score_func = self.preferential_attachment
        else:
            raise ValueError(f"Unknown method: {method}")
        
        positive_scores = [score_func(edge[0], edge[1]) for edge in test_edges]
        negative_scores = [score_func(edge[0], edge[1]) for edge in negative_edges]
        
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_scores = positive_scores + negative_scores
        return y_true, y_scores


class SVDEmbedding:
    def __init__(self, graph: nx.Graph, dimensions: int = 64):
        self.graph = graph
        self.dimensions = dimensions
        self.embeddings: Dict[str, np.ndarray] = {}
        self.node_to_idx = {}
        self.idx_to_node = {}

    def train_embeddings(self):
        print("Training SVD embeddings (Matrix Factorization)...")
        nodes = list(self.graph.nodes())
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        # Create Adjacency Matrix
        adj_matrix = nx.to_scipy_sparse_array(self.graph, nodelist=nodes, format='csr')
        
        # Perform SVD
        svd = TruncatedSVD(n_components=self.dimensions, random_state=42)
        node_factors = svd.fit_transform(adj_matrix)
        
        for i, node in enumerate(nodes):
            self.embeddings[node] = node_factors[i]
            
        print(f"Generated SVD embeddings for {len(self.embeddings)} nodes")
        return self

    def get_edge_score(self, node1: str, node2: str) -> float:
        if node1 in self.embeddings and node2 in self.embeddings:
            # Dot product similarity
            return float(np.dot(self.embeddings[node1], self.embeddings[node2]))
        return 0.0

    def predict(self, test_edges: List[Tuple], negative_edges: List[Tuple]):
        print("Predicting links using SVD Dot Product...")
        positive_scores = [self.get_edge_score(e[0], e[1]) for e in test_edges]
        negative_scores = [self.get_edge_score(e[0], e[1]) for e in negative_edges]
        
        y_true = [1] * len(positive_scores) + [0] * len(negative_scores)
        y_scores = positive_scores + negative_scores
        return y_true, y_scores


class Evaluator:
    def __init__(self):
        self.results: Dict[str, Dict] = {}

    def evaluate(self, y_true: List, y_scores: List, method_name: str):
        print(f"Evaluating {method_name}...")
        try:
            auc_score = float(roc_auc_score(y_true, y_scores))
        except Exception:
            auc_score = 0.0
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = float(average_precision_score(y_true, y_scores))
        self.results[method_name] = {
            'auc': auc_score,
            'pr_auc': pr_auc,
            'precision': precision,
            'recall': recall,
            'y_true': y_true,
            'y_scores': y_scores
        }
        print(f"{method_name} - AUC: {auc_score:.4f}, PR-AUC: {pr_auc:.4f}")
        return self

    def plot_pr_curves(self, save_path: str = 'pr_curves.png'):
        plt.figure(figsize=(10, 8))
        for method_name, result in self.results.items():
            plt.plot(result['recall'], result['precision'], label=f"{method_name} (AUC={result['pr_auc']:.3f})", linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved PR curves to: {save_path}")

    def plot_degree_distribution(self, bipartite_graph: BipartiteGraph, save_path: str = 'degree_distribution.png'):
        degree_dist = bipartite_graph.get_degree_distribution()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.hist(degree_dist['gene_degrees'], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Degree')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Gene Degree Distribution')
        ax1.grid(True, alpha=0.3)
        ax2.hist(degree_dist['disease_degrees'], bins=50, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('Degree')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Disease Degree Distribution')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved degree distribution to: {save_path}")

    def generate_results_table(self):
        lines = []
        lines.append("| Method | AUC | PR-AUC |")
        lines.append("|--------|-----|--------|")
        for method_name, result in self.results.items():
            lines.append(f"| {method_name} | {result['auc']:.4f} | {result['pr_auc']:.4f} |")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Gene–Disease bipartite link prediction')
    parser.add_argument('--data', required=True, help='Input CSV file path')
    parser.add_argument('--min_score', type=float, default=0.3, help='Minimum score threshold')
    parser.add_argument('--output_dir', default='results', help='Output directory')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("=== Gene–Disease Bipartite Link Prediction (English) ===")
    cleaner = DataCleaner(args.data)
    cleaner.load_data().clean_data(min_score=args.min_score)
    stats = cleaner.get_statistics()
    bipartite_graph = BipartiteGraph(cleaner.data)
    bipartite_graph.build_graph()
    splitter = EdgeSplitter(bipartite_graph)
    splitter.split_edges(random_seed=args.random_seed)
    splitter.generate_negative_samples(method='random')
    evaluator = Evaluator()
    baselines = BipartiteGraphBaselines(bipartite_graph.graph, bipartite_graph.genes, bipartite_graph.diseases)
    methods = ['preferential_attachment']
    for method in methods:
        y_true, y_scores = baselines.predict_links(splitter.test_edges, splitter.negative_edges, method)
        evaluator.evaluate(y_true, y_scores, method.replace('_', ' ').title())
    
    # SVD
    svd_model = SVDEmbedding(bipartite_graph.graph)
    svd_model.train_embeddings()
    y_true, y_scores = svd_model.predict(splitter.test_edges, splitter.negative_edges)
    evaluator.evaluate(y_true, y_scores, 'SVD (Matrix Factorization)')
    evaluator.plot_pr_curves(os.path.join(args.output_dir, 'pr_curves.png'))
    evaluator.plot_degree_distribution(bipartite_graph, os.path.join(args.output_dir, 'degree_distribution.png'))
    results = {
        'dataset_stats': stats,
        **{method: {'auc': result['auc'], 'pr_auc': result['pr_auc']} for method, result in evaluator.results.items()}
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    table = evaluator.generate_results_table()
    with open(os.path.join(args.output_dir, 'results_table.txt'), 'w') as f:
        f.write("Gene–Disease Link Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(table + "\n\n")
        f.write("Dataset Stats:\n")
        f.write(f"- Total records: {stats['total_records']}\n")
        f.write(f"- Genes: {stats['unique_genes']}\n")
        f.write(f"- Diseases: {stats['unique_diseases']}\n")
        f.write(f"- Test edges: {len(splitter.test_edges)}\n")
    print("\n=== Experiment finished ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()