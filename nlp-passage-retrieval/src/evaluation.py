import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def precision_at_k(relevant_items, retrieved_items, k):
    """
    Calculate Precision@k.
    
    Args:
        relevant_items (list): List of relevant item IDs
        retrieved_items (list): List of retrieved item IDs
        k (int): Cutoff for precision calculation
        
    Returns:
        float: Precision@k score
    """
    retrieved_k = retrieved_items[:k]
    relevant_retrieved = set(retrieved_k).intersection(set(relevant_items))
    return len(relevant_retrieved) / k if k > 0 else 0.0

def mean_reciprocal_rank(relevant_items, retrieved_items):
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        relevant_items (list): List of relevant item IDs
        retrieved_items (list): List of retrieved item IDs
        
    Returns:
        float: MRR score
    """
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(relevant_items, retrieved_items, k):
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        relevant_items (list): List of relevant item IDs
        retrieved_items (list): List of retrieved item IDs
        k (int): Cutoff for NDCG calculation
        
    Returns:
        float: NDCG@k score
    """
    retrieved_k = retrieved_items[:k]
    
    # Calculate DCG
    dcg = 0
    for i, item in enumerate(retrieved_k):
        if item in relevant_items:
            # Using binary relevance (1 if relevant, 0 if not)
            dcg += 1 / np.log2(i + 2)  # +2 because i is 0-indexed
    
    # Calculate ideal DCG (IDCG)
    idcg = 0
    for i in range(min(k, len(relevant_items))):
        idcg += 1 / np.log2(i + 2)
    
    # Calculate NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg

def evaluate_retrieval(tfidf_results, semantic_results, output_dir='results'):
    """
    Evaluate TF-IDF retrieval against semantic search (ground truth).
    
    Args:
        tfidf_results (dict): Dictionary mapping queries to TF-IDF search results
        semantic_results (dict): Dictionary mapping queries to semantic search results
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    evaluation_results = {}
    
    for query in tfidf_results.keys():
        # Get chunk IDs from both methods
        tfidf_chunk_ids = [result['chunk_id'] for result in tfidf_results[query]]
        semantic_chunk_ids = [result['chunk_id'] for result in semantic_results[query]]
        
        # Use semantic search results as ground truth
        p_at_5 = precision_at_k(semantic_chunk_ids, tfidf_chunk_ids, 5)
        mrr = mean_reciprocal_rank(semantic_chunk_ids, tfidf_chunk_ids)
        ndcg = ndcg_at_k(semantic_chunk_ids, tfidf_chunk_ids, 5)
        
        evaluation_results[query] = {
            'precision_at_5': p_at_5,
            'mrr': mrr,
            'ndcg_at_5': ndcg
        }
        
        print(f"\nEvaluation for query: {query}")
        print(f"Precision@5: {p_at_5:.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"NDCG@5: {ndcg:.4f}")
    
    # Calculate average metrics
    avg_p_at_5 = np.mean([res['precision_at_5'] for res in evaluation_results.values()])
    avg_mrr = np.mean([res['mrr'] for res in evaluation_results.values()])
    avg_ndcg = np.mean([res['ndcg_at_5'] for res in evaluation_results.values()])
    
    print("\nAverage Metrics:")
    print(f"Average Precision@5: {avg_p_at_5:.4f}")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average NDCG@5: {avg_ndcg:.4f}")
    
    # Save evaluation results
    evaluation_results['average'] = {
        'precision_at_5': float(avg_p_at_5),
        'mrr': float(avg_mrr),
        'ndcg_at_5': float(avg_ndcg)
    }
    
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # Visualize results
    visualize_evaluation(evaluation_results, output_dir)
    
    return evaluation_results

def visualize_evaluation(evaluation_results, output_dir='results'):
    """
    Visualize evaluation metrics.
    
    Args:
        evaluation_results (dict): Dictionary with evaluation metrics
        output_dir (str): Directory to save visualizations
    """
    # Extract queries and metrics (excluding the 'average' entry)
    queries = [q for q in evaluation_results.keys() if q != 'average']
    metrics = ['precision_at_5', 'mrr', 'ndcg_at_5']
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    x = np.arange(len(queries))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [evaluation_results[query][metric] for query in queries]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Queries')
    plt.ylabel('Score')
    plt.title('Evaluation Metrics for TF-IDF vs. miniLM')
    plt.xticks(x + width, [f'Query {i+1}' for i in range(len(queries))])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_metrics.png'))
    plt.close()
    
    # Create radar chart for average metrics
    avg_metrics = evaluation_results['average']
    
    # Number of variables
    categories = list(avg_metrics.keys())
    N = len(categories)
    
    # Create angles for each metric
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Add the metric values
    values = list(avg_metrics.values())
    values += values[:1]  # Close the loop
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', label='TF-IDF vs. miniLM')
    ax.fill(angles, values, alpha=0.25)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Set y-axis limits
    plt.ylim(0, 1)
    
    plt.title('Average Evaluation Metrics')
    plt.savefig(os.path.join(output_dir, 'average_metrics_radar.png'))
    plt.close()

def analyze_results_overlap(tfidf_results, semantic_results, output_dir='results'):
    """
    Analyze the overlap between TF-IDF and semantic search results.
    
    Args:
        tfidf_results (dict): Dictionary mapping queries to TF-IDF search results
        semantic_results (dict): Dictionary mapping queries to semantic search results
        output_dir (str): Directory to save analysis results
        
    Returns:
        dict: Dictionary with overlap statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    overlap_stats = {}
    
    for query in tfidf_results.keys():
        tfidf_chunk_ids = [result['chunk_id'] for result in tfidf_results[query]]
        semantic_chunk_ids = [result['chunk_id'] for result in semantic_results[query]]
        
        # Calculate overlap
        common_chunks = set(tfidf_chunk_ids).intersection(set(semantic_chunk_ids))
        overlap_percentage = len(common_chunks) / len(tfidf_chunk_ids) * 100
        
        overlap_stats[query] = {
            'common_chunks': list(common_chunks),
            'overlap_percentage': float(overlap_percentage),
            'tfidf_unique': list(set(tfidf_chunk_ids) - set(semantic_chunk_ids)),
            'semantic_unique': list(set(semantic_chunk_ids) - set(tfidf_chunk_ids))
        }
        
        print(f"\nOverlap analysis for query: {query}")
        print(f"Overlap percentage: {overlap_percentage:.2f}%")
        print(f"Common chunks: {len(common_chunks)}")
        print(f"TF-IDF unique chunks: {len(overlap_stats[query]['tfidf_unique'])}")
        print(f"Semantic unique chunks: {len(overlap_stats[query]['semantic_unique'])}")
    
    # Calculate average overlap
    avg_overlap = np.mean([stats['overlap_percentage'] for stats in overlap_stats.values()])
    print(f"\nAverage overlap percentage: {avg_overlap:.2f}%")
    
    # Save overlap statistics
    with open(os.path.join(output_dir, 'results_overlap.json'), 'w', encoding='utf-8') as f:
        json.dump(overlap_stats, f, ensure_ascii=False, indent=2)
    
    # Visualize overlap
    plt.figure(figsize=(10, 6))
    overlaps = [stats['overlap_percentage'] for stats in overlap_stats.values()]
    plt.bar(range(len(overlaps)), overlaps)
    plt.xlabel('Query')
    plt.ylabel('Overlap Percentage (%)')
    plt.title('Overlap Between TF-IDF and Semantic Search Results')
    plt.xticks(range(len(overlaps)), [f'Query {i+1}' for i in range(len(overlaps))])
    plt.savefig(os.path.join(output_dir, 'results_overlap.png'))
    plt.close()
    
    return overlap_stats

def load_results(tfidf_path='results/tfidf_results.json', semantic_path='results/semantic_results.json'):
    """
    Load TF-IDF and semantic search results from JSON files.
    
    Args:
        tfidf_path (str): Path to TF-IDF results file
        semantic_path (str): Path to semantic search results file
        
    Returns:
        tuple: (tfidf_results, semantic_results)
    """
    with open(tfidf_path, 'r', encoding='utf-8') as f:
        tfidf_results = json.load(f)
    
    with open(semantic_path, 'r', encoding='utf-8') as f:
        semantic_results = json.load(f)
    
    return tfidf_results, semantic_results

if __name__ == "__main__":
    # Example usage
    tfidf_results, semantic_results = load_results()
    
    # Evaluate retrieval
    evaluation_results = evaluate_retrieval(tfidf_results, semantic_results)
    
    # Analyze results overlap
    overlap_stats = analyze_results_overlap(tfidf_results, semantic_results)
