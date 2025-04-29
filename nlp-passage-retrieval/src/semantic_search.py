import torch
import numpy as np
import faiss
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModel

class SemanticSearcher:
    """
    Class for semantic search using transformer-based embeddings and Faiss for similarity search.
    """
    
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the semantic searcher with a specific model.
        
        Args:
            chunks (list): List of text chunks to index
            model_name (str): Name of the HuggingFace model to use
        """
        self.chunks = chunks
        self.model_name = model_name
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Generate embeddings for all chunks
        print("Generating embeddings for all chunks...")
        self.embeddings = self._encode_chunks()
        
        # Initialize Faiss index
        self.dimension = self.embeddings.shape[1]
        print(f"Embedding dimension: {self.dimension}")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(self.embeddings)
        print(f"Added {len(chunks)} embeddings to Faiss index")
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings.
        
        Args:
            model_output: Output from the transformer model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            torch.Tensor: Mean-pooled embeddings
        """
        # Mean pooling - take attention mask into account for correct averaging
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def _encode_chunks(self):
        """
        Encode all chunks into embeddings.
        
        Returns:
            numpy.ndarray: Matrix of embeddings for all chunks
        """
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(self.chunks), batch_size):
            batch_texts = self.chunks[i:i+batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                          max_length=512, return_tensors='pt')
            
            # Compute token embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Apply mean pooling
            batch_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Convert to numpy and normalize
            batch_embeddings = batch_embeddings.numpy()
            faiss.normalize_L2(batch_embeddings)
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def encode_query(self, query):
        """
        Encode a query into an embedding.
        
        Args:
            query (str): The search query
            
        Returns:
            numpy.ndarray: Query embedding
        """
        # Tokenize
        encoded_input = self.tokenizer([query], padding=True, truncation=True, 
                                      max_length=512, return_tensors='pt')
        
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Apply mean pooling
        query_embedding = self._mean_pooling(model_output, encoded_input['attention_mask'])
        
        # Convert to numpy and normalize
        query_embedding = query_embedding.numpy()
        faiss.normalize_L2(query_embedding)
        
        return query_embedding
    
    def search(self, query, top_k=5):
        """
        Search for chunks similar to the query.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of dictionaries containing chunk_id, text, and similarity score
        """
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search in Faiss index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            # Convert L2 distance to similarity score (1 - distance/2 maps to 0-1 range)
            similarity = 1.0 - distances[0][i]/2
            results.append({
                'chunk_id': int(idx),
                'text': self.chunks[idx],
                'score': float(similarity)
            })
        
        return results
    
    def evaluate_vector_space(self, output_dir='results'):
        """
        Evaluate the quality of the vector space.
        
        Args:
            output_dir (str): Directory to save evaluation plots
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Check uniformity - compute pairwise distances for a sample
        sample_size = min(1000, len(self.embeddings))
        sample_indices = np.random.choice(len(self.embeddings), sample_size, replace=False)
        sample_embeddings = self.embeddings[sample_indices]
        
        # Compute pairwise distances
        distances = np.zeros((sample_size, sample_size))
        for i in range(sample_size):
            for j in range(i+1, sample_size):
                dist = np.sqrt(np.sum((sample_embeddings[i] - sample_embeddings[j])**2))
                distances[i, j] = dist
                distances[j, i] = dist
        
        # Plot distance distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(distances.flatten(), bins=50)
        plt.title('Distribution of Pairwise Distances in Embedding Space')
        plt.xlabel('L2 Distance')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(output_dir, 'embedding_distance_distribution.png'))
        plt.close()
        
        # Check alignment - compare semantic similarity with vector similarity
        # This would require human judgments, but we can approximate by checking
        # if similar chunks (e.g., consecutive chunks) have similar embeddings
        consecutive_distances = []
        for i in range(len(self.embeddings)-1):
            dist = np.sqrt(np.sum((self.embeddings[i] - self.embeddings[i+1])**2))
            consecutive_distances.append(dist)
        
        random_distances = []
        for _ in range(len(consecutive_distances)):
            i, j = np.random.choice(len(self.embeddings), 2, replace=False)
            dist = np.sqrt(np.sum((self.embeddings[i] - self.embeddings[j])**2))
            random_distances.append(dist)
        
        plt.figure(figsize=(10, 6))
        sns.kdeplot(consecutive_distances, label='Consecutive Chunks')
        sns.kdeplot(random_distances, label='Random Chunks')
        plt.title('Distance Distribution: Consecutive vs. Random Chunks')
        plt.xlabel('L2 Distance')
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'consecutive_vs_random_distances.png'))
        plt.close()
        
        # Compute metrics
        metrics = {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'mean_consecutive_distance': float(np.mean(consecutive_distances)),
            'mean_random_distance': float(np.mean(random_distances)),
            'alignment_ratio': float(np.mean(consecutive_distances) / np.mean(random_distances))
        }
        
        # Save metrics
        with open(os.path.join(output_dir, 'vector_space_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics
    
    def save_model_info(self, output_dir='results'):
        """
        Save information about the model.
        
        Args:
            output_dir (str): Directory to save model information
            
        Returns:
            str: Path to the saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        model_info = {
            'model_name': self.model_name,
            'embedding_dimension': self.dimension,
            'num_chunks': len(self.chunks),
            'model_choice_rationale': self._get_model_rationale()
        }
        
        output_path = os.path.join(output_dir, 'semantic_model_info.json')
        with open(output_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model information saved to {output_path}")
        return output_path
    
    def _get_model_rationale(self):
        """
        Provide rationale for the choice of model.
        
        Returns:
            dict: Dictionary with rationale points
        """
        if self.model_name == "sentence-transformers/all-MiniLM-L6-v2":
            return {
                "performance": "Good balance between performance and speed with 6 layers",
                "training_data": "Trained on diverse datasets, making it robust for general text",
                "embedding_size": "384-dimensional embeddings (compact but expressive)",
                "benchmarks": "Strong performance on the MTEB benchmark",
                "optimization": "Specifically optimized for semantic similarity tasks"
            }
        else:
            return {
                "custom_choice": f"Custom model choice: {self.model_name}"
            }

def run_example_searches(searcher, queries, output_path=None):
    """
    Run example searches and optionally save the results.
    
    Args:
        searcher (SemanticSearcher): The searcher instance
        queries (list): List of query strings
        output_path (str): Path to save the results (optional)
        
    Returns:
        dict: Dictionary mapping queries to search results
    """
    results = {}
    
    for query in queries:
        query_results = searcher.search(query)
        results[query] = query_results
        
        # Print results
        print(f"\nQuery: {query}")
        for i, result in enumerate(query_results):
            print(f"Result {i+1} (Score: {result['score']:.4f}):")
            # Print first 150 chars of the text with ellipsis if longer
            print(f"{result['text'][:150]}..." if len(result['text']) > 150 else result['text'])
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_path}")
    
    return results

if __name__ == "__main__":
    # Example usage
    from preprocessing import load_chunks
    
    # Load chunks
    chunks = load_chunks()
    
    # Initialize semantic searcher
    searcher = SemanticSearcher(chunks)
    
    # Evaluate vector space
    metrics = searcher.evaluate_vector_space()
    print("Vector Space Evaluation Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value:.4f}")
    
    # Save model info
    searcher.save_model_info()
    
    # Example queries
    example_queries = [
        "Harry's first day at Hogwarts",
        "Quidditch match rules",
        "Voldemort and the Philosopher's Stone"
    ]
    
    # Run example searches
    results = run_example_searches(searcher, example_queries, 'results/semantic_results.json')
