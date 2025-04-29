import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

class TFIDFRetriever:
    """
    Class for retrieving text passages using TF-IDF vectorization and cosine similarity.
    """
    
    def __init__(self, chunks, stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2):
        """
        Initialize the TF-IDF retriever.
        
        Args:
            chunks (list): List of text chunks to index
            stop_words (str or list): Stop words to remove ('english' or custom list)
            ngram_range (tuple): Range of n-grams to consider (e.g., (1, 2) for unigrams and bigrams)
            max_df (float): Ignore terms that appear in more than max_df fraction of documents
            min_df (int or float): Ignore terms that appear in fewer than min_df documents
        """
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=stop_words,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df
        )
        # Fit and transform the chunks to create the TF-IDF matrix
        self.tfidf_matrix = self.vectorizer.fit_transform(chunks)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def search(self, query, top_k=5):
        """
        Search for chunks similar to the query.
        
        Args:
            query (str): The search query
            top_k (int): Number of top results to return
            
        Returns:
            list: List of dictionaries containing chunk_id, text, and similarity score
        """
        # Transform query to TF-IDF space
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all chunks
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices of top k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append({
                'chunk_id': int(idx),
                'text': self.chunks[idx],
                'score': float(similarities[idx])
            })
        
        return results
    
    def save_model(self, output_dir='results'):
        """
        Save the TF-IDF model and vocabulary.
        
        Args:
            output_dir (str): Directory to save the model
            
        Returns:
            str: Path to the saved model
        """
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save vocabulary
        vocab_path = os.path.join(output_dir, 'tfidf_vocabulary.json')
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(self.vectorizer.vocabulary_, f, ensure_ascii=False, indent=2)
        
        print(f"TF-IDF vocabulary saved to {vocab_path}")
        return vocab_path

def run_example_searches(retriever, queries, output_path=None):
    """
    Run example searches and optionally save the results.
    
    Args:
        retriever (TFIDFRetriever): The retriever instance
        queries (list): List of query strings
        output_path (str): Path to save the results (optional)
        
    Returns:
        dict: Dictionary mapping queries to search results
    """
    results = {}
    
    for query in queries:
        query_results = retriever.search(query)
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

def analyze_tfidf_issues():
    """
    Analyze and return the main issues with TF-IDF approach.
    
    Returns:
        list: List of issues with descriptions
    """
    issues = [
        {
            "issue": "Lexical gap",
            "description": "TF-IDF cannot handle synonyms or semantically related terms that don't share the same tokens."
        },
        {
            "issue": "Term frequency bias",
            "description": "Common terms in specific contexts may be underweighted, while rare but irrelevant terms may be overweighted."
        },
        {
            "issue": "No understanding of word order or context",
            "description": "TF-IDF treats documents as bags of words, ignoring the order and context which can change meaning."
        },
        {
            "issue": "Cannot capture meaning behind polysemous words",
            "description": "Words with multiple meanings are treated the same, regardless of the context they appear in."
        },
        {
            "issue": "Limited by vocabulary in corpus",
            "description": "Can only match terms that appear in the corpus, making it vulnerable to vocabulary mismatch."
        }
    ]
    
    return issues

if __name__ == "__main__":
    # Example usage
    from preprocessing import load_chunks
    
    # Load chunks
    chunks = load_chunks()
    
    # Initialize retriever
    retriever = TFIDFRetriever(chunks)
    
    # Example queries
    example_queries = [
        "Harry's first day at Hogwarts",
        "Quidditch match rules",
        "Voldemort and the Philosopher's Stone"
    ]
    
    # Run example searches
    results = run_example_searches(retriever, example_queries, 'results/tfidf_results.json')
    
    # Print TF-IDF issues
    print("\nMain issues with TF-IDF approach:")
    for issue in analyze_tfidf_issues():
        print(f"- {issue['issue']}: {issue['description']}")
