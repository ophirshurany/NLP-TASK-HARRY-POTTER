import os
import sys
import json
from tfidf_retrieval import TFIDFRetriever
from semantic_search import SemanticSearcher
from preprocessing import load_chunks, process_harry_potter_text

def interactive_retrieval_system(tfidf_retriever, semantic_searcher):
    """
    Interactive command-line interface for retrieving passages from Harry Potter.
    
    Args:
        tfidf_retriever (TFIDFRetriever): Instance of TF-IDF retriever
        semantic_searcher (SemanticSearcher): Instance of semantic searcher
    """
    print("\n" + "="*50)
    print("  Interactive Passage Retrieval System")
    print("="*50)
    print("Enter your query to retrieve passages from Harry Potter")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() == 'exit':
            print("\nThank you for using the retrieval system!")
            break
        
        print("\n" + "-"*20 + " TF-IDF Results " + "-"*20)
        tfidf_results = tfidf_retriever.search(query)
        for i, result in enumerate(tfidf_results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            # Print text with line breaks for readability
            text = result['text']
            if len(text) > 300:
                text = text[:300] + "..."
            print(text)
        
        print("\n" + "-"*20 + " Semantic Search Results " + "-"*20)
        semantic_results = semantic_searcher.search(query)
        for i, result in enumerate(semantic_results):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            # Print text with line breaks for readability
            text = result['text']
            if len(text) > 300:
                text = text[:300] + "..."
            print(text)
        
        # Save this query and results
        save_query_results(query, tfidf_results, semantic_results)

def save_query_results(query, tfidf_results, semantic_results, output_dir='results/interactive_queries'):
    """
    Save the results of an interactive query.
    
    Args:
        query (str): The search query
        tfidf_results (list): List of TF-IDF search results
        semantic_results (list): List of semantic search results
        output_dir (str): Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the query
    safe_query = "".join(c if c.isalnum() else "_" for c in query)
    safe_query = safe_query[:50]  # Limit filename length
    
    results = {
        'query': query,
        'tfidf_results': tfidf_results,
        'semantic_results': semantic_results
    }
    
    output_path = os.path.join(output_dir, f"{safe_query}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def setup_retrieval_system():
    """
    Set up the retrieval system by loading or processing the Harry Potter text
    and initializing the retrievers.
    
    Returns:
        tuple: (tfidf_retriever, semantic_searcher)
    """
    # Check if chunks file exists
    chunks_path = 'nlp-passage-retrieval/data/text_chunks.json'
    if not os.path.exists(chunks_path):
        print("Processing Harry Potter text...")
        chunks = process_harry_potter_text(chunks_output_path=chunks_path)
    else:
        print("Loading existing text chunks...")
        chunks = load_chunks(chunks_path)
    
    print(f"Loaded {len(chunks)} text chunks")
    
    # Initialize TF-IDF retriever
    print("\nInitializing TF-IDF retriever...")
    tfidf_retriever = TFIDFRetriever(chunks)
    
    # Initialize semantic searcher
    print("\nInitializing semantic searcher (this may take a moment)...")
    semantic_searcher = SemanticSearcher(chunks)
    
    return tfidf_retriever, semantic_searcher

def main():
    """
    Main function to run the interactive retrieval system.
    """
    print("Setting up the retrieval system...")
    
    try:
        tfidf_retriever, semantic_searcher = setup_retrieval_system()
        interactive_retrieval_system(tfidf_retriever, semantic_searcher)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
