# NLP Passage Retrieval - Harry Potter

This repository contains an implementation of passage retrieval techniques using both lexical (TF-IDF) and semantic (miniLM) approaches on the text of the first Harry Potter book.

## Project Overview

This project explores and compares two different passage retrieval techniques:
1. **Lexical Retrieval with TF-IDF**: Traditional approach based on term frequency and inverse document frequency
2. **Semantic Search with miniLM**: Modern approach using transformer-based embeddings

The implementation includes:
- Text preprocessing and chunking
- TF-IDF vectorization and retrieval
- Semantic embedding generation and similarity search using Faiss
- Evaluation of both approaches
- Interactive retrieval system

## Project Structure

- `data/`: Contains the Harry Potter text and processed chunks
- `notebooks/`: Jupyter notebooks demonstrating each step of the process
- `src/`: Python modules with reusable components
- `results/`: Output files and evaluation metrics

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nlp-passage-retrieval.git
   cd nlp-passage-retrieval
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the notebooks in order or use the interactive retrieval system:
   ```
   python src/interactive.py
   ```

## Implementation Details

### 1. Data Acquisition and Preprocessing
- Download the first Harry Potter book
- Clean and preprocess the text
- Divide the text into manageable chunks

### 2. Lexical Retrieval with TF-IDF
- Implement TF-IDF vectorization
- Create a search function for retrieving relevant passages
- Demonstrate with example queries

### 3. Semantic Search with miniLM
- Load and utilize the miniLM model from HuggingFace
- Generate embeddings for text chunks
- Implement efficient similarity search with Faiss
- Evaluate the vector space quality
- Demonstrate with example queries

### 4. Evaluation
- Use miniLM results as ground truth
- Evaluate TF-IDF results using:
  - Precision@k
  - Mean Reciprocal Rank (MRR)
  - Normalized Discounted Cumulative Gain (NDCG)

### 5. Interactive Retrieval System
- Command-line interface for interactive queries
- Display results from both retrieval methods

## Results and Analysis

The repository includes a detailed analysis of:
- Comparison between lexical and semantic retrieval approaches
- Potential improvements using LLMs
- Limitations of current approaches
- Methods for creating reliable ground truth data
