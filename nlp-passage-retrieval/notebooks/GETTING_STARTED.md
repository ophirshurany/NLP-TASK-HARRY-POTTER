# Getting Started with NLP Passage Retrieval

This document provides instructions for setting up and running the NLP Passage Retrieval project, which compares lexical retrieval (TF-IDF) and semantic search (miniLM) on the text of the first Harry Potter book.

## Project Structure

```
nlp-passage-retrieval/
├── data/                      # Data directory
│   └── text_chunks.json       # Processed text chunks (created during execution)
├── notebooks/                 # Jupyter notebooks
│   ├── 1_data_preprocessing.ipynb    # Data acquisition and preprocessing
│   ├── 2_tfidf_retrieval.ipynb       # TF-IDF implementation and examples
│   ├── 3_semantic_search.ipynb       # Semantic search implementation and examples
│   ├── 4_evaluation.ipynb            # Evaluation of both methods
│   └── 5_interactive_demo.ipynb      # Interactive retrieval system
├── results/                   # Results and visualizations
│   ├── analysis.md            # Detailed analysis of findings
│   └── ...                    # Various output files generated during execution
├── src/                       # Python modules
│   ├── preprocessing.py       # Text preprocessing utilities
│   ├── tfidf_retrieval.py     # TF-IDF retrieval implementation
│   ├── semantic_search.py     # Semantic search implementation
│   ├── evaluation.py          # Evaluation metrics and utilities
│   └── interactive.py         # Interactive retrieval system
├── GETTING_STARTED.md         # This file
├── README.md                  # Project overview
└── requirements.txt           # Python dependencies
```

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/nlp-passage-retrieval.git
   cd nlp-passage-retrieval
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Option 1: Jupyter Notebooks

The project includes a series of Jupyter notebooks that walk through each step of the process:

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks` directory and open the notebooks in sequence:
   - `1_data_preprocessing.ipynb`: Download and preprocess the Harry Potter text
   - `2_tfidf_retrieval.ipynb`: Implement and test TF-IDF retrieval
   - `3_semantic_search.ipynb`: Implement and test semantic search with miniLM
   - `4_evaluation.ipynb`: Evaluate and compare both methods
   - `5_interactive_demo.ipynb`: Try the interactive retrieval system

### Option 2: Command-Line Interface

For a quick demonstration of the interactive retrieval system, you can use the command-line interface:

```bash
python src/interactive.py
```

This will:
1. Download the Harry Potter text (if not already downloaded)
2. Process the text into chunks
3. Initialize both retrieval methods
4. Start an interactive prompt where you can enter queries

Example usage:
```
Enter your query: Harry's first day at Hogwarts
```

The system will display the top 5 results from both TF-IDF and semantic search.

## Expected Outputs

When running the notebooks or command-line interface, the following outputs will be generated:

1. **Data files**:
   - `data/harry_potter_book1.txt`: The downloaded Harry Potter text
   - `data/text_chunks.json`: The preprocessed text chunks

2. **Result files**:
   - `results/tfidf_results.json`: Results from TF-IDF retrieval
   - `results/semantic_results.json`: Results from semantic search
   - `results/evaluation_metrics.json`: Evaluation metrics
   - Various visualization files (PNG images)

3. **Interactive query results**:
   - `results/interactive_queries/`: Directory containing saved query results

## Notes

- The first run will download the Harry Potter text and the miniLM model, which may take some time depending on your internet connection.
- The semantic search initialization may take a moment as it loads the model and generates embeddings for all text chunks.
- If you encounter any issues with the Hugging Face Transformers library, make sure you have the latest version installed.

## Time Allocation

This project was designed to be completed within a 6-hour time limit, with time allocated as follows:

- Setup and data acquisition: 30 minutes
- Text preprocessing: 30 minutes
- TF-IDF implementation: 1 hour
- miniLM implementation: 1.5 hours
- Evaluation: 1 hour
- Discussion and analysis: 1 hour
- Documentation and GitHub setup: 30 minutes
