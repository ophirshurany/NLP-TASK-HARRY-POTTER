import requests
import re
import json
import os
import nltk

# Ensure punkt is available before any tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

from nltk.tokenize import sent_tokenize

def download_harry_potter_text(url):
    """
    Download the text of the first Harry Potter book from GitHub raw.
    Returns:
        str: The text content of the book.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error downloading Harry Potter text: {e}")
        return None

def save_text_to_file(text, output_path):
    """
    Save text to a file.
    Args:
        text (str): The text to save.
        output_path (str): Path to save the text.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Text saved to {output_path}")

def preprocess_text(text):
    """
    Clean and preprocess the text.
    Args:
        text (str): Raw text.
    Returns:
        str: Preprocessed text.
    """
    # Remove lines that are all uppercase (headers/chapter titles)
    text = re.sub(r'^[A-Z\s]+$', '', text, flags=re.MULTILINE)
    # Basic cleaning
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=500):
    """
    Divide text into chunks of approximately chunk_size characters,
    trying to break at sentence boundaries when possible.
    Args:
        text (str): The text to chunk.
        chunk_size (int): Target size for each chunk in characters.
    Returns:
        list: List of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def save_chunks(chunks, output_path):
    """
    Save the text chunks to a JSON file.
    Args:
        chunks (list): List of text chunks.
        output_path (str): Path to save the chunks.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Text chunks saved to {output_path}")

def load_chunks(input_path):
    """
    Load text chunks from a JSON file.
    Args:
        input_path (str): Path to the JSON file containing chunks.
    Returns:
        list: List of text chunks.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_harry_potter_text(book_path=None, chunks_output_path='data/text_chunks.json', chunk_size=500):
    """
    Process the Harry Potter text: download (if needed), preprocess, chunk, and save.
    Args:
        book_path (str or None): Path to a local text file. If None, downloads from GitHub.
        chunks_output_path (str): Path to save the processed chunks.
        chunk_size (int): Target size for each chunk in characters.
    Returns:
        list: List of text chunks.
    """
    # Get the text
    if book_path is None:
        text = download_harry_potter_text()
        if text is None:
            return None
    else:
        with open(book_path, 'r', encoding='utf-8') as f:
            text = f.read()

    preprocessed_text = preprocess_text(text)
    chunks = chunk_text(preprocessed_text, chunk_size)
    save_chunks(chunks, chunks_output_path)
    print(f"Processed {len(chunks)} chunks from Harry Potter text")
    return chunks

if __name__ == "__main__":
    # Example usage
    chunks = process_harry_potter_text()
    if chunks:
        print(f"First chunk: {chunks[0][:100]}...")