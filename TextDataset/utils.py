import os
import json
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .TextDataset import TextDataset, Document, Chunk

def get_csv_file_paths(directory_path: str) -> List[str]:
    """
    Get all CSV file paths in the specified directory.

    Args:
        directory_path (str): The directory to search for CSV files.

    Returns:
        List[str]: A list of paths to CSV files in the directory.
    """
    csv_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files

def make_dataset(data_dir: str, chunker=None, save_chunks=False) -> TextDataset:
    """
    Create a TextDataset from CSV files in the specified directory.

    Args:
        data_dir (str): The directory containing the CSV files.
        chunker: The chunker object used to chunk the documents.
        save_chunks (bool): Whether to save chunks in the dataset.

    Returns:
        TextDataset: The created TextDataset instance.
    """
    dataset = TextDataset(chunker=chunker)

    file_paths = get_csv_file_paths(data_dir)

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            doc = Document(
                doc_id=row['doc_id'],
                title=row['title'],
                text=row['text']
            )
            dataset.add_document(doc)

            if save_chunks and chunker:
                _ = dataset.get_chunks_for_document(row['doc_id'], save=True)

    return dataset

def chunk_document(doc_id, document, text_dataset):
    chunks = text_dataset.get_chunks_for_document(doc_id, save=True)
    chunk_texts = [text_dataset.get_chunk_text(chunk) for chunk in chunks]
    chunk_id_to_index = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
    index_to_chunk_id = {i: chunk.chunk_id for i, chunk in enumerate(chunks)}
    return chunks, chunk_texts, chunk_id_to_index, index_to_chunk_id

def chunk_and_embed_dataset(text_dataset: TextDataset, chunker, embedding_model, embeddings_path: str, threading=True):
    """
    Chunk all documents in the TextDataset, embed the chunks, and save the embeddings and index mapping.

    Args:
        text_dataset (TextDataset): The TextDataset instance.
        chunker: The chunker object used to chunk the documents.
        embedding_model: The embedding model with an `embed` method to get embeddings.
        embeddings_path (str): Path to save the embeddings numpy array.
        threading (bool): Whether to use threading for parallel processing.
    """
    all_chunks = []
    chunk_texts = []

    # Add chunker to dataset if it doesn't have one
    if not text_dataset.chunker:
        text_dataset.chunker = chunker

    chunk_id_to_index = {}
    index_to_chunk_id = {}

    if threading:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Parallelize chunking process
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(chunk_document, doc_id, document, text_dataset): doc_id for doc_id, document in text_dataset.documents.items()}
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Chunking documents"):
                results.append(future.result())

        # Update shared state in a single-threaded context
        for chunks, texts, chunk_id_map, index_map in results:
            offset = len(chunk_id_to_index)
            chunk_texts.extend(texts)
            all_chunks.extend(chunks)
            for k, v in chunk_id_map.items():
                chunk_id_to_index[k] = v + offset
            for k, v in index_map.items():
                index_to_chunk_id[k + offset] = v

    #non-parallel version
    else:
        for doc_id, document in tqdm(text_dataset.documents.items(), desc="Chunking documents"):
            chunks = text_dataset.get_chunks_for_document(doc_id=doc_id, save=True)  # will save chunks in dataset
            chunk_texts.extend(text_dataset.get_chunk_text(chunk) for chunk in chunks)
            for chunk in chunks:
                text.dataset
                current_index = len(chunk_id_to_index)
                chunk_id_to_index[chunk.chunk_id] = current_index
                index_to_chunk_id[current_index] = chunk.chunk_id

    # Embed all chunks
    embeddings = embedding_model.embed(chunk_texts)

    # Save the embeddings array as a single .npz file
    np.savez_compressed(embeddings_path, embeddings=embeddings)

    # Update the dataset's index mappings
    text_dataset.chunk_id_to_index = chunk_id_to_index
    text_dataset.index_to_chunk_id = index_to_chunk_id


def save_text_dataset(dataset: TextDataset, directory_path: str):
    """
    Save the TextDataset to the specified directory.

    Args:
        dataset (TextDataset): The TextDataset instance to save.
        directory_path (str): The directory to save the dataset.
    """
    os.makedirs(directory_path, exist_ok=True)
    # Save documents
    documents = {doc_id: vars(doc) for doc_id, doc in dataset.documents.items()}
    with open(os.path.join(directory_path, 'documents.json'), 'w') as f:
        json.dump(documents, f)

    # Save chunks
    chunks = {chunk_id: vars(chunk) for chunk_id, chunk in dataset.chunks.items()}
    with open(os.path.join(directory_path, 'chunks.json'), 'w') as f:
        json.dump(chunks, f)

    # Save index mappings
    index_mappings = {
        'chunk_id_to_index': dataset.chunk_id_to_index,
        'index_to_chunk_id': dataset.index_to_chunk_id
    }
    with open(os.path.join(directory_path, 'index_mappings.json'), 'w') as f:
        json.dump(index_mappings, f)


def load_text_dataset(directory_path: str, chunker=None) -> TextDataset:
    """
    Load a TextDataset from the specified directory.

    Args:
        directory_path (str): The directory to load the dataset from.
        chunker: The chunker object used to chunk the documents.

    Returns:
        TextDataset: The loaded TextDataset instance.
    """
    dataset = TextDataset(chunker=chunker)

    # Load documents
    with open(os.path.join(directory_path, 'documents.json'), 'r') as f:
        documents = json.load(f)
        for doc_id, doc_data in documents.items():
            doc = Document(**doc_data)
            dataset.add_document(doc)

    # Load chunks
    with open(os.path.join(directory_path, 'chunks.json'), 'r') as f:
        chunks = json.load(f)
        for chunk_id, chunk_data in chunks.items():
            chunk = Chunk(**chunk_data)
            dataset.add_chunk(chunk)

    # Load index mappings
    with open(os.path.join(directory_path, 'index_mappings.json'), 'r') as f:
        index_mappings = json.load(f)
        dataset.chunk_id_to_index = {k: int(v) for k, v in index_mappings['chunk_id_to_index'].items()}
        dataset.index_to_chunk_id = {int(k): v for k, v in index_mappings['index_to_chunk_id'].items()}

    return dataset
