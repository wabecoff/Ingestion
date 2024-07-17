import os
import pandas as pd
import json
from typing import List, Dict, Optional

class Document:
    """
    A class representing a document.

    Attributes:
        doc_id (str): The unique identifier for the document.
        title (str): The title of the document.
        text (str): The text content of the document.
    """
    def __init__(self, doc_id: str, title: str, text: str):
        self.doc_id = doc_id
        self.title = title
        self.text = text

class Chunk:
    """
    A class representing a chunk of a document.
    Does not save chunk text or embedding to avoid repeated

    Attributes:
        doc_id (str): The unique identifier for the document.
        chunk_index (int): The index of the chunk within the document.
        chunk_id (Optional[str]): The unique identifier for the chunk.
    """
    def __init__(self, doc_id: str, chunk_index: int, chunk_id: Optional[str] = None):
        self.doc_id = doc_id
        self.chunk_index = chunk_index
        self.chunk_id = f'{self.doc_id}_{self.chunk_index}'

class TextDataset:
    """
    A class representing a dataset of documents and their chunks.

    Attributes:
        documents (Dict[str, Document]): A dictionary of Document objects keyed by their document IDs.
        chunks (Dict[str, Chunk]): A dictionary of Chunk objects keyed by their chunk IDs.
        chunker (Optional[object]): An optional chunker object used to split documents into chunks.
        chunk_id_to_index (Dict[str, int]): A dictionary mapping chunk IDs to their indices.
        index_to_chunk_id (Dict[int, str]): A dictionary mapping indices to their chunk IDs.
    """
    def __init__(self, chunker=None):

        """
            Initialize an empty TextDataset object.
            Populating a TextDataset is handled in utils

            Args:
                chunker (Optional[object]): An optional chunker object used to split documents into chunks.
        """
        self.documents: Dict[str, Document] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.chunker = chunker
        self.chunk_id_to_index: Dict[str, int] = {}
        self.index_to_chunk_id: Dict[int, str] = {}

    def add_document(self, document: Document):
        self.documents[document.doc_id] = document

    def add_chunk(self, chunk: Chunk):
        self.chunks[chunk.chunk_id] = chunk

    def get_document_by_id(self, doc_id: str) -> Document:
        return self.documents.get(doc_id)

    def get_chunks_for_document(self, doc_id: str, save: bool = False) -> List[Chunk]:
        """
        Retrieve and optionally save chunks for a document.

        Args:
            doc_id (str): The unique identifier for the document.
            save (bool): Whether to save the chunks to the dataset. Default is False.

        Returns:
            List[Chunk]: A list of Chunk objects for the specified document.
        """
        document = self.get_document_by_id(doc_id)

        if not self.chunker:
            return []

        chunk_texts = self.chunker.chunk_text(document.text)
        chunks = [Chunk(doc_id=doc_id, chunk_index=i) for i in range(len(chunk_texts))]

        if save:
            for chunk in chunks:
                self.add_chunk(chunk)

        return chunks

    def get_chunk_text(self, chunk: Chunk) -> str:
        """
        Retrieve the text of a chunk
        (chunks text is found at documents)

        Args:
            chunk (Chunk): The Chunk object for which to retrieve the text.

        Returns:
            str: The text content of the specified chunk.
        """
        if not self.chunker:
            print('Please add a chunker to the TextDataset class.')
            return ""

        document = self.get_document_by_id(chunk.doc_id)
        chunk_texts = self.chunker.chunk_text(document.text)
        return chunk_texts[chunk.chunk_index]
