import faiss
import numpy as np
from typing import List

class VSS:
    def __init__(self, text_dataset, embedding_model, embeddings: np.ndarray):
        """
        Initialize the VSS class.
        Embedding model needs to be the same embedding model used to generate embeddings.

        Args:
            text_dataset (TextDataset): The TextDataset instance containing the data.
            embedding_model: The embedding model with an `embed` method.
            embeddings (np.ndarray): The numpy array of embeddings, shape (n, d).
        """
        self.text_dataset = text_dataset
        self.embedding_model = embedding_model
        self.embeddings = embeddings
        self.index = self._create_faiss_index(embeddings)

    def _create_faiss_index(self, embeddings: np.ndarray):
        """
        Create and populate a FAISS index with the given embeddings.

        Args:
            embeddings (np.ndarray): The embeddings to index, shape (n, d).

        Returns:
            faiss.Index: The populated FAISS index.
        """
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        return index

    def _retrieve_passages(self, indices: List[int]):
        """
        Retrieve the text passages corresponding to the given indices.

        Args:
            indices (List[int]): The indices of the top_k most similar embeddings.

        Returns:
            List[str]: The text passages corresponding to the indices.
        """
        passages = []

        #trace the indices back to the original chunk text.
        for idx in indices:
            chunk_id = self.text_dataset.index_to_chunk_id[idx]
            chunk = self.text_dataset.chunks[chunk_id] 
            passage = self.text_dataset.get_chunk_text(chunk)
            passages.append(passage)
        return passages

    def similarity_search(self, query: str, top_k: int = 5):
        """
        Search for the top_k most similar chunks to the query.

        Args:
            query (str): The query string to search for.
            top_k (int): The number of top results to return.

        Returns:
            List[str]: The text passages corresponding to the top_k most similar chunks.
        """
        query_embedding = self.embedding_model.embed([query])
        distances, indices = self.index.search(query_embedding, top_k)
        return self._retrieve_passages(indices[0])
