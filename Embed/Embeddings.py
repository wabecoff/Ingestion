from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
from typing import List, Optional

class EmbeddingModel:
    def __init__(self, model_name: Optional[str] = None, batch_size: int = 8, device: str = 'cpu'):
        """
        Initialize the EmbeddingModel with the specified model name, batch size, and device.
        Mainly a wrapper for easy use of HF SentenceTransformer class.
        Browse models at https://huggingface.co/sentence-transformers

        Args:
            model_name (Optional[str]): The name of the sentence transformer model to use. Defaults to 'all-mpnet-base-v2'.
            batch_size (int): The size of the batches for encoding.
            device (str): The device to run the model on, either 'cpu' or 'cuda'.
        """
        self.model_name = model_name if model_name else 'all-mpnet-base-v2'

        print(f'Downloading {self.model_name} from sentence transformers.')
        self.model = SentenceTransformer(f'sentence-transformers/{self.model_name}').to(device)
        print(f'Model max sequence length is {self.model.max_seq_length}.')
        self.batch_size = batch_size

    def embed(self, passages: List[str]) -> np.ndarray:
        """
        Embed a list of passages using the sentence transformer model.

        Args:
            passages (List[str]): A list of passages to embed.

        Returns:
            np.ndarray: A numpy array — (n_passages, dim) — containing the embeddings of the passages.
        """

        if len(passages) == 0:
            return None

        encoded_batches = []
        for i in tqdm(range(0, len(passages), self.batch_size), desc="Embedding passages"):
            batch = passages[i:i + self.batch_size]
            encoded_batch = self.model.encode(batch)
            encoded_batches.append(encoded_batch)

        # Stack all the encoded batches into a single numpy array
        embeddings = np.vstack(encoded_batches)

        return embeddings
