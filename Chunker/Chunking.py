import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from typing import List
import yaml

class Chunker:
    def __init__(self, chunk_type: str = 'length', max_len: int = 128):
        """
        Initialize the Chunker with the specified type and maximum length.

        Args:
            type (str): The type of chunking to perform, either 'sentence' or 'length'.
            max_len (int): The maximum number of tokens per chunk.
        """
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        self.max_tokens = max_len
        self.type = chunk_type
        self.config_dir = 'Chunker/config'

        # Download necessary NLTK data for sentence tokenization
        nltk.download('punkt')

    def sentence_chunk(self, passage: str, quiet = True) -> List[str]:
        """
        Chunk the passage into smaller sub-passages based on sentences.
        Leverages NLTK sentence tokenization

        Args:
            passage (str): The passage to be chunked.

        Returns:
            List[str]: A list of sub-passages where each sub-passage has fewer than max_tokens.
        """

        if not passage:
            return []

        try:
            sentences = sent_tokenize(passage)
        except:
            if not quiet:
                print('Sentence tokenization Error for input:', passage)
            return []

        chunks = []

        for sentence in sentences:
                tokenized_chunk = self.tokenizer.tokenize(sentence)

                # Split up sentence further if too long
                if len(tokenized_chunk) > self.max_tokens:
                    chunks.extend(self.length_chunk(sentence))
                else:
                    chunks.append(sentence)


        return chunks

    def length_chunk(self, passage: str, max_tokens: int = None, quiet = True) -> List[str]:
        """
        Chunk the passage into smaller sub-passages based on token length.

        Args:
            passage (str): The passage to be chunked.
            max_tokens (int): The maximum number of tokens per chunk.

        Returns:
            List[str]: A list of sub-passages where each sub-passage has fewer than max_tokens.
        """

        if not passage:
            return ['']

        if max_tokens is None:
            max_tokens = self.max_tokens

        try:
            tokens = self.tokenizer.tokenize(passage)
        except:
            if not quiet:
                print('HF Tokenization Error for input:', passage)
            return ['']

        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        sub_passages = [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

        return sub_passages

    def chunk_text(self, passage: str) -> List[str]:
        """
        Chunk the passage into smaller sub-passages based on the specified chunking type.

        Args:
            passage (str): The passage to be chunked.

        Returns:
            List[str]: A list of sub-passages.
        """
        if self.type == 'sentence':
            return self.sentence_chunk(passage)
        elif self.type == 'length':
            return self.length_chunk(passage)
        else:
            raise ValueError(f"Unsupported chunking type: {self.type}")

    def to_config(self, config_fn: str) -> None:
        """
        Save the Chunker configuration to a YAML file.
        Configs are saved in Chunker/config dir.

        Args:
            config_fn  (str): File name (with .yml or .yaml extenstion) for the configuration will be saved.
        """
        config = {
            'chunk_type': self.type,
            'max_len': self.max_tokens
        }
        config_path = f'{self.config_dir}/{config_fn}'
        with open(config_path, 'w') as file:
            yaml.dump(config, file)
        print(f"Configuration saved to {config_path}")

def from_config(config_path: str) -> Chunker:
    """
    Load the Chunker configuration from a YAML file and return a Chunker instance.

    Args:
        config_path (str): The path to the YAML file with the Chunker configuration.

    Returns:
        Chunker: A Chunker instance with the loaded configuration.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    return Chunker(chunk_type=config['chunk_type'], max_len=config['max_len'])
