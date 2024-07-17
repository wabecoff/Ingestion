# tests/test_embeddings.py

import pytest
from Embed.Embeddings import EmbeddingModel
import numpy as np
from typing import List

# Create a fixture for the EmbeddingModel instance with default settings
@pytest.fixture
def default_embedding_model():
    return EmbeddingModel()

# Create a fixture for the EmbeddingModel instance with custom settings
@pytest.fixture
def custom_embedding_model():
    return EmbeddingModel(model_name='paraphrase-MiniLM-L6-v2', batch_size=4, device='cpu')

# Test the initialization of the EmbeddingModel class
def test_embedding_model_initialization(default_embedding_model):
    assert default_embedding_model.model_name == 'all-mpnet-base-v2'
    assert default_embedding_model.batch_size == 8

def test_custom_embedding_model_initialization(custom_embedding_model):
    assert custom_embedding_model.model_name == 'paraphrase-MiniLM-L6-v2'
    assert custom_embedding_model.batch_size == 4

# Test the embed method
def test_embed_method(default_embedding_model):
    passages = ["This is a test passage.", "Here is another sentence for embedding."]
    embeddings = default_embedding_model.embed(passages)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(passages)
    assert embeddings.shape[1] == default_embedding_model.model.get_sentence_embedding_dimension()

def test_embed_method_empty_list(default_embedding_model):
    passages: List[str] = []
    embeddings = default_embedding_model.embed(passages)
    assert embeddings == None


def test_embed_method_custom_model(custom_embedding_model):
    passages = ["Custom model embedding test.", "This should use a different model."]
    embeddings = custom_embedding_model.embed(passages)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(passages)
    assert embeddings.shape[1] == custom_embedding_model.model.get_sentence_embedding_dimension()
