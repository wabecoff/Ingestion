# tests/test_chunking.py

import pytest
from Chunker.Chunking import Chunker, from_config
import os
import yaml

# Create a fixture for the Chunker instance with default settings
@pytest.fixture
def default_chunker():
    return Chunker()

# Create a fixture for the Chunker instance with length chunking but short chunks
@pytest.fixture
def short_chunker():
    return Chunker(chunk_type='length', max_len = 16)

# Create a fixture for the Chunker instance with sentence chunking
@pytest.fixture
def sentence_chunker():
    return Chunker(chunk_type='sentence', max_len = 32)

# Test the initialization of the Chunker class
def test_chunker_initialization(default_chunker):
    assert default_chunker.type == 'length'
    assert default_chunker.max_tokens == 128

def test_sentence_chunker_initialization(sentence_chunker):
    assert sentence_chunker.type == 'sentence'
    assert sentence_chunker.max_tokens == 32

# Test the sentence_chunk method
def test_sentence_chunk(default_chunker):
    passage = "This is a sentence. This is another sentence."
    chunks = default_chunker.sentence_chunk(passage)
    assert chunks == ["This is a sentence.", "This is another sentence."]

def test_sentence_chunk_with_long_sentence(sentence_chunker):
    passage = "This is a very long sentence that should be chunked based on length because it exceeds the maximum token limit set for sentence chunking, it's like super duper long ."
    chunks = sentence_chunker.sentence_chunk(passage)
    assert len(chunks) > 1

def test_sentence_chunk_empty_input(default_chunker):
    chunks = default_chunker.sentence_chunk("")
    assert chunks == []

# Test the length_chunk method
def test_length_chunk(default_chunker):
    passage = "This is a sentence that should be chunked based on length. The quick brown fox jumps over the lazy dog."
    chunks = default_chunker.length_chunk(passage)
    assert len(chunks) == 1

def test_length_chunk_empty_input(default_chunker):
    chunks = default_chunker.length_chunk("")
    assert chunks == [""]

# Test the chunk_text method
def test_chunk_text_with_sentence_chunking(sentence_chunker):
    passage = "This is a sentence. This is another sentence."
    chunks = sentence_chunker.chunk_text(passage)
    assert chunks == ["This is a sentence.", "This is another sentence."]

def test_chunk_text_with_length_chunking(short_chunker):
    passage = "This is a sentence that should be chunked based on length. The quick brown fox jumps over the lazy dog."
    chunks = short_chunker.chunk_text(passage)
    assert len(chunks) > 1

def test_chunk_text_invalid_type():
    with pytest.raises(ValueError):
        chunker = Chunker(chunk_type='invalid')
        chunker.chunk_text("This is a test.")

# Test saving and loading configuration
def test_to_config(default_chunker):
    config_fn = 'test_config.yaml'
    default_chunker.to_config(config_fn)
    config_path = f'{default_chunker.config_dir}/{config_fn}'
    assert os.path.exists(config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    assert config['chunk_type'] == 'length'
    assert config['max_len'] == 128
    os.remove(config_path)

def test_from_config(default_chunker):
    config = {
        'chunk_type': 'length',
        'max_len': 128
    }
    config_fn = 'test_config.yaml'
    config_path = f'{default_chunker.config_dir}/{config_fn}'
    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    chunker = from_config(config_path)
    assert chunker.type == 'length'
    assert chunker.max_tokens == 128
    os.remove(config_path)
