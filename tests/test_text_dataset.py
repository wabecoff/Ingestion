# tests/test_text_dataset.py

import pytest
from TextDataset.TextDataset import Document, Chunk, TextDataset

# Mock Chunker class for testing
class MockChunker:
    def chunk_text(self, text):
        # Simple mock chunker that splits text into words as chunks
        return text.split()

# Fixtures for Document and Chunk
@pytest.fixture
def document():
    return Document(doc_id="doc1", title="Test Document", text="This is a test document.")

@pytest.fixture
def chunk():
    return Chunk(doc_id="doc1", chunk_index=0)

# Fixture for TextDataset with a mock chunker
@pytest.fixture
def text_dataset():
    return TextDataset(chunker=MockChunker())

# Test Document class initialization
def test_document_initialization(document):
    assert document.doc_id == "doc1"
    assert document.title == "Test Document"
    assert document.text == "This is a test document."

# Test Chunk class initialization
def test_chunk_initialization(chunk):
    assert chunk.doc_id == "doc1"
    assert chunk.chunk_index == 0
    assert chunk.chunk_id == "doc1_0"

# Test adding a document to TextDataset
def test_add_document(text_dataset, document):
    text_dataset.add_document(document)
    assert "doc1" in text_dataset.documents
    assert text_dataset.documents["doc1"] == document

# Test adding a chunk to TextDataset
def test_add_chunk(text_dataset, chunk):
    text_dataset.add_chunk(chunk)
    assert "doc1_0" in text_dataset.chunks
    assert text_dataset.chunks["doc1_0"] == chunk

# Test retrieving a document by ID
def test_get_document_by_id(text_dataset, document):
    text_dataset.add_document(document)
    retrieved_document = text_dataset.get_document_by_id("doc1")
    assert retrieved_document == document

# Test retrieving chunks for a document without saving
def test_get_chunks_for_document_without_saving(text_dataset, document):
    text_dataset.add_document(document)
    chunks = text_dataset.get_chunks_for_document("doc1", save=False)
    assert len(chunks) == 5
    assert all(isinstance(chunk, Chunk) for chunk in chunks)

# Test retrieving chunks for a document with saving
def test_get_chunks_for_document_with_saving(text_dataset, document):
    text_dataset.add_document(document)
    chunks = text_dataset.get_chunks_for_document("doc1", save=True)
    assert len(chunks) == 5
    assert "doc1_0" in text_dataset.chunks
    assert "doc1_4" in text_dataset.chunks

# Test getting chunk text
def test_get_chunk_text(text_dataset, document):
    text_dataset.add_document(document)
    chunks = text_dataset.get_chunks_for_document("doc1", save=True)
    chunk_text = text_dataset.get_chunk_text(chunks[0])
    assert chunk_text == "This"

# Test get_chunks_for_document without a chunker
def test_get_chunks_for_document_without_chunker(document):
    text_dataset = TextDataset()
    text_dataset.add_document(document)
    chunks = text_dataset.get_chunks_for_document("doc1")
    assert chunks == []

# Test get_chunk_text without a chunker
def test_get_chunk_text_without_chunker(document):
    text_dataset = TextDataset()
    text_dataset.add_document(document)
    chunk = Chunk(doc_id="doc1", chunk_index=0)
    chunk_text = text_dataset.get_chunk_text(chunk)
    assert chunk_text == ""
