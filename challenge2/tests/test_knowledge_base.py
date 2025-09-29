"""
Tests for Challenge 2 KnowledgeBase class.
"""

import pytest
import os
import tempfile
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from challenge2.knowledge_base import KnowledgeBase
from challenge2.models import DocumentType, DocumentInfo


class TestKnowledgeBase:
    """Test KnowledgeBase class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def knowledge_base(self, temp_dir):
        """Create a KnowledgeBase instance for testing."""
        with patch('challenge2.knowledge_base.chromadb.PersistentClient'), \
             patch('challenge2.knowledge_base.SentenceTransformer'):
            kb = KnowledgeBase(persist_directory=temp_dir)
            # Mock the collection
            kb.collection = Mock()
            kb.collection.get = Mock(return_value={"metadatas": []})
            kb.collection.add = Mock()
            kb.collection.query = Mock(return_value={
                "documents": [["Sample document content"]],
                "metadatas": [[{"document_id": "test_doc", "filename": "test.pdf"}]],
                "distances": [[0.1]]
            })
            kb.collection.delete = Mock()
            return kb
    
    def test_knowledge_base_initialization(self, knowledge_base, temp_dir):
        """Test KnowledgeBase initialization."""
        assert knowledge_base.persist_directory == temp_dir
        assert knowledge_base.documents_dir == "./data/documents"
        assert knowledge_base.document_metadata == {}
        assert knowledge_base.collection is not None
    
    def test_get_document_type_pdf(self, knowledge_base):
        """Test _get_document_type for PDF."""
        doc_type = knowledge_base._get_document_type("test.pdf", "application/pdf")
        assert doc_type == DocumentType.PDF
        
        doc_type = knowledge_base._get_document_type("test.PDF", "application/pdf")
        assert doc_type == DocumentType.PDF
    
    def test_get_document_type_docx(self, knowledge_base):
        """Test _get_document_type for DOCX."""
        doc_type = knowledge_base._get_document_type("test.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert doc_type == DocumentType.DOCX
        
        doc_type = knowledge_base._get_document_type("test.DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert doc_type == DocumentType.DOCX
    
    def test_get_document_type_html(self, knowledge_base):
        """Test _get_document_type for HTML."""
        doc_type = knowledge_base._get_document_type("test.html", "text/html")
        assert doc_type == DocumentType.HTML
        
        doc_type = knowledge_base._get_document_type("test.HTM", "text/html")
        assert doc_type == DocumentType.HTML
    
    def test_get_document_type_txt(self, knowledge_base):
        """Test _get_document_type for TXT."""
        doc_type = knowledge_base._get_document_type("test.txt", "text/plain")
        assert doc_type == DocumentType.TXT
        
        doc_type = knowledge_base._get_document_type("test.unknown", "application/unknown")
        assert doc_type == DocumentType.TXT
    
    def test_extract_pdf_text(self, knowledge_base):
        """Test _extract_pdf_text method."""
        # Mock PDF content
        pdf_content = b"Sample PDF content for testing"
        
        with patch('challenge2.knowledge_base.PyPDF2.PdfReader') as mock_reader:
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample PDF content for testing"
            mock_reader.return_value.pages = [mock_page]
            
            result = knowledge_base._extract_pdf_text(pdf_content)
            assert result == "Sample PDF content for testing\n"
    
    def test_extract_docx_text(self, knowledge_base):
        """Test _extract_docx_text method."""
        # Mock DOCX content
        docx_content = b"Sample DOCX content for testing"
        
        with patch('challenge2.knowledge_base.Document') as mock_doc:
            mock_paragraph = Mock()
            mock_paragraph.text = "Sample DOCX content for testing"
            mock_doc.return_value.paragraphs = [mock_paragraph]
            
            result = knowledge_base._extract_docx_text(docx_content)
            assert result == "Sample DOCX content for testing\n"
    
    def test_extract_html_text(self, knowledge_base):
        """Test _extract_html_text method."""
        # Mock HTML content
        html_content = b"<html><body><p>Sample HTML content for testing</p></body></html>"
        
        with patch('challenge2.knowledge_base.BeautifulSoup') as mock_soup:
            mock_soup.return_value.get_text.return_value = "Sample HTML content for testing"
            
            result = knowledge_base._extract_html_text(html_content)
            assert result == "Sample HTML content for testing"
    
    @pytest.mark.asyncio
    async def test_extract_text_unknown_type(self, knowledge_base):
        """Test _extract_text with unknown document type."""
        content = b"Sample text content"
        result = await knowledge_base._extract_text(content, DocumentType.TXT)
        assert result == "Sample text content"
    
    @pytest.mark.asyncio
    async def test_extract_text_exception(self, knowledge_base):
        """Test _extract_text with exception."""
        content = b"Invalid content"
        
        with patch('challenge2.knowledge_base.PyPDF2.PdfReader', side_effect=Exception("PDF error")):
            with pytest.raises(Exception) as exc_info:
                await knowledge_base._extract_text(content, DocumentType.PDF)
            assert "Text extraction failed" in str(exc_info.value)
    
    def test_chunk_text_small_content(self, knowledge_base):
        """Test _chunk_text with small content."""
        text = "Short text"
        chunks = knowledge_base._chunk_text(text, chunk_size=100, overlap=20)
        assert chunks == ["Short text"]
    
    def test_chunk_text_large_content(self, knowledge_base):
        """Test _chunk_text with large content."""
        text = "This is a long text that should be chunked into multiple pieces. " * 2
        chunks = knowledge_base._chunk_text(text, chunk_size=30, overlap=5)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 30 for chunk in chunks)
        assert all(len(chunk) > 0 for chunk in chunks)
    
    def test_chunk_text_sentence_boundary(self, knowledge_base):
        """Test _chunk_text with sentence boundary detection."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = knowledge_base._chunk_text(text, chunk_size=30, overlap=5)
        
        # Should try to break at sentence boundaries
        assert len(chunks) > 1
        assert all(len(chunk) <= 30 for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_add_document_success(self, knowledge_base, temp_dir):
        """Test successful document addition."""
        # Mock file operations
        mock_file = AsyncMock()
        mock_context_manager = AsyncMock()
        mock_context_manager.__aenter__ = AsyncMock(return_value=mock_file)
        mock_context_manager.__aexit__ = AsyncMock(return_value=None)
        
        with patch('challenge2.knowledge_base.aiofiles.open', return_value=mock_context_manager):
            
            # Mock text extraction
            knowledge_base._extract_text = AsyncMock(return_value="Sample text content")
            
            # Mock chunking
            knowledge_base._chunk_text = Mock(return_value=["Chunk 1", "Chunk 2"])
            
            # Mock embedding model
            mock_embedding = Mock()
            mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
            knowledge_base.embedding_model.encode = Mock(return_value=mock_embedding)
            
            doc_info = await knowledge_base.add_document(
                filename="test.pdf",
                content=b"Sample PDF content",
                content_type="application/pdf"
            )
        
        assert doc_info is not None
        assert hasattr(doc_info, 'filename')
        assert doc_info.filename == "test.pdf"
        assert doc_info.content_type == "application/pdf"
        assert doc_info.document_type == DocumentType.PDF
        assert doc_info.size_bytes == len(b"Sample PDF content")
        assert doc_info.chunk_count == 2
        
        # Check that file was written
        mock_file.write.assert_called_once_with(b"Sample PDF content")
        
        # Check that ChromaDB was called
        knowledge_base.collection.add.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_document_failure(self, knowledge_base):
        """Test document addition failure."""
        # Mock file operations to fail
        with patch('challenge2.knowledge_base.aiofiles.open', side_effect=Exception("File error")):
            with pytest.raises(Exception) as exc_info:
                await knowledge_base.add_document(
                    filename="test.pdf",
                    content=b"Sample PDF content",
                    content_type="application/pdf"
                )
            assert "Failed to add document test.pdf" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_search_documents_success(self, knowledge_base):
        """Test successful document search."""
        # Mock embedding model
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        knowledge_base.embedding_model.encode = Mock(return_value=mock_embedding)
        
        # Mock ChromaDB collection query
        knowledge_base.collection.query = Mock(return_value={
            "documents": [["Sample document content"]],
            "metadatas": [[{"document_id": "test_doc", "filename": "test.pdf"}]],
            "distances": [[0.1]]
        })
        
        documents, metadatas = await knowledge_base.search_documents("test query", max_results=5)
        
        assert documents == ["Sample document content"]
        assert metadatas == [{"document_id": "test_doc", "filename": "test.pdf"}]
        
        # Check that ChromaDB query was called
        knowledge_base.collection.query.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_documents_failure(self, knowledge_base):
        """Test document search failure."""
        # Mock embedding model to fail
        knowledge_base.embedding_model.encode = Mock(side_effect=Exception("Embedding error"))
        
        with pytest.raises(Exception) as exc_info:
            await knowledge_base.search_documents("test query")
        assert "Search failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_document_by_id_existing(self, knowledge_base):
        """Test get_document_by_id with existing document."""
        doc_info = DocumentInfo(
            id="test_doc",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.now(),
            size_bytes=1024,
            chunk_count=5
        )
        knowledge_base.document_metadata["test_doc"] = doc_info
        
        result = await knowledge_base.get_document_by_id("test_doc")
        assert result == doc_info
    
    @pytest.mark.asyncio
    async def test_get_document_by_id_nonexistent(self, knowledge_base):
        """Test get_document_by_id with non-existent document."""
        result = await knowledge_base.get_document_by_id("nonexistent_doc")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_list_documents(self, knowledge_base):
        """Test list_documents method."""
        doc_info1 = DocumentInfo(
            id="doc1",
            filename="test1.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.now(),
            size_bytes=1024,
            chunk_count=5
        )
        doc_info2 = DocumentInfo(
            id="doc2",
            filename="test2.txt",
            content_type="text/plain",
            document_type=DocumentType.TXT,
            upload_date=datetime.now(),
            size_bytes=512,
            chunk_count=3
        )
        
        knowledge_base.document_metadata["doc1"] = doc_info1
        knowledge_base.document_metadata["doc2"] = doc_info2
        
        documents = await knowledge_base.list_documents()
        assert len(documents) == 2
        assert doc_info1 in documents
        assert doc_info2 in documents
    
    @pytest.mark.asyncio
    async def test_get_document_chunks(self, knowledge_base):
        """Test get_document_chunks method."""
        knowledge_base.collection.get.return_value = {
            "documents": ["Chunk 1", "Chunk 2", "Chunk 3"]
        }
        
        chunks = await knowledge_base.get_document_chunks("test_doc")
        assert chunks == ["Chunk 1", "Chunk 2", "Chunk 3"]
        
        # Check that ChromaDB get was called
        knowledge_base.collection.get.assert_called_once_with(
            where={"document_id": "test_doc"},
            include=["documents"]
        )
    
    @pytest.mark.asyncio
    async def test_get_document_chunks_failure(self, knowledge_base):
        """Test get_document_chunks failure."""
        knowledge_base.collection.get.side_effect = Exception("ChromaDB error")
        
        with pytest.raises(Exception) as exc_info:
            await knowledge_base.get_document_chunks("test_doc")
        assert "Failed to get document chunks" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_delete_document_success(self, knowledge_base, temp_dir):
        """Test successful document deletion."""
        # Mock document metadata
        doc_info = DocumentInfo(
            id="test_doc",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.now(),
            size_bytes=1024,
            chunk_count=5,
            metadata={"file_path": os.path.join(temp_dir, "test.pdf")}
        )
        knowledge_base.document_metadata["test_doc"] = doc_info
        
        # Mock ChromaDB get to return chunk IDs
        knowledge_base.collection.get.return_value = {
            "ids": ["test_doc_chunk_0", "test_doc_chunk_1"]
        }
        
        # Mock file removal
        with patch('os.path.exists', return_value=True), \
             patch('os.remove') as mock_remove:
            result = await knowledge_base.delete_document("test_doc")
        
        assert result is True
        assert "test_doc" not in knowledge_base.document_metadata
        
        # Check that ChromaDB delete was called
        knowledge_base.collection.delete.assert_called_once_with(ids=["test_doc_chunk_0", "test_doc_chunk_1"])
        
        # Check that file was removed
        mock_remove.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_document_nonexistent(self, knowledge_base):
        """Test document deletion with non-existent document."""
        result = await knowledge_base.delete_document("nonexistent_doc")
        assert result is True  # Should return True even if document doesn't exist
    
    @pytest.mark.asyncio
    async def test_delete_document_failure(self, knowledge_base):
        """Test document deletion failure."""
        knowledge_base.collection.get.side_effect = Exception("ChromaDB error")
        
        with pytest.raises(Exception) as exc_info:
            await knowledge_base.delete_document("test_doc")
        assert "Failed to delete document" in str(exc_info.value)
    
    def test_load_document_metadata_success(self, knowledge_base):
        """Test _load_document_metadata success."""
        # Mock ChromaDB get to return metadata
        knowledge_base.collection.get.return_value = {
            "metadatas": [
                {
                    "document_id": "doc1",
                    "filename": "test1.pdf",
                    "content_type": "application/pdf",
                    "document_type": "pdf",
                    "upload_date": "2024-01-01T00:00:00"
                },
                {
                    "document_id": "doc1",
                    "filename": "test1.pdf",
                    "content_type": "application/pdf",
                    "document_type": "pdf",
                    "upload_date": "2024-01-01T00:00:00"
                },
                {
                    "document_id": "doc2",
                    "filename": "test2.txt",
                    "content_type": "text/plain",
                    "document_type": "txt",
                    "upload_date": "2024-01-02T00:00:00"
                }
            ]
        }
        
        knowledge_base._load_document_metadata()
        
        assert len(knowledge_base.document_metadata) == 2
        assert "doc1" in knowledge_base.document_metadata
        assert "doc2" in knowledge_base.document_metadata
        
        # Check chunk count
        assert knowledge_base.document_metadata["doc1"].chunk_count == 2
        assert knowledge_base.document_metadata["doc2"].chunk_count == 1
    
    def test_load_document_metadata_failure(self, knowledge_base):
        """Test _load_document_metadata failure."""
        knowledge_base.collection.get.side_effect = Exception("ChromaDB error")
        
        # Should not raise exception, just log error
        knowledge_base._load_document_metadata()
        assert knowledge_base.document_metadata == {}
    
    def test_load_document_metadata_invalid_data(self, knowledge_base):
        """Test _load_document_metadata with invalid data."""
        knowledge_base.collection.get.return_value = {
            "metadatas": [
                {
                    "document_id": "doc1",
                    "filename": "test1.pdf",
                    # Missing required fields
                }
            ]
        }
        
        # Should not raise exception, just skip invalid entries
        knowledge_base._load_document_metadata()
        assert knowledge_base.document_metadata == {}
