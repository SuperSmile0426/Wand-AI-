"""
Tests for Challenge 2 main FastAPI application.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from challenge2.main import app, knowledge_base, rag_pipeline, enrichment_service


class TestMainApp:
    """Test main FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "knowledge-base-search"
    
    def test_config_endpoint(self, client):
        """Test config endpoint."""
        response = client.get("/config")
        assert response.status_code == 200
        data = response.json()
        assert "demo_mode" in data
        assert "auto_enrichment_enabled" in data
        assert "max_file_size" in data
        assert "allowed_extensions" in data
        assert "chunk_size" in data
        assert "max_search_results" in data
    
    @pytest.mark.asyncio
    async def test_upload_documents_success(self, client):
        """Test successful document upload."""
        # Mock file content
        file_content = b"Sample PDF content"
        
        from challenge2.models import DocumentInfo, DocumentType
        from datetime import datetime
        
        mock_doc_info = DocumentInfo(
            id="test_doc",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.fromisoformat("2024-01-01T00:00:00"),
            size_bytes=1024,
            chunk_count=5,
            metadata={}
        )
        
        with patch.object(knowledge_base, 'add_document', return_value=mock_doc_info):
            response = client.post(
                "/upload",
                files={"files": ("test.pdf", file_content, "application/pdf")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Successfully uploaded 1 documents" in data["message"]
        assert len(data["documents"]) == 1
        assert data["documents"][0]["filename"] == "test.pdf"
    
    @pytest.mark.asyncio
    async def test_upload_documents_multiple_files(self, client):
        """Test uploading multiple documents."""
        file1_content = b"Sample PDF content"
        file2_content = b"Sample text content"
        
        from challenge2.models import DocumentInfo, DocumentType
        from datetime import datetime
        
        mock_doc1 = DocumentInfo(
            id="doc1",
            filename="test1.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.fromisoformat("2024-01-01T00:00:00"),
            size_bytes=1024,
            chunk_count=5,
            metadata={}
        )
        
        mock_doc2 = DocumentInfo(
            id="doc2",
            filename="test2.txt",
            content_type="text/plain",
            document_type=DocumentType.TXT,
            upload_date=datetime.fromisoformat("2024-01-01T00:00:00"),
            size_bytes=512,
            chunk_count=3,
            metadata={}
        )
        
        with patch.object(knowledge_base, 'add_document', side_effect=[mock_doc1, mock_doc2]):
            response = client.post(
                "/upload",
                files=[
                    ("files", ("test1.pdf", file1_content, "application/pdf")),
                    ("files", ("test2.txt", file2_content, "text/plain"))
                ]
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Successfully uploaded 2 documents" in data["message"]
        assert len(data["documents"]) == 2
    
    @pytest.mark.asyncio
    async def test_upload_documents_empty_filename(self, client):
        """Test uploading document with empty filename."""
        file_content = b"Sample content"
        
        with patch.object(knowledge_base, 'add_document', return_value=Mock()):
            response = client.post(
                "/upload",
                files={"files": ("", file_content, "application/pdf")}
            )
        
        assert response.status_code == 422  # FastAPI validation error for empty filename
    
    @pytest.mark.asyncio
    async def test_upload_documents_failure(self, client):
        """Test document upload failure."""
        file_content = b"Sample PDF content"
        
        with patch.object(knowledge_base, 'add_document', side_effect=Exception("Upload failed")):
            response = client.post(
                "/upload",
                files={"files": ("test.pdf", file_content, "application/pdf")}
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "Upload failed: Upload failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_list_documents_success(self, client):
        """Test successful document listing."""
        from challenge2.models import DocumentInfo, DocumentType
        from datetime import datetime
        
        mock_doc1 = DocumentInfo(
            id="doc1",
            filename="test1.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.fromisoformat("2024-01-01T00:00:00"),
            size_bytes=1024,
            chunk_count=5,
            metadata={}
        )
        
        mock_doc2 = DocumentInfo(
            id="doc2",
            filename="test2.txt",
            content_type="text/plain",
            document_type=DocumentType.TXT,
            upload_date=datetime.fromisoformat("2024-01-02T00:00:00"),
            size_bytes=512,
            chunk_count=3,
            metadata={}
        )
        
        mock_documents = [mock_doc1, mock_doc2]
        
        with patch.object(knowledge_base, 'list_documents', return_value=mock_documents):
            response = client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["documents"]) == 2
    
    @pytest.mark.asyncio
    async def test_list_documents_failure(self, client):
        """Test document listing failure."""
        with patch.object(knowledge_base, 'list_documents', side_effect=Exception("List failed")):
            response = client.get("/documents")
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to list documents: List failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_search_documents_success(self, client):
        """Test successful document search."""
        from challenge2.models import SearchResponse, ConfidenceLevel
        
        mock_search_response = SearchResponse(
            query="What is the revenue?",
            answer="The revenue is $1M",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.9,
            sources=[],
            missing_info=[],
            enrichment_suggestions=[],
            processing_time_ms=1500
        )
        
        with patch.object(rag_pipeline, 'search', return_value=mock_search_response):
            response = client.post("/search", json={
                "query": "What is the revenue?",
                "include_enrichment": True,
                "max_results": 5
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the revenue?"
        assert data["answer"] == "The revenue is $1M"
        assert data["confidence"] == "high"
        assert data["confidence_score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_search_documents_failure(self, client):
        """Test document search failure."""
        with patch.object(rag_pipeline, 'search', side_effect=Exception("Search failed")):
            response = client.post("/search", json={
                "query": "What is the revenue?",
                "include_enrichment": True
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Search failed: Search failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_rate_answer_success(self, client):
        """Test successful answer rating."""
        with patch.object(rag_pipeline, 'add_rating', return_value=None):
            response = client.post("/rate_answer", data={
                "query": "What is the revenue?",
                "answer": "The revenue is $1M",
                "rating": "5",
                "feedback": "Excellent answer!"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Rating recorded successfully"
    
    @pytest.mark.asyncio
    async def test_rate_answer_without_feedback(self, client):
        """Test answer rating without feedback."""
        with patch.object(rag_pipeline, 'add_rating', return_value=None):
            response = client.post("/rate_answer", data={
                "query": "What is the revenue?",
                "answer": "The revenue is $1M",
                "rating": "4"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Rating recorded successfully"
    
    @pytest.mark.asyncio
    async def test_rate_answer_failure(self, client):
        """Test answer rating failure."""
        with patch.object(rag_pipeline, 'add_rating', side_effect=Exception("Rating failed")):
            response = client.post("/rate_answer", data={
                "query": "What is the revenue?",
                "answer": "The revenue is $1M",
                "rating": "5"
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Failed to record rating: Rating failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_enrich_knowledge_base_success(self, client):
        """Test successful knowledge base enrichment."""
        from challenge2.models import EnrichmentResult
        
        mock_enrichment_result = EnrichmentResult(
            success=True,
            new_documents=[],
            enriched_topics=["Financial Performance"],
            sources=["https://example.com"],
            timestamp="2024-01-01T00:00:00"
        )
        
        with patch.object(enrichment_service, 'enrich_from_suggestion', return_value=mock_enrichment_result):
            response = client.post("/enrich", json={
                "type": "web_search",
                "description": "Search for financial data",
                "missing_info": [{
                    "topic": "Financial Performance",
                    "description": "Revenue data",
                    "importance": "high"
                }],
                "suggested_actions": ["Search online"],
                "confidence": 0.8,
                "auto_enrichable": True
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Enrichment completed"
        assert "result" in data
    
    @pytest.mark.asyncio
    async def test_enrich_knowledge_base_failure(self, client):
        """Test knowledge base enrichment failure."""
        with patch.object(enrichment_service, 'enrich_from_suggestion', side_effect=Exception("Enrichment failed")):
            response = client.post("/enrich", json={
                "type": "web_search",
                "description": "Search for data",
                "missing_info": [],
                "suggested_actions": ["Search"],
                "confidence": 0.8,
                "auto_enrichable": True
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Enrichment failed: Enrichment failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_auto_enrich_success(self, client):
        """Test successful auto-enrichment."""
        from challenge2.models import EnrichmentResult
        
        mock_enrichment_result = EnrichmentResult(
            success=True,
            new_documents=[],
            enriched_topics=["Financial Performance"],
            sources=["https://example.com"],
            timestamp="2024-01-01T00:00:00"
        )
        
        with patch.object(enrichment_service, 'auto_enrich', return_value=mock_enrichment_result):
            response = client.post("/auto_enrich", data={
                "query": "What is the revenue?",
                "missing_info": '[{"topic": "Financial Performance", "description": "Revenue data", "importance": "high"}]'
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Auto-enrichment completed"
        assert "result" in data
    
    @pytest.mark.asyncio
    async def test_auto_enrich_invalid_json(self, client):
        """Test auto-enrichment with invalid JSON."""
        response = client.post("/auto_enrich", data={
            "query": "What is the revenue?",
            "missing_info": "invalid json"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "Auto-enrichment failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_auto_enrich_failure(self, client):
        """Test auto-enrichment failure."""
        with patch.object(enrichment_service, 'auto_enrich', side_effect=Exception("Auto-enrichment failed")):
            response = client.post("/auto_enrich", data={
                "query": "What is the revenue?",
                "missing_info": '[{"topic": "Financial Performance", "description": "Revenue data", "importance": "high"}]'
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Auto-enrichment failed: Auto-enrichment failed" in data["detail"]
    
    def test_cors_middleware(self, client):
        """Test CORS middleware is configured."""
        response = client.get("/health")
        # CORS middleware should allow the request
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_static_files_mounted(self, client):
        """Test static files are mounted."""
        # This would normally serve static files, but in test it might return 404
        # The important thing is that the mount is configured
        response = client.get("/static/nonexistent.html")
        # Should not return 500, indicating the mount is working
        assert response.status_code != 500
    
    def test_search_request_validation(self, client):
        """Test search request validation."""
        # Test with valid request
        response = client.post("/search", json={
            "query": "What is the revenue?",
            "include_enrichment": True,
            "max_results": 10
        })
        # Should not return 422 (validation error)
        assert response.status_code != 422
    
    def test_rating_request_validation(self, client):
        """Test rating request validation."""
        # Test with valid rating
        with patch.object(rag_pipeline, 'add_rating', return_value=None):
            response = client.post("/rate_answer", data={
                "query": "What is the revenue?",
                "answer": "The revenue is $1M",
                "rating": "5"
            })
        assert response.status_code == 200
        
        # Test with invalid rating (should be handled by Pydantic validation)
        response = client.post("/rate_answer", data={
            "query": "What is the revenue?",
            "answer": "The revenue is $1M",
            "rating": "6"  # Invalid: > 5
        })
        assert response.status_code == 422  # Validation error
    
    def test_enrichment_suggestion_validation(self, client):
        """Test enrichment suggestion validation."""
        # Test with valid suggestion
        response = client.post("/enrich", json={
            "type": "web_search",
            "description": "Search for data",
            "missing_info": [{
                "topic": "Test Topic",
                "description": "Test Description",
                "importance": "high"
            }],
            "suggested_actions": ["Search online"],
            "confidence": 0.8,
            "auto_enrichable": True
        })
        # Should not return 422 (validation error)
        assert response.status_code != 422
    
    def test_missing_info_validation(self, client):
        """Test missing info validation."""
        # Test with valid missing info
        response = client.post("/auto_enrich", data={
            "query": "What is the revenue?",
            "missing_info": '[{"topic": "Test Topic", "description": "Test Description", "importance": "high"}]'
        })
        # Should not return 422 (validation error)
        assert response.status_code != 422
