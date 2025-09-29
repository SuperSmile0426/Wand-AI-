"""
Tests for Challenge 2 models.
"""

import pytest
from datetime import datetime
from challenge2.models import (
    DocumentType, ConfidenceLevel, EnrichmentType, DocumentInfo, SearchRequest,
    MissingInfo, EnrichmentSuggestion, SearchResponse, RatingRequest, EnrichmentResult
)


class TestDocumentType:
    """Test DocumentType enum."""
    
    def test_document_type_values(self):
        """Test DocumentType enum values."""
        assert DocumentType.PDF == "pdf"
        assert DocumentType.DOCX == "docx"
        assert DocumentType.TXT == "txt"
        assert DocumentType.HTML == "html"
    
    def test_document_type_membership(self):
        """Test DocumentType membership."""
        assert "pdf" in DocumentType.__members__.values()
        assert "docx" in DocumentType.__members__.values()
        assert "txt" in DocumentType.__members__.values()
        assert "html" in DocumentType.__members__.values()


class TestConfidenceLevel:
    """Test ConfidenceLevel enum."""
    
    def test_confidence_level_values(self):
        """Test ConfidenceLevel enum values."""
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"
    
    def test_confidence_level_membership(self):
        """Test ConfidenceLevel membership."""
        assert "high" in ConfidenceLevel.__members__.values()
        assert "medium" in ConfidenceLevel.__members__.values()
        assert "low" in ConfidenceLevel.__members__.values()


class TestEnrichmentType:
    """Test EnrichmentType enum."""
    
    def test_enrichment_type_values(self):
        """Test EnrichmentType enum values."""
        assert EnrichmentType.DOCUMENT == "document"
        assert EnrichmentType.WEB_SEARCH == "web_search"
        assert EnrichmentType.API_FETCH == "api_fetch"
        assert EnrichmentType.USER_INPUT == "user_input"
    
    def test_enrichment_type_membership(self):
        """Test EnrichmentType membership."""
        assert "document" in EnrichmentType.__members__.values()
        assert "web_search" in EnrichmentType.__members__.values()
        assert "api_fetch" in EnrichmentType.__members__.values()
        assert "user_input" in EnrichmentType.__members__.values()


class TestDocumentInfo:
    """Test DocumentInfo model."""
    
    def test_document_info_creation(self):
        """Test DocumentInfo creation."""
        now = datetime.now()
        doc_info = DocumentInfo(
            id="test_doc_123",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=now,
            size_bytes=1024,
            chunk_count=5,
            metadata={"test": "data"}
        )
        
        assert doc_info.id == "test_doc_123"
        assert doc_info.filename == "test.pdf"
        assert doc_info.content_type == "application/pdf"
        assert doc_info.document_type == DocumentType.PDF
        assert doc_info.upload_date == now
        assert doc_info.size_bytes == 1024
        assert doc_info.chunk_count == 5
        assert doc_info.metadata == {"test": "data"}
    
    def test_document_info_default_metadata(self):
        """Test DocumentInfo with default metadata."""
        doc_info = DocumentInfo(
            id="test_doc_123",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=datetime.now(),
            size_bytes=1024,
            chunk_count=5
        )
        
        assert doc_info.metadata == {}


class TestSearchRequest:
    """Test SearchRequest model."""
    
    def test_search_request_creation(self):
        """Test SearchRequest creation."""
        request = SearchRequest(
            query="What is the company's revenue?",
            include_enrichment=True,
            max_results=10
        )
        
        assert request.query == "What is the company's revenue?"
        assert request.include_enrichment is True
        assert request.max_results == 10
    
    def test_search_request_defaults(self):
        """Test SearchRequest with default values."""
        request = SearchRequest(query="Test query")
        
        assert request.query == "Test query"
        assert request.include_enrichment is True  # Default value
        assert request.max_results == 5  # Default value
    
    def test_search_request_validation(self):
        """Test SearchRequest validation."""
        # Valid request
        request = SearchRequest(query="Valid query")
        assert request.query == "Valid query"
        
        # Empty query should still be valid (validation happens at business logic level)
        request = SearchRequest(query="")
        assert request.query == ""


class TestMissingInfo:
    """Test MissingInfo model."""
    
    def test_missing_info_creation(self):
        """Test MissingInfo creation."""
        missing_info = MissingInfo(
            topic="Financial Performance",
            description="Detailed financial metrics and KPIs",
            importance="high"
        )
        
        assert missing_info.topic == "Financial Performance"
        assert missing_info.description == "Detailed financial metrics and KPIs"
        assert missing_info.importance == "high"
    
    def test_missing_info_validation(self):
        """Test MissingInfo validation."""
        # Valid missing info
        missing_info = MissingInfo(
            topic="Test Topic",
            description="Test Description",
            importance="medium"
        )
        assert missing_info.topic == "Test Topic"
        assert missing_info.description == "Test Description"
        assert missing_info.importance == "medium"


class TestEnrichmentSuggestion:
    """Test EnrichmentSuggestion model."""
    
    def test_enrichment_suggestion_creation(self):
        """Test EnrichmentSuggestion creation."""
        missing_info = MissingInfo(
            topic="Test Topic",
            description="Test Description",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.DOCUMENT,
            description="Upload additional documents",
            missing_info=[missing_info],
            suggested_actions=["Find relevant documents", "Upload files"],
            confidence=0.8,
            auto_enrichable=False
        )
        
        assert suggestion.type == EnrichmentType.DOCUMENT
        assert suggestion.description == "Upload additional documents"
        assert suggestion.missing_info == [missing_info]
        assert suggestion.suggested_actions == ["Find relevant documents", "Upload files"]
        assert suggestion.confidence == 0.8
        assert suggestion.auto_enrichable is False
    
    def test_enrichment_suggestion_defaults(self):
        """Test EnrichmentSuggestion with default values."""
        missing_info = MissingInfo(
            topic="Test Topic",
            description="Test Description",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.WEB_SEARCH,
            description="Search the web",
            missing_info=[missing_info],
            suggested_actions=["Search online"],
            confidence=0.6
        )
        
        assert suggestion.auto_enrichable is False  # Default value
    
    def test_enrichment_suggestion_confidence_validation(self):
        """Test EnrichmentSuggestion confidence validation."""
        missing_info = MissingInfo(
            topic="Test Topic",
            description="Test Description",
            importance="high"
        )
        
        # Valid confidence values
        for confidence in [0.0, 0.5, 1.0]:
            suggestion = EnrichmentSuggestion(
                type=EnrichmentType.DOCUMENT,
                description="Test",
                missing_info=[missing_info],
                suggested_actions=["Test"],
                confidence=confidence
            )
            assert suggestion.confidence == confidence
        
        # Invalid confidence values should raise validation error
        with pytest.raises(ValueError):
            EnrichmentSuggestion(
                type=EnrichmentType.DOCUMENT,
                description="Test",
                missing_info=[missing_info],
                suggested_actions=["Test"],
                confidence=1.5  # Invalid: > 1.0
            )
        
        with pytest.raises(ValueError):
            EnrichmentSuggestion(
                type=EnrichmentType.DOCUMENT,
                description="Test",
                missing_info=[missing_info],
                suggested_actions=["Test"],
                confidence=-0.1  # Invalid: < 0.0
            )


class TestSearchResponse:
    """Test SearchResponse model."""
    
    def test_search_response_creation(self):
        """Test SearchResponse creation."""
        now = datetime.now()
        doc_info = DocumentInfo(
            id="test_doc",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=now,
            size_bytes=1024,
            chunk_count=5
        )
        
        missing_info = MissingInfo(
            topic="Test Topic",
            description="Test Description",
            importance="high"
        )
        
        enrichment_suggestion = EnrichmentSuggestion(
            type=EnrichmentType.DOCUMENT,
            description="Upload more documents",
            missing_info=[missing_info],
            suggested_actions=["Find documents"],
            confidence=0.8
        )
        
        response = SearchResponse(
            query="What is the revenue?",
            answer="The revenue is $1M",
            confidence=ConfidenceLevel.HIGH,
            confidence_score=0.9,
            sources=[doc_info],
            missing_info=[missing_info],
            enrichment_suggestions=[enrichment_suggestion],
            processing_time_ms=1500
        )
        
        assert response.query == "What is the revenue?"
        assert response.answer == "The revenue is $1M"
        assert response.confidence == ConfidenceLevel.HIGH
        assert response.confidence_score == 0.9
        assert response.sources == [doc_info]
        assert response.missing_info == [missing_info]
        assert response.enrichment_suggestions == [enrichment_suggestion]
        assert response.processing_time_ms == 1500
        assert isinstance(response.timestamp, datetime)
    
    def test_search_response_default_timestamp(self):
        """Test SearchResponse with default timestamp."""
        response = SearchResponse(
            query="Test query",
            answer="Test answer",
            confidence=ConfidenceLevel.MEDIUM,
            confidence_score=0.5,
            sources=[],
            missing_info=[],
            enrichment_suggestions=[],
            processing_time_ms=1000
        )
        
        assert isinstance(response.timestamp, datetime)
        # Should be recent (within last minute)
        assert (datetime.now() - response.timestamp).total_seconds() < 60
    
    def test_search_response_confidence_score_validation(self):
        """Test SearchResponse confidence score validation."""
        # Valid confidence scores
        for score in [0.0, 0.5, 1.0]:
            response = SearchResponse(
                query="Test query",
                answer="Test answer",
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=score,
                sources=[],
                missing_info=[],
                enrichment_suggestions=[],
                processing_time_ms=1000
            )
            assert response.confidence_score == score
        
        # Invalid confidence scores should raise validation error
        with pytest.raises(ValueError):
            SearchResponse(
                query="Test query",
                answer="Test answer",
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=1.5,  # Invalid: > 1.0
                sources=[],
                missing_info=[],
                enrichment_suggestions=[],
                processing_time_ms=1000
            )


class TestRatingRequest:
    """Test RatingRequest model."""
    
    def test_rating_request_creation(self):
        """Test RatingRequest creation."""
        request = RatingRequest(
            query="What is the revenue?",
            answer="The revenue is $1M",
            rating=5,
            feedback="Excellent answer!"
        )
        
        assert request.query == "What is the revenue?"
        assert request.answer == "The revenue is $1M"
        assert request.rating == 5
        assert request.feedback == "Excellent answer!"
    
    def test_rating_request_without_feedback(self):
        """Test RatingRequest without feedback."""
        request = RatingRequest(
            query="What is the revenue?",
            answer="The revenue is $1M",
            rating=4
        )
        
        assert request.query == "What is the revenue?"
        assert request.answer == "The revenue is $1M"
        assert request.rating == 4
        assert request.feedback is None
    
    def test_rating_request_rating_validation(self):
        """Test RatingRequest rating validation."""
        # Valid ratings
        for rating in [1, 2, 3, 4, 5]:
            request = RatingRequest(
                query="Test query",
                answer="Test answer",
                rating=rating
            )
            assert request.rating == rating
        
        # Invalid ratings should raise validation error
        with pytest.raises(ValueError):
            RatingRequest(
                query="Test query",
                answer="Test answer",
                rating=0  # Invalid: < 1
            )
        
        with pytest.raises(ValueError):
            RatingRequest(
                query="Test query",
                answer="Test answer",
                rating=6  # Invalid: > 5
            )


class TestEnrichmentResult:
    """Test EnrichmentResult model."""
    
    def test_enrichment_result_creation(self):
        """Test EnrichmentResult creation."""
        now = datetime.now()
        doc_info = DocumentInfo(
            id="enriched_doc",
            filename="enriched.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=now,
            size_bytes=2048,
            chunk_count=10
        )
        
        result = EnrichmentResult(
            success=True,
            new_documents=[doc_info],
            enriched_topics=["Financial Performance", "Market Analysis"],
            sources=["https://example.com/doc1", "https://example.com/doc2"],
            timestamp=now
        )
        
        assert result.success is True
        assert result.new_documents == [doc_info]
        assert result.enriched_topics == ["Financial Performance", "Market Analysis"]
        assert result.sources == ["https://example.com/doc1", "https://example.com/doc2"]
        assert result.timestamp == now
    
    def test_enrichment_result_default_timestamp(self):
        """Test EnrichmentResult with default timestamp."""
        result = EnrichmentResult(
            success=False,
            new_documents=[],
            enriched_topics=[],
            sources=[]
        )
        
        assert isinstance(result.timestamp, datetime)
        # Should be recent (within last minute)
        assert (datetime.now() - result.timestamp).total_seconds() < 60
    
    def test_enrichment_result_failure(self):
        """Test EnrichmentResult for failure case."""
        result = EnrichmentResult(
            success=False,
            new_documents=[],
            enriched_topics=[],
            sources=["Error: Failed to fetch data"]
        )
        
        assert result.success is False
        assert result.new_documents == []
        assert result.enriched_topics == []
        assert result.sources == ["Error: Failed to fetch data"]


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_document_info_serialization(self):
        """Test DocumentInfo serialization."""
        now = datetime.now()
        doc_info = DocumentInfo(
            id="test_doc",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date=now,
            size_bytes=1024,
            chunk_count=5,
            metadata={"test": "data"}
        )
        
        data = doc_info.model_dump()
        assert data["id"] == "test_doc"
        assert data["filename"] == "test.pdf"
        assert data["content_type"] == "application/pdf"
        assert data["document_type"] == "pdf"
        assert data["size_bytes"] == 1024
        assert data["chunk_count"] == 5
        assert data["metadata"] == {"test": "data"}
    
    def test_search_request_serialization(self):
        """Test SearchRequest serialization."""
        request = SearchRequest(
            query="What is the revenue?",
            include_enrichment=True,
            max_results=10
        )
        
        data = request.model_dump()
        assert data["query"] == "What is the revenue?"
        assert data["include_enrichment"] is True
        assert data["max_results"] == 10
    
    def test_missing_info_serialization(self):
        """Test MissingInfo serialization."""
        missing_info = MissingInfo(
            topic="Financial Performance",
            description="Detailed financial metrics",
            importance="high"
        )
        
        data = missing_info.model_dump()
        assert data["topic"] == "Financial Performance"
        assert data["description"] == "Detailed financial metrics"
        assert data["importance"] == "high"
    
    def test_enrichment_suggestion_serialization(self):
        """Test EnrichmentSuggestion serialization."""
        missing_info = MissingInfo(
            topic="Test Topic",
            description="Test Description",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.DOCUMENT,
            description="Upload documents",
            missing_info=[missing_info],
            suggested_actions=["Find docs"],
            confidence=0.8,
            auto_enrichable=True
        )
        
        data = suggestion.model_dump()
        assert data["type"] == "document"
        assert data["description"] == "Upload documents"
        assert data["confidence"] == 0.8
        assert data["auto_enrichable"] is True
        assert len(data["missing_info"]) == 1
        assert data["missing_info"][0]["topic"] == "Test Topic"
    
    def test_model_deserialization(self):
        """Test model deserialization from dict."""
        data = {
            "query": "What is the revenue?",
            "include_enrichment": True,
            "max_results": 5
        }
        request = SearchRequest(**data)
        assert request.query == "What is the revenue?"
        assert request.include_enrichment is True
        assert request.max_results == 5
    
    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        request = SearchRequest(query="Test query")
        json_str = request.model_dump_json()
        assert "Test query" in json_str
        assert "query" in json_str
