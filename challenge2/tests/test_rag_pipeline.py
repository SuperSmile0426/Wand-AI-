"""
Tests for Challenge 2 RAGPipeline class.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from challenge2.rag_pipeline import RAGPipeline, MockLLM
from challenge2.models import (
    SearchResponse, SearchRequest, DocumentInfo, DocumentType, 
    MissingInfo, EnrichmentSuggestion, EnrichmentType, ConfidenceLevel
)


class TestRAGPipeline:
    """Test RAGPipeline class."""
    
    @pytest.fixture
    def rag_pipeline(self):
        """Create a RAGPipeline instance for testing."""
        with patch('challenge2.rag_pipeline.ChatOpenAI'):
            return RAGPipeline(openai_api_key="test_key")
    
    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base."""
        kb = Mock()
        kb.search_documents = AsyncMock(return_value=(
            ["Sample document content 1", "Sample document content 2"],
            [{"document_id": "doc1", "filename": "test1.pdf"}, {"document_id": "doc2", "filename": "test2.txt"}]
        ))
        kb.get_document_by_id = AsyncMock(return_value=DocumentInfo(
            id="doc1",
            filename="test1.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date="2024-01-01T00:00:00",
            size_bytes=1024,
            chunk_count=5
        ))
        return kb
    
    def test_rag_pipeline_initialization_with_key(self):
        """Test RAGPipeline initialization with API key."""
        with patch('challenge2.rag_pipeline.ChatOpenAI') as mock_chat:
            pipeline = RAGPipeline(openai_api_key="test_key")
            
            assert pipeline.openai_api_key == "test_key"
            assert pipeline.llm is not None
            assert pipeline.ratings == []
            mock_chat.assert_called_once()
    
    def test_rag_pipeline_initialization_without_key(self):
        """Test RAGPipeline initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            pipeline = RAGPipeline()
            
            assert pipeline.openai_api_key is None
            assert isinstance(pipeline.llm, MockLLM)
            assert pipeline.ratings == []
    
    def test_rag_pipeline_initialization_with_env_key(self):
        """Test RAGPipeline initialization with environment variable."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env_key'}):
            with patch('challenge2.rag_pipeline.ChatOpenAI') as mock_chat:
                pipeline = RAGPipeline()
                
                assert pipeline.openai_api_key == "env_key"
                assert pipeline.llm is not None
                mock_chat.assert_called_once()
    
    def test_setup_prompts(self, rag_pipeline):
        """Test _setup_prompts method."""
        assert rag_pipeline.answer_prompt is not None
        assert rag_pipeline.completeness_prompt is not None
        assert rag_pipeline.enrichment_prompt is not None
        
        # Check that prompts have required input variables
        assert "query" in rag_pipeline.answer_prompt.input_variables
        assert "context" in rag_pipeline.answer_prompt.input_variables
        assert "documents" in rag_pipeline.answer_prompt.input_variables
        
        assert "query" in rag_pipeline.completeness_prompt.input_variables
        assert "answer" in rag_pipeline.completeness_prompt.input_variables
        assert "context" in rag_pipeline.completeness_prompt.input_variables
        
        assert "query" in rag_pipeline.enrichment_prompt.input_variables
        assert "missing_info" in rag_pipeline.enrichment_prompt.input_variables
        assert "available_documents" in rag_pipeline.enrichment_prompt.input_variables
    
    @pytest.mark.asyncio
    async def test_search_success(self, rag_pipeline, mock_knowledge_base):
        """Test successful search."""
        result = await rag_pipeline.search("What is the company's revenue?", mock_knowledge_base)
        
        assert isinstance(result, SearchResponse)
        assert result.query == "What is the company's revenue?"
        assert result.answer is not None
        assert result.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
        assert 0.0 <= result.confidence_score <= 1.0
        assert len(result.sources) > 0
        assert isinstance(result.missing_info, list)
        assert isinstance(result.enrichment_suggestions, list)
        assert result.processing_time_ms > 0
        assert result.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_search_without_enrichment(self, rag_pipeline, mock_knowledge_base):
        """Test search without enrichment suggestions."""
        result = await rag_pipeline.search("What is the company's revenue?", mock_knowledge_base, include_enrichment=False)
        
        assert isinstance(result, SearchResponse)
        assert result.query == "What is the company's revenue?"
        assert result.enrichment_suggestions == []
    
    @pytest.mark.asyncio
    async def test_search_failure(self, rag_pipeline, mock_knowledge_base):
        """Test search failure."""
        mock_knowledge_base.search_documents.side_effect = Exception("Search failed")
        
        with pytest.raises(Exception) as exc_info:
            await rag_pipeline.search("What is the company's revenue?", mock_knowledge_base)
        assert "RAG search failed" in str(exc_info.value)
    
    def test_prepare_context(self, rag_pipeline):
        """Test _prepare_context method."""
        documents = ["Document 1 content", "Document 2 content"]
        metadatas = [{"filename": "doc1.pdf"}, {"filename": "doc2.txt"}]
        
        context = rag_pipeline._prepare_context(documents, metadatas)
        
        assert "Document 1 (doc1.pdf)" in context
        assert "Document 2 (doc2.txt)" in context
        assert "Document 1 content" in context
        assert "Document 2 content" in context
    
    @pytest.mark.asyncio
    async def test_get_document_infos(self, rag_pipeline, mock_knowledge_base):
        """Test _get_document_infos method."""
        metadatas = [
            {"document_id": "doc1", "filename": "test1.pdf"},
            {"document_id": "doc2", "filename": "test2.txt"},
            {"document_id": "doc1", "filename": "test1.pdf"}  # Duplicate
        ]
        
        # Mock get_document_by_id for both documents
        mock_knowledge_base.get_document_by_id.side_effect = [
            DocumentInfo(
                id="doc1",
                filename="test1.pdf",
                content_type="application/pdf",
                document_type=DocumentType.PDF,
                upload_date="2024-01-01T00:00:00",
                size_bytes=1024,
                chunk_count=5
            ),
            DocumentInfo(
                id="doc2",
                filename="test2.txt",
                content_type="text/plain",
                document_type=DocumentType.TXT,
                upload_date="2024-01-02T00:00:00",
                size_bytes=512,
                chunk_count=3
            )
        ]
        
        document_infos = await rag_pipeline._get_document_infos(metadatas, mock_knowledge_base)
        
        assert len(document_infos) == 2  # Should deduplicate
        assert any(doc.id == "doc1" for doc in document_infos)
        assert any(doc.id == "doc2" for doc in document_infos)
    
    @pytest.mark.asyncio
    async def test_generate_answer_with_chatopenai(self, rag_pipeline, mock_knowledge_base):
        """Test _generate_answer with ChatOpenAI."""
        # Mock ChatOpenAI
        mock_response = Mock()
        mock_response.content = "AI-generated answer"
        rag_pipeline.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        query = "What is the revenue?"
        context = "Sample context"
        documents = [DocumentInfo(
            id="doc1",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date="2024-01-01T00:00:00",
            size_bytes=1024,
            chunk_count=5
        )]
        
        answer = await rag_pipeline._generate_answer(query, context, documents)
        
        assert answer == "AI-generated answer"
        rag_pipeline.llm.ainvoke.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_answer_with_mock_llm(self, rag_pipeline, mock_knowledge_base):
        """Test _generate_answer with MockLLM."""
        # Set up MockLLM
        rag_pipeline.llm = MockLLM()
        
        query = "What is the revenue?"
        context = "Sample context with revenue data"
        documents = [DocumentInfo(
            id="doc1",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date="2024-01-01T00:00:00",
            size_bytes=1024,
            chunk_count=5
        )]
        
        answer = await rag_pipeline._generate_answer(query, context, documents)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    @pytest.mark.asyncio
    async def test_generate_answer_fallback(self, rag_pipeline, mock_knowledge_base):
        """Test _generate_answer fallback on error."""
        # Mock ChatOpenAI to raise exception
        rag_pipeline.llm.ainvoke = AsyncMock(side_effect=Exception("API error"))
        
        query = "What is the revenue?"
        context = "Sample context"
        documents = []
        
        answer = await rag_pipeline._generate_answer(query, context, documents)
        
        assert "Based on the available documents" in answer
        assert "Sample context" in answer
    
    @pytest.mark.asyncio
    async def test_check_completeness_with_chatopenai(self, rag_pipeline):
        """Test _check_completeness with ChatOpenAI."""
        # Mock ChatOpenAI
        mock_response = Mock()
        mock_response.content = json.dumps({
            "confidence_score": 0.8,
            "confidence_level": "high",
            "missing_info": [
                {
                    "topic": "Additional context",
                    "description": "More details needed",
                    "importance": "medium"
                }
            ],
            "completeness_reasoning": "Answer is complete"
        })
        rag_pipeline.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        query = "What is the revenue?"
        answer = "The revenue is $1M"
        context = "Sample context"
        
        result = await rag_pipeline._check_completeness(query, answer, context)
        
        assert result["confidence_score"] == 0.8
        assert result["confidence_level"] == "high"
        assert len(result["missing_info"]) == 1
        assert result["missing_info"][0]["topic"] == "Additional context"
        assert result["completeness_reasoning"] == "Answer is complete"
    
    @pytest.mark.asyncio
    async def test_check_completeness_with_mock_llm(self, rag_pipeline):
        """Test _check_completeness with MockLLM."""
        # Set up MockLLM
        rag_pipeline.llm = MockLLM()
        
        query = "What is the revenue?"
        answer = "The revenue is $1M"
        context = "Sample context"
        
        result = await rag_pipeline._check_completeness(query, answer, context)
        
        assert "confidence_score" in result
        assert "confidence_level" in result
        assert "missing_info" in result
        assert "completeness_reasoning" in result
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert result["confidence_level"] in ["high", "medium", "low"]
    
    @pytest.mark.asyncio
    async def test_check_completeness_json_error(self, rag_pipeline):
        """Test _check_completeness with JSON decode error."""
        # Mock ChatOpenAI to return invalid JSON
        mock_response = Mock()
        mock_response.content = "Invalid JSON response"
        rag_pipeline.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        query = "What is the revenue?"
        answer = "The revenue is $1M"
        context = "Sample context"
        
        result = await rag_pipeline._check_completeness(query, answer, context)
        
        # Should fallback to heuristic analysis
        assert "confidence_score" in result
        assert "confidence_level" in result
        assert "missing_info" in result
        assert "completeness_reasoning" in result
    
    def test_fallback_completeness_analysis(self, rag_pipeline):
        """Test _fallback_completeness_analysis method."""
        query = "What is the company's revenue?"
        answer = "The company's revenue is $1M based on the uploaded documents."
        context = "Sample context with revenue information"
        
        result = rag_pipeline._fallback_completeness_analysis(query, answer, context)
        
        assert "confidence_score" in result
        assert "confidence_level" in result
        assert "missing_info" in result
        assert "completeness_reasoning" in result
        assert 0.0 <= result["confidence_score"] <= 1.0
        assert result["confidence_level"] in ["high", "medium", "low"]
        assert isinstance(result["missing_info"], list)
        assert isinstance(result["completeness_reasoning"], str)
    
    def test_fallback_completeness_analysis_uncertainty(self, rag_pipeline):
        """Test _fallback_completeness_analysis with uncertainty indicators."""
        query = "What is the company's revenue?"
        answer = "I don't know the exact revenue, but it might be around $1M."
        context = "Limited context"
        
        result = rag_pipeline._fallback_completeness_analysis(query, answer, context)
        
        # Should have lower confidence due to uncertainty indicators
        assert result["confidence_score"] < 0.8
        assert len(result["missing_info"]) > 0
    
    def test_fallback_completeness_analysis_missing_info(self, rag_pipeline):
        """Test _fallback_completeness_analysis with missing information indicators."""
        query = "What is the company's revenue?"
        answer = "The revenue information is not mentioned in the available documents."
        context = "Limited context"
        
        result = rag_pipeline._fallback_completeness_analysis(query, answer, context)
        
        # Should have lower confidence due to missing info indicators
        assert result["confidence_score"] < 0.8
        assert len(result["missing_info"]) > 0
    
    @pytest.mark.asyncio
    async def test_generate_enrichment_suggestions_with_chatopenai(self, rag_pipeline):
        """Test _generate_enrichment_suggestions with ChatOpenAI."""
        # Mock ChatOpenAI
        mock_response = Mock()
        mock_response.content = json.dumps({
            "suggestions": [
                {
                    "type": "document",
                    "description": "Upload additional documents",
                    "missing_info": [{"topic": "Additional context", "description": "More details", "importance": "high"}],
                    "suggested_actions": ["Find documents", "Upload files"],
                    "confidence": 0.8,
                    "auto_enrichable": False
                }
            ]
        })
        rag_pipeline.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        query = "What is the revenue?"
        missing_info = [{"topic": "Additional context", "description": "More details", "importance": "high"}]
        documents = [DocumentInfo(
            id="doc1",
            filename="test.pdf",
            content_type="application/pdf",
            document_type=DocumentType.PDF,
            upload_date="2024-01-01T00:00:00",
            size_bytes=1024,
            chunk_count=5
        )]
        
        suggestions = await rag_pipeline._generate_enrichment_suggestions(query, missing_info, documents)
        
        assert len(suggestions) == 1
        assert suggestions[0].type == EnrichmentType.DOCUMENT
        assert suggestions[0].description == "Upload additional documents"
        assert suggestions[0].confidence == 0.8
        assert suggestions[0].auto_enrichable is False
    
    @pytest.mark.asyncio
    async def test_generate_enrichment_suggestions_with_mock_llm(self, rag_pipeline):
        """Test _generate_enrichment_suggestions with MockLLM."""
        # Set up MockLLM
        rag_pipeline.llm = MockLLM()
        
        query = "What is the revenue?"
        missing_info = [{"topic": "Additional context", "description": "More details", "importance": "high"}]
        documents = []
        
        suggestions = await rag_pipeline._generate_enrichment_suggestions(query, missing_info, documents)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, EnrichmentSuggestion) for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_generate_enrichment_suggestions_json_error(self, rag_pipeline):
        """Test _generate_enrichment_suggestions with JSON decode error."""
        # Mock ChatOpenAI to return invalid JSON
        mock_response = Mock()
        mock_response.content = "Invalid JSON response"
        rag_pipeline.llm.ainvoke = AsyncMock(return_value=mock_response)
        
        query = "What is the revenue?"
        missing_info = [{"topic": "Additional context", "description": "More details", "importance": "high"}]
        documents = []
        
        suggestions = await rag_pipeline._generate_enrichment_suggestions(query, missing_info, documents)
        
        # Should fallback to heuristic suggestions
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, EnrichmentSuggestion) for s in suggestions)
    
    def test_fallback_enrichment_suggestions(self, rag_pipeline):
        """Test _fallback_enrichment_suggestions method."""
        missing_info = [
            {"topic": "Financial Performance", "description": "Revenue data", "importance": "high"},
            {"topic": "Market Analysis", "description": "Competitor data", "importance": "medium"},
            {"topic": "Contact Info", "description": "Email addresses", "importance": "low"}
        ]
        
        suggestions = rag_pipeline._fallback_enrichment_suggestions(missing_info)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(s, EnrichmentSuggestion) for s in suggestions)
        
        # Check that high importance items get multiple suggestions
        high_importance_suggestions = [s for s in suggestions if any(
            info.importance == "high" for info in s.missing_info
        )]
        assert len(high_importance_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_add_rating(self, rag_pipeline):
        """Test add_rating method."""
        query = "What is the revenue?"
        answer = "The revenue is $1M"
        rating = 5
        feedback = "Excellent answer!"
        
        await rag_pipeline.add_rating(query, answer, rating, feedback)
        
        assert len(rag_pipeline.ratings) == 1
        rating_data = rag_pipeline.ratings[0]
        assert rating_data["query"] == query
        assert rating_data["answer"] == answer
        assert rating_data["rating"] == rating
        assert rating_data["feedback"] == feedback
        assert "timestamp" in rating_data
    
    @pytest.mark.asyncio
    async def test_add_rating_without_feedback(self, rag_pipeline):
        """Test add_rating method without feedback."""
        query = "What is the revenue?"
        answer = "The revenue is $1M"
        rating = 4
        
        await rag_pipeline.add_rating(query, answer, rating)
        
        assert len(rag_pipeline.ratings) == 1
        rating_data = rag_pipeline.ratings[0]
        assert rating_data["query"] == query
        assert rating_data["answer"] == answer
        assert rating_data["rating"] == rating
        assert rating_data["feedback"] is None
    
    def test_get_ratings(self, rag_pipeline):
        """Test get_ratings method."""
        # Add some test ratings
        rag_pipeline.ratings = [
            {"query": "Q1", "answer": "A1", "rating": 5, "feedback": "Great!"},
            {"query": "Q2", "answer": "A2", "rating": 3, "feedback": "Okay"}
        ]
        
        ratings = rag_pipeline.get_ratings()
        
        assert len(ratings) == 2
        assert ratings[0]["query"] == "Q1"
        assert ratings[1]["query"] == "Q2"
        # Should return a copy, not the original list
        assert ratings is not rag_pipeline.ratings


class TestMockLLM:
    """Test MockLLM class."""
    
    def test_mock_llm_initialization(self):
        """Test MockLLM initialization."""
        llm = MockLLM()
        assert llm is not None
    
    def test_mock_llm_arun_completeness(self):
        """Test MockLLM arun method for completeness analysis."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the revenue?",
            answer="The revenue is $1M",
            context="Sample context",
            template="completeness analysis"
        )
        
        data = json.loads(result)
        assert "confidence_score" in data
        assert "confidence_level" in data
        assert "missing_info" in data
        assert "completeness_reasoning" in data
        assert 0.0 <= data["confidence_score"] <= 1.0
        assert data["confidence_level"] in ["high", "medium", "low"]
    
    def test_mock_llm_arun_enrichment(self):
        """Test MockLLM arun method for enrichment suggestions."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the revenue?",
            missing_info='[{"topic": "Additional context", "description": "More details", "importance": "high"}]',
            available_documents="test.pdf",
            template="enrichment suggestions"
        )
        
        data = json.loads(result)
        assert "suggestions" in data
        assert isinstance(data["suggestions"], list)
        assert len(data["suggestions"]) > 0
        
        suggestion = data["suggestions"][0]
        assert "type" in suggestion
        assert "description" in suggestion
        assert "missing_info" in suggestion
        assert "suggested_actions" in suggestion
        assert "confidence" in suggestion
        assert "auto_enrichable" in suggestion
    
    def test_mock_llm_arun_answer_generation(self):
        """Test MockLLM arun method for answer generation."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the company's revenue?",
            context="The company's revenue for Q1 2023 was $1.2M, showing a 15% increase from the previous quarter.",
            documents="test.pdf"
        )
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "revenue" in result.lower() or "company" in result.lower()
    
    def test_mock_llm_arun_no_context(self):
        """Test MockLLM arun method with no context."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the company's revenue?",
            context="",
            documents=""
        )
        
        assert isinstance(result, str)
        assert "don't have enough information" in result.lower()
    
    def test_mock_llm_arun_short_context(self):
        """Test MockLLM arun method with short context."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the company's revenue?",
            context="Short",
            documents=""
        )
        
        assert isinstance(result, str)
        assert "don't have enough information" in result.lower()
    
    def test_mock_llm_arun_name_query(self):
        """Test MockLLM arun method with name query."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the person's name?",
            context="My name is John Smith and I work as a software engineer.",
            documents="resume.pdf"
        )
        
        assert isinstance(result, str)
        assert "John Smith" in result or "name" in result.lower()
    
    def test_mock_llm_arun_skill_query(self):
        """Test MockLLM arun method with skill query."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What are the person's skills?",
            context="I have experience in Python, JavaScript, and machine learning.",
            documents="resume.pdf"
        )
        
        assert isinstance(result, str)
        assert "python" in result.lower() or "javascript" in result.lower() or "skills" in result.lower()
    
    def test_mock_llm_arun_education_query(self):
        """Test MockLLM arun method with education query."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the person's education?",
            context="I graduated from MIT with a Bachelor's degree in Computer Science.",
            documents="resume.pdf"
        )
        
        assert isinstance(result, str)
        assert "mit" in result.lower() or "computer science" in result.lower() or "education" in result.lower()
    
    def test_mock_llm_arun_experience_query(self):
        """Test MockLLM arun method with experience query."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the person's work experience?",
            context="I worked at Google for 3 years as a software engineer.",
            documents="resume.pdf"
        )
        
        assert isinstance(result, str)
        assert "google" in result.lower() or "software engineer" in result.lower() or "experience" in result.lower()
    
    def test_mock_llm_arun_contact_query(self):
        """Test MockLLM arun method with contact query."""
        llm = MockLLM()
        
        result = llm.arun(
            query="What is the person's contact information?",
            context="You can reach me at john@example.com or call me at 555-1234.",
            documents="resume.pdf"
        )
        
        assert isinstance(result, str)
        assert "john@example.com" in result.lower() or "555-1234" in result.lower() or "contact" in result.lower()
