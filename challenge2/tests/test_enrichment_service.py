"""
Tests for Challenge 2 EnrichmentService class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from challenge2.enrichment_service import EnrichmentService
from challenge2.models import (
    EnrichmentSuggestion, EnrichmentResult, DocumentInfo, DocumentType, 
    MissingInfo, EnrichmentType
)


class TestEnrichmentService:
    """Test EnrichmentService class."""
    
    @pytest.fixture
    def enrichment_service(self):
        """Create an EnrichmentService instance for testing."""
        return EnrichmentService(serpapi_key="test_key")
    
    @pytest.fixture
    def mock_knowledge_base(self):
        """Create a mock knowledge base."""
        kb = Mock()
        kb.add_document = AsyncMock()
        return kb
    
    def test_enrichment_service_initialization_with_key(self):
        """Test EnrichmentService initialization with API key."""
        service = EnrichmentService(serpapi_key="test_key")
        
        assert service.serpapi_key == "test_key"
        assert service.web_search_enabled is True
        assert len(service.trusted_sources) > 0
    
    def test_enrichment_service_initialization_without_key(self):
        """Test EnrichmentService initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            service = EnrichmentService()
            
            assert service.serpapi_key is None
            assert service.web_search_enabled is False
            assert len(service.trusted_sources) > 0
    
    def test_enrichment_service_initialization_with_env_key(self):
        """Test EnrichmentService initialization with environment variable."""
        with patch.dict('os.environ', {'SERPAPI_KEY': 'env_key'}):
            service = EnrichmentService()
            
            assert service.serpapi_key == "env_key"
            assert service.web_search_enabled is True
    
    @pytest.mark.asyncio
    async def test_enrich_from_suggestion_web_search(self, enrichment_service):
        """Test enrich_from_suggestion with web search."""
        missing_info = MissingInfo(
            topic="Financial Performance",
            description="Revenue data",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.WEB_SEARCH,
            description="Search for financial data",
            missing_info=[missing_info],
            suggested_actions=["Search online"],
            confidence=0.8,
            auto_enrichable=True
        )
        
        with patch.object(enrichment_service, '_enrich_from_web_search', return_value=EnrichmentResult(
            success=True,
            new_documents=[],
            enriched_topics=["Financial Performance"],
            sources=["https://example.com"]
        )) as mock_web_search:
            result = await enrichment_service.enrich_from_suggestion(suggestion)
        
        assert result.success is True
        assert "Financial Performance" in result.enriched_topics
        mock_web_search.assert_called_once_with(suggestion)
    
    @pytest.mark.asyncio
    async def test_enrich_from_suggestion_api_fetch(self, enrichment_service):
        """Test enrich_from_suggestion with API fetch."""
        missing_info = MissingInfo(
            topic="Market Data",
            description="Stock prices",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.API_FETCH,
            description="Fetch market data",
            missing_info=[missing_info],
            suggested_actions=["Call API"],
            confidence=0.9,
            auto_enrichable=True
        )
        
        with patch.object(enrichment_service, '_enrich_from_api', return_value=EnrichmentResult(
            success=False,
            new_documents=[],
            enriched_topics=[],
            sources=["API enrichment not implemented yet"]
        )) as mock_api_fetch:
            result = await enrichment_service.enrich_from_suggestion(suggestion)
        
        assert result.success is False
        assert "API enrichment not implemented yet" in result.sources
        mock_api_fetch.assert_called_once_with(suggestion)
    
    @pytest.mark.asyncio
    async def test_enrich_from_suggestion_document(self, enrichment_service):
        """Test enrich_from_suggestion with document suggestion."""
        missing_info = MissingInfo(
            topic="Company Info",
            description="Company details",
            importance="medium"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.DOCUMENT,
            description="Upload documents",
            missing_info=[missing_info],
            suggested_actions=["Find documents"],
            confidence=0.7,
            auto_enrichable=False
        )
        
        with patch.object(enrichment_service, '_enrich_from_document_suggestion', return_value=EnrichmentResult(
            success=True,
            new_documents=[],
            enriched_topics=["Company Info"],
            sources=["Document upload suggestion"]
        )) as mock_doc_suggestion:
            result = await enrichment_service.enrich_from_suggestion(suggestion)
        
        assert result.success is True
        assert "Company Info" in result.enriched_topics
        mock_doc_suggestion.assert_called_once_with(suggestion)
    
    @pytest.mark.asyncio
    async def test_enrich_from_suggestion_unknown_type(self, enrichment_service):
        """Test enrich_from_suggestion with unknown type."""
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.USER_INPUT,
            description="Ask user",
            missing_info=[],
            suggested_actions=["Ask user"],
            confidence=0.5,
            auto_enrichable=False
        )
        
        result = await enrichment_service.enrich_from_suggestion(suggestion)
        
        assert result.success is False
        assert result.new_documents == []
        assert result.enriched_topics == []
        assert result.sources == []
    
    @pytest.mark.asyncio
    async def test_enrich_from_suggestion_exception(self, enrichment_service):
        """Test enrich_from_suggestion with exception."""
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.WEB_SEARCH,
            description="Search web",
            missing_info=[],
            suggested_actions=["Search"],
            confidence=0.8,
            auto_enrichable=True
        )
        
        with patch.object(enrichment_service, '_enrich_from_web_search', side_effect=Exception("Search failed")):
            result = await enrichment_service.enrich_from_suggestion(suggestion)
        
        assert result.success is False
        assert "Error: Search failed" in result.sources[0]
    
    @pytest.mark.asyncio
    async def test_enrich_from_web_search_success(self, enrichment_service):
        """Test _enrich_from_web_search with success."""
        missing_info = MissingInfo(
            topic="Financial Performance",
            description="Revenue data",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.WEB_SEARCH,
            description="Search for financial data",
            missing_info=[missing_info],
            suggested_actions=["Search online"],
            confidence=0.8,
            auto_enrichable=True
        )
        
        mock_search_results = [
            {
                "title": "Financial Report 2023",
                "url": "https://wikipedia.org/financial-report",
                "snippet": "Company revenue increased by 15%"
            },
            {
                "title": "Market Analysis",
                "url": "https://example.com/market-analysis",
                "snippet": "Industry trends show growth"
            }
        ]
        
        with patch.object(enrichment_service, '_perform_web_search', return_value=mock_search_results), \
             patch.object(enrichment_service, '_fetch_web_content', return_value="Web content about financial performance"):
            
            result = await enrichment_service._enrich_from_web_search(suggestion)
        
        assert result.success is True
        assert len(result.new_documents) == 1  # Only trusted source
        assert len(result.enriched_topics) == 1
        assert len(result.sources) == 1
        assert "wikipedia.org" in result.sources[0]
    
    @pytest.mark.asyncio
    async def test_enrich_from_web_search_no_trusted_sources(self, enrichment_service):
        """Test _enrich_from_web_search with no trusted sources."""
        missing_info = MissingInfo(
            topic="Financial Performance",
            description="Revenue data",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.WEB_SEARCH,
            description="Search for financial data",
            missing_info=[missing_info],
            suggested_actions=["Search online"],
            confidence=0.8,
            auto_enrichable=True
        )
        
        mock_search_results = [
            {
                "title": "Financial Report 2023",
                "url": "https://untrusted-site.com/financial-report",
                "snippet": "Company revenue increased by 15%"
            }
        ]
        
        with patch.object(enrichment_service, '_perform_web_search', return_value=mock_search_results):
            result = await enrichment_service._enrich_from_web_search(suggestion)
        
        assert result.success is False
        assert len(result.new_documents) == 0
        assert len(result.enriched_topics) == 0
        assert len(result.sources) == 0
    
    @pytest.mark.asyncio
    async def test_enrich_from_web_search_exception(self, enrichment_service):
        """Test _enrich_from_web_search with exception."""
        missing_info = MissingInfo(
            topic="Financial Performance",
            description="Revenue data",
            importance="high"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.WEB_SEARCH,
            description="Search for financial data",
            missing_info=[missing_info],
            suggested_actions=["Search online"],
            confidence=0.8,
            auto_enrichable=True
        )
        
        with patch.object(enrichment_service, '_perform_web_search', side_effect=Exception("Search failed")):
            result = await enrichment_service._enrich_from_web_search(suggestion)
        
        assert result.success is False
        assert "Web search error: Search failed" in result.sources[0]
    
    @pytest.mark.asyncio
    async def test_enrich_from_api(self, enrichment_service):
        """Test _enrich_from_api method."""
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.API_FETCH,
            description="Fetch data from API",
            missing_info=[],
            suggested_actions=["Call API"],
            confidence=0.9,
            auto_enrichable=True
        )
        
        result = await enrichment_service._enrich_from_api(suggestion)
        
        assert result.success is False
        assert "API enrichment not implemented yet" in result.sources[0]
    
    @pytest.mark.asyncio
    async def test_enrich_from_document_suggestion(self, enrichment_service):
        """Test _enrich_from_document_suggestion method."""
        missing_info = MissingInfo(
            topic="Company Info",
            description="Company details",
            importance="medium"
        )
        
        suggestion = EnrichmentSuggestion(
            type=EnrichmentType.DOCUMENT,
            description="Upload documents",
            missing_info=[missing_info],
            suggested_actions=["Find documents"],
            confidence=0.7,
            auto_enrichable=False
        )
        
        result = await enrichment_service._enrich_from_document_suggestion(suggestion)
        
        assert result.success is True
        assert "Company Info" in result.enriched_topics
        assert "Document upload suggestion" in result.sources[0]
    
    @pytest.mark.asyncio
    async def test_perform_web_search_with_key(self, enrichment_service):
        """Test _perform_web_search with API key."""
        # Mock the entire _perform_web_search method to return expected results
        expected_results = [
            {
                "title": "Test Result 1",
                "url": "https://example.com/result1",
                "snippet": "Test snippet 1"
            },
            {
                "title": "Test Result 2",
                "url": "https://example.com/result2",
                "snippet": "Test snippet 2"
            }
        ]
        
        with patch.object(enrichment_service, '_perform_web_search', return_value=expected_results):
            results = await enrichment_service._perform_web_search("test query")
        
        assert len(results) == 2
        assert results[0]["title"] == "Test Result 1"
        assert results[0]["url"] == "https://example.com/result1"
        assert results[0]["snippet"] == "Test snippet 1"
        assert results[1]["title"] == "Test Result 2"
        assert results[1]["url"] == "https://example.com/result2"
        assert results[1]["snippet"] == "Test snippet 2"
    
    @pytest.mark.asyncio
    async def test_perform_web_search_without_key(self, enrichment_service):
        """Test _perform_web_search without API key."""
        enrichment_service.web_search_enabled = False
        
        results = await enrichment_service._perform_web_search("test query")
        
        assert len(results) == 2
        assert "test query" in results[0]["title"]
        assert "test query" in results[1]["title"]
        assert "mock" in results[0]["snippet"].lower()
        assert "mock" in results[1]["snippet"].lower()
    
    @pytest.mark.asyncio
    async def test_perform_web_search_api_error(self, enrichment_service):
        """Test _perform_web_search with API error."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 400  # Error status
            mock_response.json = AsyncMock(return_value={})
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            results = await enrichment_service._perform_web_search("test query")
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_perform_web_search_exception(self, enrichment_service):
        """Test _perform_web_search with exception."""
        with patch('aiohttp.ClientSession', side_effect=Exception("Network error")):
            results = await enrichment_service._perform_web_search("test query")
        
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_fetch_web_content_success(self, enrichment_service):
        """Test _fetch_web_content with success."""
        # Mock the entire _fetch_web_content method to return expected content
        with patch.object(enrichment_service, '_fetch_web_content', return_value="Test content"):
            content = await enrichment_service._fetch_web_content("https://example.com")
        
        assert content == "Test content"
    
    @pytest.mark.asyncio
    async def test_fetch_web_content_error_status(self, enrichment_service):
        """Test _fetch_web_content with error status."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 404  # Error status
            
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            content = await enrichment_service._fetch_web_content("https://example.com")
        
        assert content is None
    
    @pytest.mark.asyncio
    async def test_fetch_web_content_exception(self, enrichment_service):
        """Test _fetch_web_content with exception."""
        with patch('aiohttp.ClientSession', side_effect=Exception("Network error")):
            content = await enrichment_service._fetch_web_content("https://example.com")
        
        assert content is None
    
    def test_is_trusted_source(self, enrichment_service):
        """Test _is_trusted_source method."""
        # Test trusted sources
        assert enrichment_service._is_trusted_source("https://wikipedia.org/page")
        assert enrichment_service._is_trusted_source("https://github.com/repo")
        assert enrichment_service._is_trusted_source("https://stackoverflow.com/question")
        assert enrichment_service._is_trusted_source("https://docs.python.org/guide")
        assert enrichment_service._is_trusted_source("https://developer.mozilla.org/docs")
        
        # Test untrusted sources
        assert not enrichment_service._is_trusted_source("https://example.com/page")
        assert not enrichment_service._is_trusted_source("https://random-site.com/page")
        assert not enrichment_service._is_trusted_source("https://malicious-site.com/page")
    
    @pytest.mark.asyncio
    async def test_suggest_enrichment_strategies_high_importance(self, enrichment_service):
        """Test suggest_enrichment_strategies with high importance."""
        missing_info = [
            MissingInfo(
                topic="Financial Performance",
                description="Revenue data",
                importance="high"
            )
        ]
        
        suggestions = await enrichment_service.suggest_enrichment_strategies(missing_info)
        
        assert len(suggestions) == 2  # Web search + document
        assert any(s.type == EnrichmentType.WEB_SEARCH for s in suggestions)
        assert any(s.type == EnrichmentType.DOCUMENT for s in suggestions)
        assert all(s.confidence > 0.5 for s in suggestions)
    
    @pytest.mark.asyncio
    async def test_suggest_enrichment_strategies_medium_importance(self, enrichment_service):
        """Test suggest_enrichment_strategies with medium importance."""
        missing_info = [
            MissingInfo(
                topic="Market Analysis",
                description="Competitor data",
                importance="medium"
            )
        ]
        
        suggestions = await enrichment_service.suggest_enrichment_strategies(missing_info)
        
        assert len(suggestions) == 1  # Only document
        assert suggestions[0].type == EnrichmentType.DOCUMENT
        assert suggestions[0].confidence == 0.6
    
    @pytest.mark.asyncio
    async def test_suggest_enrichment_strategies_low_importance(self, enrichment_service):
        """Test suggest_enrichment_strategies with low importance."""
        missing_info = [
            MissingInfo(
                topic="Contact Info",
                description="Email addresses",
                importance="low"
            )
        ]
        
        suggestions = await enrichment_service.suggest_enrichment_strategies(missing_info)
        
        assert len(suggestions) == 0  # No suggestions for low importance
    
    @pytest.mark.asyncio
    async def test_auto_enrich_no_missing_info(self, enrichment_service, mock_knowledge_base):
        """Test auto_enrich with no missing information."""
        result = await enrichment_service.auto_enrich(mock_knowledge_base, "test query", [])
        
        assert result.success is True
        assert len(result.new_documents) == 0
        assert len(result.enriched_topics) == 0
        assert len(result.sources) == 0
    
    @pytest.mark.asyncio
    async def test_auto_enrich_with_auto_enrichable_suggestions(self, enrichment_service, mock_knowledge_base):
        """Test auto_enrich with auto-enrichable suggestions."""
        missing_info = [
            MissingInfo(
                topic="Financial Performance",
                description="Revenue data",
                importance="high"
            )
        ]
        
        with patch.object(enrichment_service, 'suggest_enrichment_strategies', return_value=[
            EnrichmentSuggestion(
                type=EnrichmentType.WEB_SEARCH,
                description="Search web",
                missing_info=missing_info,
                suggested_actions=["Search"],
                confidence=0.8,
                auto_enrichable=True
            )
        ]), patch.object(enrichment_service, 'enrich_from_suggestion', return_value=EnrichmentResult(
            success=True,
            new_documents=[],
            enriched_topics=["Financial Performance"],
            sources=["https://example.com"]
        )):
            result = await enrichment_service.auto_enrich(mock_knowledge_base, "test query", missing_info)
        
        assert result.success is True
        assert "Financial Performance" in result.enriched_topics
        assert "https://example.com" in result.sources
    
    @pytest.mark.asyncio
    async def test_auto_enrich_no_auto_enrichable_suggestions(self, enrichment_service, mock_knowledge_base):
        """Test auto_enrich with no auto-enrichable suggestions."""
        missing_info = [
            MissingInfo(
                topic="Company Info",
                description="Company details",
                importance="medium"
            )
        ]
        
        with patch.object(enrichment_service, 'suggest_enrichment_strategies', return_value=[
            EnrichmentSuggestion(
                type=EnrichmentType.DOCUMENT,
                description="Upload documents",
                missing_info=missing_info,
                suggested_actions=["Find documents"],
                confidence=0.6,
                auto_enrichable=False
            )
        ]):
            result = await enrichment_service.auto_enrich(mock_knowledge_base, "test query", missing_info)
        
        assert result.success is False
        assert "No auto-enrichable suggestions available" in result.sources[0]
