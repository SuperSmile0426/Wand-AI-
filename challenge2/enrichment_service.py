"""
Enrichment service for automatically fetching missing data and suggesting improvements.
Includes web search, API fetching, and document enrichment capabilities.
"""

import os
import asyncio
import aiohttp
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from urllib.parse import quote_plus

from challenge2.models import EnrichmentSuggestion, EnrichmentResult, DocumentInfo, DocumentType, MissingInfo, EnrichmentType
from challenge2.knowledge_base import KnowledgeBase

class EnrichmentService:
    """Service for enriching the knowledge base with additional information."""
    
    def __init__(self, serpapi_key: Optional[str] = None):
        """Initialize the enrichment service."""
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        self.web_search_enabled = bool(self.serpapi_key)
        
        # Trusted sources for auto-enrichment
        self.trusted_sources = [
            "wikipedia.org",
            "github.com",
            "stackoverflow.com",
            "docs.python.org",
            "developer.mozilla.org"
        ]
    
    async def enrich_from_suggestion(self, suggestion: EnrichmentSuggestion) -> EnrichmentResult:
        """Enrich the knowledge base based on a suggestion."""
        try:
            if suggestion.type.value == "web_search" and self.web_search_enabled:
                return await self._enrich_from_web_search(suggestion)
            elif suggestion.type.value == "api_fetch":
                return await self._enrich_from_api(suggestion)
            elif suggestion.type.value == "document":
                return await self._enrich_from_document_suggestion(suggestion)
            else:
                return EnrichmentResult(
                    success=False,
                    new_documents=[],
                    enriched_topics=[],
                    sources=[],
                    timestamp=datetime.now()
                )
        except Exception as e:
            return EnrichmentResult(
                success=False,
                new_documents=[],
                enriched_topics=[],
                sources=[f"Error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def _enrich_from_web_search(self, suggestion: EnrichmentSuggestion) -> EnrichmentResult:
        """Enrich by searching the web for missing information."""
        try:
            # Extract search terms from missing info
            search_terms = []
            for missing_info in suggestion.missing_info:
                search_terms.append(missing_info.topic)
            
            # Perform web search
            search_query = " ".join(search_terms)
            search_results = await self._perform_web_search(search_query)
            
            # Process results and create documents
            new_documents = []
            enriched_topics = []
            sources = []
            
            for result in search_results[:3]:  # Limit to top 3 results
                if self._is_trusted_source(result.get("url", "")):
                    # Create document from web content
                    doc_content = await self._fetch_web_content(result["url"])
                    if doc_content:
                        # Create a mock document info
                        doc_info = DocumentInfo(
                            id=f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(new_documents)}",
                            filename=f"web_search_{result.get('title', 'result')[:50]}.html",
                            content_type="text/html",
                            document_type=DocumentType.HTML,
                            upload_date=datetime.now(),
                            size_bytes=len(doc_content.encode('utf-8')),
                            chunk_count=1,
                            metadata={
                                "source_url": result["url"],
                                "search_query": search_query,
                                "enrichment_type": "web_search"
                            }
                        )
                        new_documents.append(doc_info)
                        enriched_topics.append(search_query)
                        sources.append(result["url"])
            
            return EnrichmentResult(
                success=len(new_documents) > 0,
                new_documents=new_documents,
                enriched_topics=enriched_topics,
                sources=sources,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return EnrichmentResult(
                success=False,
                new_documents=[],
                enriched_topics=[],
                sources=[f"Web search error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    async def _enrich_from_api(self, suggestion: EnrichmentSuggestion) -> EnrichmentResult:
        """Enrich by fetching data from APIs."""
        # This would implement API-based enrichment
        # For now, return a placeholder
        return EnrichmentResult(
            success=False,
            new_documents=[],
            enriched_topics=[],
            sources=["API enrichment not implemented yet"],
            timestamp=datetime.now()
        )
    
    async def _enrich_from_document_suggestion(self, suggestion: EnrichmentSuggestion) -> EnrichmentResult:
        """Provide suggestions for document-based enrichment."""
        return EnrichmentResult(
            success=True,
            new_documents=[],
            enriched_topics=[info.topic for info in suggestion.missing_info],
            sources=["Document upload suggestion"],
            timestamp=datetime.now()
        )
    
    async def _perform_web_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform web search using SerpAPI."""
        if not self.web_search_enabled:
            # Return mock results for demonstration
            return [
                {
                    "title": f"Search results for: {query}",
                    "url": "https://example.com/mock-result-1",
                    "snippet": f"This is a mock search result for '{query}'. In a real implementation, this would be actual search results."
                },
                {
                    "title": f"Additional information about: {query}",
                    "url": "https://example.com/mock-result-2", 
                    "snippet": f"More mock content related to '{query}' that would help enrich the knowledge base."
                }
            ]
        
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    "q": query,
                    "api_key": self.serpapi_key,
                    "engine": "google",
                    "num": 5
                }
                
                async with session.get("https://serpapi.com/search", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = []
                        
                        for result in data.get("organic_results", []):
                            results.append({
                                "title": result.get("title", ""),
                                "url": result.get("link", ""),
                                "snippet": result.get("snippet", "")
                            })
                        
                        return results
                    else:
                        return []
        except Exception as e:
            print(f"Web search error: {e}")
            return []
    
    async def _fetch_web_content(self, url: str) -> Optional[str]:
        """Fetch content from a web URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Simple text extraction (in production, use proper HTML parsing)
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(content, 'html.parser')
                        return soup.get_text()[:5000]  # Limit content length
                    return None
        except Exception as e:
            print(f"Error fetching content from {url}: {e}")
            return None
    
    def _is_trusted_source(self, url: str) -> bool:
        """Check if a URL is from a trusted source."""
        for trusted_domain in self.trusted_sources:
            if trusted_domain in url.lower():
                return True
        return False
    
    async def suggest_enrichment_strategies(self, missing_info: List[MissingInfo]) -> List[EnrichmentSuggestion]:
        """Suggest enrichment strategies based on missing information."""
        suggestions = []
        
        for info in missing_info:
            if info.importance == "high":
                # High importance - suggest multiple strategies
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.WEB_SEARCH,
                    description=f"Search the web for recent information about {info.topic}",
                    missing_info=[info],
                    suggested_actions=[
                        f"Search for '{info.topic}' on trusted sources",
                        "Verify information accuracy",
                        "Add to knowledge base"
                    ],
                    confidence=0.8,
                    auto_enrichable=self.web_search_enabled
                ))
                
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.DOCUMENT,
                    description=f"Upload documents specifically about {info.topic}",
                    missing_info=[info],
                    suggested_actions=[
                        f"Find and upload documents related to {info.topic}",
                        "Consider official documentation or research papers",
                        "Include multiple perspectives on the topic"
                    ],
                    confidence=0.9,
                    auto_enrichable=False
                ))
            
            elif info.importance == "medium":
                # Medium importance - suggest document upload
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.DOCUMENT,
                    description=f"Add more information about {info.topic}",
                    missing_info=[info],
                    suggested_actions=[
                        f"Look for additional documents covering {info.topic}",
                        "Consider user-provided information"
                    ],
                    confidence=0.6,
                    auto_enrichable=False
                ))
        
        return suggestions
    
    async def auto_enrich(self, knowledge_base: KnowledgeBase, query: str, missing_info: List[MissingInfo]) -> EnrichmentResult:
        """Automatically enrich the knowledge base based on missing information."""
        if not missing_info:
            return EnrichmentResult(
                success=True,
                new_documents=[],
                enriched_topics=[],
                sources=[],
                timestamp=datetime.now()
            )
        
        # Get enrichment suggestions
        suggestions = await self.suggest_enrichment_strategies(missing_info)
        
        # Filter for auto-enrichable suggestions
        auto_suggestions = [s for s in suggestions if s.auto_enrichable]
        
        if not auto_suggestions:
            return EnrichmentResult(
                success=False,
                new_documents=[],
                enriched_topics=[],
                sources=["No auto-enrichable suggestions available"],
                timestamp=datetime.now()
            )
        
        # Execute auto-enrichment
        all_results = []
        for suggestion in auto_suggestions:
            result = await self.enrich_from_suggestion(suggestion)
            all_results.append(result)
        
        # Combine results
        combined_result = EnrichmentResult(
            success=any(r.success for r in all_results),
            new_documents=[],
            enriched_topics=[],
            sources=[],
            timestamp=datetime.now()
        )
        
        for result in all_results:
            combined_result.new_documents.extend(result.new_documents)
            combined_result.enriched_topics.extend(result.enriched_topics)
            combined_result.sources.extend(result.sources)
        
        return combined_result
