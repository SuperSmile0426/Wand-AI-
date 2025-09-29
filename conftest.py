"""
Global pytest configuration and fixtures for both challenge projects.
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import json

# Set up test environment
os.environ["TESTING"] = "true"
os.environ["OPENAI_API_KEY"] = "test-key"
os.environ["SERPAPI_KEY"] = "test-serpapi-key"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "Mock response from OpenAI"
                }
            }
        ]
    }

@pytest.fixture
def mock_agent_response():
    """Mock agent response."""
    return {
        "agent_type": "test_agent",
        "task_id": "test_task",
        "status": "completed",
        "result": {"test": "data"},
        "progress": 100
    }

@pytest.fixture
def sample_document_content():
    """Sample document content for testing."""
    return {
        "pdf": b"Sample PDF content for testing",
        "docx": b"Sample DOCX content for testing",
        "txt": b"Sample text content for testing",
        "html": b"<html><body>Sample HTML content for testing</body></html>"
    }

@pytest.fixture
def sample_financial_data():
    """Sample financial data for testing."""
    return {
        "quarters": ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"],
        "revenue": [1250000, 1380000, 1450000, 1520000],
        "profit": [150000, 180000, 195000, 210000],
        "expenses": [1100000, 1200000, 1255000, 1310000]
    }

@pytest.fixture
def sample_search_request():
    """Sample search request for testing."""
    return {
        "query": "What is the company's revenue growth?",
        "include_enrichment": True,
        "max_results": 5
    }

@pytest.fixture
def sample_missing_info():
    """Sample missing information for testing."""
    return [
        {
            "topic": "Financial Performance",
            "description": "Detailed financial metrics and KPIs",
            "importance": "high"
        },
        {
            "topic": "Market Analysis",
            "description": "Competitive landscape and market trends",
            "importance": "medium"
        }
    ]

@pytest.fixture
def mock_web_search_results():
    """Mock web search results."""
    return [
        {
            "title": "Test Search Result 1",
            "url": "https://example.com/result1",
            "snippet": "This is a test search result snippet"
        },
        {
            "title": "Test Search Result 2", 
            "url": "https://example.com/result2",
            "snippet": "Another test search result snippet"
        }
    ]

@pytest.fixture
def mock_chromadb_collection():
    """Mock ChromaDB collection."""
    collection = Mock()
    collection.add = Mock()
    collection.query = Mock(return_value={
        "documents": [["Sample document content"]],
        "metadatas": [[{"document_id": "test_doc", "filename": "test.pdf"}]],
        "distances": [[0.1]]
    })
    collection.get = Mock(return_value={
        "ids": ["test_chunk_1"],
        "documents": [["Sample chunk content"]],
        "metadatas": [{"document_id": "test_doc", "filename": "test.pdf"}]
    })
    collection.delete = Mock()
    return collection

@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for embeddings."""
    model = Mock()
    model.encode = Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    return model

@pytest.fixture
def mock_async_openai_client():
    """Mock async OpenAI client."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=Mock(
        choices=[Mock(message=Mock(content="Mock AI response"))]
    ))
    return client

@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp session for web requests."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.text = AsyncMock(return_value="<html>Mock web content</html>")
    response.json = AsyncMock(return_value={"organic_results": []})
    session.get = AsyncMock(return_value=response)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=None)
    return session

@pytest.fixture
def mock_document_processor():
    """Mock document processor."""
    processor = Mock()
    processor.extract_text = AsyncMock(return_value="Extracted text content")
    processor.chunk_text = Mock(return_value=["Chunk 1", "Chunk 2"])
    return processor

@pytest.fixture
def sample_task_execution():
    """Sample task execution for testing."""
    return {
        "session_id": "test_session_123",
        "main_request": "Analyze quarterly financial performance",
        "subtasks": [
            {
                "id": "financial_analysis",
                "agent_type": "financial_analyst",
                "description": "Perform financial analysis",
                "dependencies": []
            }
        ],
        "status": "in_progress",
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00"
    }

@pytest.fixture
def mock_progress_callback():
    """Mock progress callback for testing."""
    return AsyncMock()

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    websocket = Mock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock(return_value="test message")
    return websocket

@pytest.fixture
def mock_connection_manager():
    """Mock WebSocket connection manager."""
    manager = Mock()
    manager.connect = AsyncMock()
    manager.disconnect = Mock()
    manager.send_personal_message = AsyncMock()
    manager.broadcast = AsyncMock()
    manager.active_connections = []
    return manager

# Test data fixtures
@pytest.fixture
def test_documents():
    """Test documents for knowledge base testing."""
    return [
        {
            "filename": "test1.pdf",
            "content": b"Sample PDF content about financial performance",
            "content_type": "application/pdf"
        },
        {
            "filename": "test2.txt",
            "content": b"Sample text content about market analysis",
            "content_type": "text/plain"
        }
    ]

@pytest.fixture
def test_queries():
    """Test queries for search testing."""
    return [
        "What is the company's revenue?",
        "How has the market performed?",
        "What are the key financial metrics?",
        "Tell me about the competitive landscape"
    ]

@pytest.fixture
def test_agent_types():
    """Test agent types for multi-agent testing."""
    return [
        "planner",
        "financial_analyst", 
        "data_analyst",
        "chart_generator",
        "aggregator"
    ]

@pytest.fixture
def test_task_statuses():
    """Test task statuses."""
    return [
        "pending",
        "in_progress", 
        "completed",
        "failed"
    ]

# Mock configurations
@pytest.fixture(autouse=True)
def mock_config():
    """Mock configuration for testing."""
    with patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "SERPAPI_KEY": "test-serpapi-key",
        "CHROMA_PERSIST_DIR": "./test_data/chromadb",
        "DOCUMENTS_DIR": "./test_data/documents",
        "TESTING": "true"
    }):
        yield

# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Clean up test data after each test."""
    yield
    # Cleanup logic can be added here if needed
    pass
