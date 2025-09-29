# Wand AI - Multi-Agent System & Knowledge Base

A comprehensive AI-powered system featuring multi-agent orchestration and intelligent knowledge base search with enrichment capabilities.

## 🚀 Project Overview

This project consists of two main challenges:

### Challenge 1: Multi-Agent Task Orchestration
An intelligent multi-agent system that can plan, execute, and aggregate complex tasks using specialized AI agents:
- **Planner Agent**: Breaks down complex tasks into manageable subtasks
- **Data Agent**: Performs data analysis and statistical operations
- **Chart Agent**: Creates visualizations and charts
- **Financial Agent**: Handles financial analysis and market research
- **Aggregator Agent**: Combines results from all agents into comprehensive reports

### Challenge 2: AI-Powered Knowledge Base Search & Enrichment
An intelligent document search system with AI-generated answers and automatic enrichment:
- **Document Processing**: Upload and process PDF, DOCX, TXT, and HTML files
- **Vector Search**: Semantic search using ChromaDB and sentence transformers
- **AI Answer Generation**: RAG-powered responses with confidence scoring
- **Auto-Enrichment**: Automatic knowledge base enhancement based on missing information

## 📋 Prerequisites

- Python 3.8+
- pip package manager

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Wand-AI-
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements-test.txt
   ```

3. **Set up environment variables (optional):**
   ```bash
   # Create .env file for API keys
   OPENAI_API_KEY=your_openai_api_key
   SERPAPI_KEY=your_serpapi_key
   ```

## 🧪 Running Tests

### Run All Tests
```bash
python -m pytest
```

### Run Tests with Verbose Output
```bash
python -m pytest -v
```

### Run Tests for Specific Challenge
```bash
# Challenge 1 (Multi-Agent System)
python -m pytest challenge1/tests/ -v

# Challenge 2 (Knowledge Base)
python -m pytest challenge2/tests/ -v
```

### Run Tests with Coverage Report
```bash
python -m pytest --cov=challenge1 --cov=challenge2 --cov-report=html
```

### Run Specific Test Files
```bash
# Test specific agent
python -m pytest challenge1/tests/test_planner_agent.py -v

# Test main application
python -m pytest challenge1/tests/test_main.py -v
python -m pytest challenge2/tests/test_main.py -v
```

### Run Tests with Different Output Formats
```bash
# Short traceback format
python -m pytest --tb=short

# Line-by-line traceback
python -m pytest --tb=line

# Stop on first failure
python -m pytest -x

# Run tests in parallel (if pytest-xdist is installed)
python -m pytest -n auto
```

## 📊 Test Results

The test suite includes **361 comprehensive tests** covering:

### Challenge 1 Tests (200+ tests)
- **Agent Tests**: Individual agent functionality and behavior
- **Integration Tests**: Multi-agent orchestration and communication
- **API Tests**: FastAPI endpoints and WebSocket connections
- **Model Tests**: Data validation and serialization
- **Tool Tests**: Python execution, web search, and data analysis

### Challenge 2 Tests (160+ tests)
- **Knowledge Base Tests**: Document processing and vector storage
- **RAG Pipeline Tests**: AI answer generation and completeness checking
- **Enrichment Service Tests**: Auto-enrichment and web search
- **API Tests**: FastAPI endpoints for document management
- **Model Tests**: Pydantic model validation

## 🏃‍♂️ Running the Applications

### Challenge 1: Multi-Agent System
```bash
cd challenge1
python main.py
```
Access the API at: `http://localhost:8000`

### Challenge 2: Knowledge Base System
```bash
cd challenge2
python main.py
```
Access the API at: `http://localhost:8000`

## 📁 Project Structure

```
Wand-AI-/
├── challenge1/                    # Multi-Agent System
│   ├── agents/                   # Individual AI agents
│   │   ├── base_agent.py
│   │   ├── planner_agent.py
│   │   ├── data_agent.py
│   │   ├── chart_agent.py
│   │   ├── financial_agent.py
│   │   └── aggregator_agent.py
│   ├── tests/                    # Challenge 1 tests
│   ├── main.py                   # FastAPI application
│   ├── orchestrator.py           # Task orchestration
│   ├── models.py                 # Data models
│   └── config.py                 # Configuration
├── challenge2/                   # Knowledge Base System
│   ├── data/                     # Document storage
│   │   ├── chromadb/            # Vector database
│   │   └── documents/           # Uploaded documents
│   ├── tests/                    # Challenge 2 tests
│   ├── main.py                   # FastAPI application
│   ├── knowledge_base.py         # Document processing
│   ├── rag_pipeline.py          # RAG implementation
│   ├── enrichment_service.py    # Auto-enrichment
│   ├── models.py                 # Data models
│   └── config.py                 # Configuration
├── data/                         # Shared data directory
├── conftest.py                   # Pytest configuration
├── pytest.ini                   # Test configuration
├── requirements-test.txt         # Dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Test Configuration (pytest.ini)
- **Test Paths**: `challenge1/tests` and `challenge2/tests`
- **Coverage**: 100% coverage requirement
- **Markers**: Unit, integration, API, agent, RAG, and knowledge base tests
- **Output**: Verbose with HTML coverage reports

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key for LLM functionality
- `SERPAPI_KEY`: SerpAPI key for web search enrichment
- `CHROMA_PERSIST_DIR`: ChromaDB storage directory
- `DEBUG`: Enable debug mode

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements-test.txt
   ```

2. **ChromaDB Issues**: Clear the database if corrupted
   ```bash
   rm -rf data/chromadb/
   ```

3. **API Key Issues**: Set environment variables or use demo mode
   ```bash
   export OPENAI_API_KEY=your_key
   export SERPAPI_KEY=your_key
   ```

4. **Test Failures**: Run with verbose output to see detailed errors
   ```bash
   python -m pytest -v --tb=long
   ```

### Performance Tips

- Use `-x` flag to stop on first failure during development
- Use `--tb=short` for faster output
- Run specific test files during development
- Use `--cov` to identify untested code

## 📈 Test Coverage

The project maintains **100% test coverage** across all modules:
- All functions and methods are tested
- Edge cases and error conditions are covered
- Integration between components is verified
- API endpoints are thoroughly tested

## 🤝 Contributing

1. Run tests before making changes: `python -m pytest`
2. Ensure all tests pass: `python -m pytest --cov-fail-under=100`
3. Add tests for new functionality
4. Update documentation as needed

## 📝 License

This project is part of the Wand AI challenge and is intended for educational and demonstration purposes.

---

**Total Tests**: 361 ✅  
**Coverage**: 100% 📊  
**Status**: All tests passing 🚀
