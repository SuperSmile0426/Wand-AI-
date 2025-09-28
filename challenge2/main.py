"""
Challenge 2: AI-Powered Knowledge Base Search & Enrichment
Main FastAPI application for document upload, search, and enrichment suggestions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List, Optional
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from models import SearchRequest, SearchResponse, DocumentInfo, EnrichmentSuggestion
from knowledge_base import KnowledgeBase
from rag_pipeline import RAGPipeline
from enrichment_service import EnrichmentService
from config import Config

app = FastAPI(title="AI Knowledge Base Search & Enrichment", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Validate configuration
Config.validate()

# Initialize services
knowledge_base = KnowledgeBase(persist_directory=Config.CHROMA_PERSIST_DIR)
rag_pipeline = RAGPipeline(openai_api_key=Config.OPENAI_API_KEY)
enrichment_service = EnrichmentService(serpapi_key=Config.SERPAPI_KEY)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web interface."""
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload multiple documents to the knowledge base."""
    try:
        uploaded_docs = []
        for file in files:
            if not file.filename:
                continue
                
            # Read file content
            content = await file.read()
            
            # Process and store document
            doc_info = await knowledge_base.add_document(
                filename=file.filename,
                content=content,
                content_type=file.content_type
            )
            uploaded_docs.append(doc_info)
        
        return {
            "message": f"Successfully uploaded {len(uploaded_docs)} documents",
            "documents": uploaded_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    try:
        documents = await knowledge_base.list_documents()
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search documents and get AI-generated answers with enrichment suggestions."""
    try:
        # Perform RAG search
        result = await rag_pipeline.search(
            query=request.query,
            knowledge_base=knowledge_base,
            include_enrichment=request.include_enrichment
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/rate_answer")
async def rate_answer(
    query: str = Form(...),
    answer: str = Form(...),
    rating: int = Form(...),
    feedback: Optional[str] = Form(None)
):
    """Rate the quality of an answer to improve the pipeline."""
    try:
        await rag_pipeline.add_rating(
            query=query,
            answer=answer,
            rating=rating,
            feedback=feedback
        )
        return {"message": "Rating recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to record rating: {str(e)}")

@app.post("/enrich")
async def enrich_knowledge_base(suggestion: EnrichmentSuggestion):
    """Manually trigger enrichment based on suggestions."""
    try:
        result = await enrichment_service.enrich_from_suggestion(suggestion)
        return {"message": "Enrichment completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enrichment failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "knowledge-base-search"}

@app.post("/auto_enrich")
async def auto_enrich(
    query: str = Form(...),
    missing_info: str = Form(...)
):
    """Trigger auto-enrichment based on missing information."""
    try:
        import json
        missing_info_list = json.loads(missing_info)
        missing_objects = [MissingInfo(**info) for info in missing_info_list]
        
        result = await enrichment_service.auto_enrich(
            knowledge_base=knowledge_base,
            query=query,
            missing_info=missing_objects
        )
        
        return {"message": "Auto-enrichment completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-enrichment failed: {str(e)}")

@app.get("/config")
async def get_config():
    """Get current configuration (excluding sensitive data)."""
    return {
        "demo_mode": Config.is_demo_mode(),
        "auto_enrichment_enabled": Config.AUTO_ENRICHMENT_ENABLED,
        "max_file_size": Config.MAX_FILE_SIZE,
        "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
        "chunk_size": Config.CHUNK_SIZE,
        "max_search_results": Config.MAX_SEARCH_RESULTS
    }

if __name__ == "__main__":
    uvicorn.run(app, host=Config.HOST, port=Config.PORT, reload=Config.DEBUG)
