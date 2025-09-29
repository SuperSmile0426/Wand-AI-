"""
Knowledge Base system for document storage, processing, and retrieval.
Handles document upload, text extraction, chunking, and vector storage.
"""

import os
import uuid
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import aiofiles

# Document processing
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import re

# Vector storage and embeddings
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

from models import DocumentInfo, DocumentType, MissingInfo, EnrichmentSuggestion, EnrichmentType
from config import Config

class KnowledgeBase:
    """Manages document storage, processing, and retrieval."""
    
    def __init__(self, persist_directory: str = "./data/chromadb"):
        """Initialize the knowledge base with vector storage."""
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.chroma_client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        
        # Document storage
        self.documents_dir = Config.DOCUMENTS_DIR
        os.makedirs(self.documents_dir, exist_ok=True)
        
        # Document metadata cache
        self.document_metadata: Dict[str, DocumentInfo] = {}
        
        # Load existing document metadata from ChromaDB
        self._load_document_metadata()
    
    def _load_document_metadata(self):
        """Load document metadata from ChromaDB on startup."""
        try:
            # Get all documents from ChromaDB
            results = self.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            # Group by document_id and create DocumentInfo objects
            doc_info_map = {}
            for metadata in metadatas:
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in doc_info_map:
                    # Check if we have minimum required fields
                    if not metadata.get("filename") or not metadata.get("content_type"):
                        print(f"⚠️ Skipping invalid metadata for {doc_id}: missing required fields")
                        continue
                    
                    try:
                        doc_info = DocumentInfo(
                            id=doc_id,
                            filename=metadata.get("filename"),
                            content_type=metadata.get("content_type"),
                            document_type=DocumentType(metadata.get("document_type", "txt")),
                            upload_date=datetime.fromisoformat(metadata.get("upload_date", datetime.now().isoformat())),
                            size_bytes=0,  # We don't store this in metadata
                            chunk_count=0,  # Will be calculated
                            metadata={}
                        )
                        doc_info_map[doc_id] = doc_info
                    except Exception as e:
                        print(f"⚠️ Failed to load metadata for {doc_id}: {e}")
                        continue
            
            # Count chunks per document
            for metadata in metadatas:
                doc_id = metadata.get("document_id")
                if doc_id in doc_info_map:
                    doc_info_map[doc_id].chunk_count += 1
            
            self.document_metadata = doc_info_map
            print(f"📚 Loaded {len(self.document_metadata)} documents from ChromaDB")
            
        except Exception as e:
            print(f"⚠️ Failed to load document metadata: {e}")
            self.document_metadata = {}
    
    async def add_document(self, filename: str, content: bytes, content_type: str) -> DocumentInfo:
        """Add a document to the knowledge base."""
        try:
            # Generate unique document ID
            doc_id = str(uuid.uuid4())
            
            # Determine document type
            doc_type = self._get_document_type(filename, content_type)
            
            # Save document file
            file_path = os.path.join(self.documents_dir, f"{doc_id}_{filename}")
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(content)
            
            # Extract text content
            text_content = await self._extract_text(content, doc_type)
            
            # Chunk the document
            chunks = self._chunk_text(text_content)
            
            # Generate embeddings and store in vector database
            chunk_ids = []
            embeddings = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_ids.append(chunk_id)
                
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                embeddings.append(embedding)
                
                # Create metadata
                metadata = {
                    "document_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "content_type": content_type,
                    "document_type": doc_type.value,
                    "upload_date": datetime.now().isoformat()
                }
                metadatas.append(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            # Create document info
            doc_info = DocumentInfo(
                id=doc_id,
                filename=filename,
                content_type=content_type,
                document_type=doc_type,
                upload_date=datetime.now(),
                size_bytes=len(content),
                chunk_count=len(chunks),
                metadata={"file_path": file_path, "text_length": len(text_content)}
            )
            
            # Cache metadata
            self.document_metadata[doc_id] = doc_info
            
            return doc_info
            
        except Exception as e:
            raise Exception(f"Failed to add document {filename}: {str(e)}")
    
    async def search_documents(self, query: str, max_results: int = 5) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Search for relevant document chunks."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Extract results
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            return documents, metadatas
            
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")
    
    async def get_document_by_id(self, doc_id: str) -> Optional[DocumentInfo]:
        """Get document information by ID."""
        return self.document_metadata.get(doc_id)
    
    async def list_documents(self) -> List[DocumentInfo]:
        """List all documents in the knowledge base."""
        return list(self.document_metadata.values())
    
    def _get_document_type(self, filename: str, content_type: str) -> DocumentType:
        """Determine document type from filename and content type."""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf') or content_type == 'application/pdf':
            return DocumentType.PDF
        elif filename_lower.endswith('.docx') or content_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            return DocumentType.DOCX
        elif filename_lower.endswith('.html') or filename_lower.endswith('.htm') or content_type == 'text/html':
            return DocumentType.HTML
        else:
            return DocumentType.TXT
    
    async def _extract_text(self, content: bytes, doc_type: DocumentType) -> str:
        """Extract text content from different document types."""
        try:
            if doc_type == DocumentType.PDF:
                return self._extract_pdf_text(content)
            elif doc_type == DocumentType.DOCX:
                return self._extract_docx_text(content)
            elif doc_type == DocumentType.HTML:
                return self._extract_html_text(content)
            else:
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            raise Exception(f"Text extraction failed: {str(e)}")
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF content."""
        import io
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX content."""
        import io
        doc = Document(io.BytesIO(content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def _extract_html_text(self, content: bytes) -> str:
        """Extract text from HTML content."""
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()
    
    def _chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks."""
        if chunk_size is None:
            chunk_size = Config.CHUNK_SIZE
        if overlap is None:
            overlap = Config.CHUNK_OVERLAP

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(start, end - 100)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward by chunk_size - overlap, ensuring we always advance
            start = start + chunk_size - overlap
            # Safety check to prevent infinite loops
            if start <= 0:
                start = start + chunk_size

        return chunks
    
    async def get_document_chunks(self, doc_id: str) -> List[str]:
        """Get all chunks for a specific document."""
        try:
            results = self.collection.get(
                where={"document_id": doc_id},
                include=["documents"]
            )
            return results["documents"] if results["documents"] else []
        except Exception as e:
            raise Exception(f"Failed to get document chunks: {str(e)}")
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from the knowledge base."""
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"document_id": doc_id},
                include=["metadatas"]
            )
            
            # Check if there are any chunks to delete
            if results and "ids" in results and results["ids"]:
                # Delete chunks from ChromaDB
                self.collection.delete(ids=results["ids"])
            
            # Remove from metadata cache
            if doc_id in self.document_metadata:
                doc_info = self.document_metadata[doc_id]
                # Delete file
                if "file_path" in doc_info.metadata:
                    file_path = doc_info.metadata["file_path"]
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                del self.document_metadata[doc_id]
            
            return True
            
        except Exception as e:
            raise Exception(f"Failed to delete document: {str(e)}")
