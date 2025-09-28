"""
RAG (Retrieval-Augmented Generation) pipeline for AI-powered document search and answer generation.
Includes completeness checking and enrichment suggestions.
"""

import os
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

# LLM and AI
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler

from models import (
    SearchResponse, SearchRequest, DocumentInfo, MissingInfo, 
    EnrichmentSuggestion, EnrichmentType, ConfidenceLevel
)
from knowledge_base import KnowledgeBase
from config import Config

class RAGPipeline:
    """RAG pipeline for document search and AI answer generation."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the RAG pipeline."""
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            # Use a mock LLM for demonstration if no API key
            print("⚠️ No OpenAI API key found. Using MockLLM for demonstration.")
            self.llm = MockLLM()
        else:
            llm_config = Config.get_llm_config()
            print(f"✅ Using OpenAI API with model: gpt-3.5-turbo")
            self.llm = ChatOpenAI(
                openai_api_key=self.openai_api_key,
                model_name="gpt-3.5-turbo",
                temperature=llm_config["temperature"],
                max_tokens=llm_config["max_tokens"]
            )
        
        # Initialize prompt templates
        self._setup_prompts()
        
        # Rating storage (in production, use a proper database)
        self.ratings = []
    
    def _setup_prompts(self):
        """Set up prompt templates for different tasks."""
        
        # Main answer generation prompt
        self.answer_prompt = PromptTemplate(
            input_variables=["query", "context", "documents"],
            template="""
You are an AI assistant that answers questions based on provided documents. Use the following context to answer the user's question.

Context from documents:
{context}

Available documents:
{documents}

Question: {query}

Instructions:
1. Answer the question based on the provided context
2. If the context doesn't contain enough information, clearly state what's missing
3. Cite specific documents when possible
4. Be concise but comprehensive
5. If you're uncertain about something, express that uncertainty

Answer:
"""
        )
        
        # Completeness check prompt
        self.completeness_prompt = PromptTemplate(
            input_variables=["query", "answer", "context"],
            template="""
Analyze the completeness of this answer for the given question.

Question: {query}
Answer: {answer}
Context used: {context}

Rate the completeness and identify missing information. Respond in JSON format:
{{
    "confidence_score": 0.0-1.0,
    "confidence_level": "high/medium/low",
    "missing_info": [
        {{
            "topic": "specific topic that's missing",
            "description": "what information is needed",
            "importance": "high/medium/low"
        }}
    ],
    "completeness_reasoning": "explanation of why the answer is complete/incomplete"
}}

JSON Response:
"""
        )
        
        # Enrichment suggestion prompt
        self.enrichment_prompt = PromptTemplate(
            input_variables=["query", "missing_info", "available_documents"],
            template="""
Based on the missing information identified, suggest how to enrich the knowledge base.

Question: {query}
Missing information: {missing_info}
Available documents: {available_documents}

Suggest enrichment strategies. Respond in JSON format:
{{
    "suggestions": [
        {{
            "type": "document/web_search/api_fetch/user_input",
            "description": "what to do to get the missing information",
            "missing_info": ["list of missing topics this addresses"],
            "suggested_actions": ["specific actions to take"],
            "confidence": 0.0-1.0,
            "auto_enrichable": true/false
        }}
    ]
}}

JSON Response:
"""
        )
    
    async def search(self, query: str, knowledge_base: KnowledgeBase, include_enrichment: bool = True) -> SearchResponse:
        """Perform RAG search and generate answer with enrichment suggestions."""
        start_time = datetime.now()
        
        try:
            print(f"🔍 Starting search for query: '{query}'")
            
            # 1. Retrieve relevant documents
            documents, metadatas = await knowledge_base.search_documents(query, max_results=5)
            print(f"📚 Retrieved {len(documents)} document chunks")
            
            # 2. Prepare context
            context = self._prepare_context(documents, metadatas)
            document_infos = await self._get_document_infos(metadatas, knowledge_base)
            print(f"📄 Context length: {len(context)} characters")
            print(f"📋 Document infos: {len(document_infos)} documents")
            
            # 3. Generate answer
            answer = await self._generate_answer(query, context, document_infos)
            print(f"🤖 Generated answer: {len(answer)} characters")
            
            # 4. Check completeness
            completeness_result = await self._check_completeness(query, answer, context)
            print(f"📊 Confidence: {completeness_result['confidence_level']} ({completeness_result['confidence_score']:.2f})")
            print(f"⚠️ Missing info: {len(completeness_result['missing_info'])} items")
            
            # 5. Generate enrichment suggestions if requested
            enrichment_suggestions = []
            if include_enrichment and completeness_result["missing_info"]:
                enrichment_suggestions = await self._generate_enrichment_suggestions(
                    query, completeness_result["missing_info"], document_infos
                )
                print(f"💡 Enrichment suggestions: {len(enrichment_suggestions)} strategies")
            
            # 6. Calculate processing time
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            print(f"⏱️ Processing time: {processing_time}ms")
            
            return SearchResponse(
                query=query,
                answer=answer,
                confidence=ConfidenceLevel(completeness_result["confidence_level"]),
                confidence_score=completeness_result["confidence_score"],
                sources=document_infos,
                missing_info=[MissingInfo(**info) for info in completeness_result["missing_info"]],
                enrichment_suggestions=enrichment_suggestions,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            print(f"❌ RAG search failed: {str(e)}")
            raise Exception(f"RAG search failed: {str(e)}")
    
    def _prepare_context(self, documents: List[str], metadatas: List[Dict[str, Any]]) -> str:
        """Prepare context from retrieved documents."""
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            context_parts.append(f"Document {i+1} ({metadata.get('filename', 'Unknown')}):\n{doc}\n")
        return "\n".join(context_parts)
    
    async def _get_document_infos(self, metadatas: List[Dict[str, Any]], knowledge_base: KnowledgeBase) -> List[DocumentInfo]:
        """Get DocumentInfo objects for the retrieved documents."""
        document_infos = []
        seen_docs = set()
        
        print(f"🔍 Processing {len(metadatas)} metadata entries")
        
        for i, metadata in enumerate(metadatas):
            print(f"📋 Metadata {i}: {metadata}")
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in seen_docs:
                doc_info = await knowledge_base.get_document_by_id(doc_id)
                if doc_info:
                    document_infos.append(doc_info)
                    seen_docs.add(doc_id)
                    print(f"✅ Found document info for {doc_id}: {doc_info.filename}")
                else:
                    print(f"❌ No document info found for {doc_id}")
            else:
                print(f"⚠️ Skipping metadata {i}: doc_id={doc_id}, seen={doc_id in seen_docs if doc_id else 'N/A'}")
        
        print(f"📚 Final document_infos count: {len(document_infos)}")
        return document_infos
    
    async def _generate_answer(self, query: str, context: str, documents: List[DocumentInfo]) -> str:
        """Generate answer using the LLM."""
        try:
            # Prepare document list for context
            doc_list = [f"- {doc.filename} ({doc.document_type.value})" for doc in documents]
            
            # Format the prompt
            formatted_prompt = self.answer_prompt.format(
                query=query,
                context=context,
                documents="\n".join(doc_list)
            )
            
            # Generate answer using ChatOpenAI
            if hasattr(self.llm, 'ainvoke'):
                # Modern ChatOpenAI
                result = await self.llm.ainvoke(formatted_prompt)
                if hasattr(result, 'content'):
                    return result.content.strip()
                else:
                    return str(result).strip()
            else:
                # Fallback for MockLLM
                result = self.llm.arun(query=query, context=context, documents="\n".join(doc_list))
                return result.strip()
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            # Fallback to simple context-based answer
            return f"Based on the available documents, here's what I found:\n\n{context[:500]}..."
    
    async def _check_completeness(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Check the completeness of the answer and identify missing information."""
        try:
            # Format the prompt
            formatted_prompt = self.completeness_prompt.format(
                query=query,
                answer=answer,
                context=context
            )
            
            # Generate completeness analysis using ChatOpenAI
            if hasattr(self.llm, 'ainvoke'):
                # Modern ChatOpenAI
                result = await self.llm.ainvoke(formatted_prompt)
                if hasattr(result, 'content'):
                    response_text = result.content.strip()
                else:
                    response_text = str(result).strip()
            else:
                # Fallback for MockLLM
                result = self.llm.arun(query=query, answer=answer, context=context)
                response_text = result.strip()
            
            # Parse JSON response
            try:
                analysis = json.loads(response_text)
                return analysis
            except json.JSONDecodeError:
                # Fallback analysis
                return self._fallback_completeness_analysis(query, answer, context)
                
        except Exception as e:
            return self._fallback_completeness_analysis(query, answer, context)
    
    def _fallback_completeness_analysis(self, query: str, answer: str, context: str) -> Dict[str, Any]:
        """Fallback completeness analysis using simple heuristics."""
        # Simple heuristics for completeness
        uncertainty_indicators = [
            "i don't know", "i'm not sure", "unclear", "uncertain", 
            "might be", "could be", "possibly", "perhaps", "not sure"
        ]
        
        missing_indicators = [
            "not mentioned", "not found", "not available", "missing",
            "no information", "not specified", "unclear", "don't have enough"
        ]
        
        answer_lower = answer.lower()
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in answer_lower)
        missing_count = sum(1 for indicator in missing_indicators if indicator in answer_lower)
        
        # Check answer length and content quality
        answer_length = len(answer.strip())
        has_context = len(context.strip()) > 50
        has_specific_info = any(word in answer_lower for word in query.lower().split() if len(word) > 3)
        
        # Calculate confidence based on multiple factors
        confidence_factors = []
        
        # Length factor
        if answer_length > 200:
            confidence_factors.append(0.2)
        elif answer_length > 100:
            confidence_factors.append(0.1)
        else:
            confidence_factors.append(-0.1)
        
        # Context factor
        if has_context:
            confidence_factors.append(0.3)
        else:
            confidence_factors.append(-0.2)
        
        # Specificity factor
        if has_specific_info:
            confidence_factors.append(0.2)
        else:
            confidence_factors.append(-0.1)
        
        # Uncertainty penalty
        confidence_factors.append(-uncertainty_count * 0.1)
        confidence_factors.append(-missing_count * 0.15)
        
        # Calculate final confidence
        base_confidence = 0.5
        confidence_score = max(0.1, min(1.0, base_confidence + sum(confidence_factors)))
        
        # Additional check for query relevance
        query_words = query.lower().split()
        answer_words = answer_lower.split()
        relevance_score = len(set(query_words) & set(answer_words)) / max(len(query_words), 1)
        confidence_score = (confidence_score + relevance_score * 0.3) / 1.3
        
        if confidence_score >= 0.7:
            confidence_level = "high"
        elif confidence_score >= 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Identify missing topics based on query analysis
        missing_topics = []
        query_lower = query.lower()
        
        # Enhanced missing information detection based on query patterns
        if "name" in query_lower and "name" not in answer_lower and "called" not in answer_lower:
            missing_topics.append({
                "topic": "Personal Name",
                "description": "The person's full name or identity information",
                "importance": "high"
            })
        
        if "skill" in query_lower and ("skill" not in answer_lower or "expertise" not in answer_lower):
            missing_topics.append({
                "topic": "Skills and Expertise",
                "description": "Detailed skills, technical abilities, or professional competencies",
                "importance": "high"
            })
        
        if "education" in query_lower or "university" in query_lower or "degree" in query_lower:
            if not any(word in answer_lower for word in ["university", "education", "degree", "studied", "graduated"]):
                missing_topics.append({
                    "topic": "Educational Background",
                    "description": "University, degree, educational qualifications, or academic achievements",
                    "importance": "high"
                })
        
        if "experience" in query_lower or "work" in query_lower or "job" in query_lower:
            if not any(word in answer_lower for word in ["experience", "worked", "job", "position", "career"]):
                missing_topics.append({
                    "topic": "Work Experience",
                    "description": "Professional experience, job history, career background, or employment details",
                    "importance": "high"
                })
        
        if "contact" in query_lower or "email" in query_lower or "phone" in query_lower:
            if not any(word in answer_lower for word in ["email", "phone", "contact", "address"]):
                missing_topics.append({
                    "topic": "Contact Information",
                    "description": "Email address, phone number, or other contact details",
                    "importance": "medium"
                })
        
        if "how" in query_lower and "steps" not in answer_lower and "process" not in answer_lower:
            missing_topics.append({
                "topic": "Process details",
                "description": "Step-by-step instructions or process explanation",
                "importance": "high"
            })
        
        if "what" in query_lower and "definition" not in answer_lower and "is" not in answer_lower:
            missing_topics.append({
                "topic": "Definition",
                "description": "Clear definition or explanation of the concept",
                "importance": "high"
            })
        
        if "when" in query_lower and "time" not in answer_lower and "date" not in answer_lower:
            missing_topics.append({
                "topic": "Timeline information",
                "description": "When events occurred or time-related details",
                "importance": "medium"
            })
        
        if "where" in query_lower and "location" not in answer_lower and "place" not in answer_lower:
            missing_topics.append({
                "topic": "Location details",
                "description": "Where something occurred or is located",
                "importance": "medium"
            })
        
        # Check if answer is too generic
        if "based on the uploaded documents" in answer_lower and len(answer_lower) < 200:
            missing_topics.append({
                "topic": "Specific Details",
                "description": "More specific and detailed information from the documents",
                "importance": "high"
            })
        
        if not has_context:
            missing_topics.append({
                "topic": "Relevant context",
                "description": "More relevant documents or information needed",
                "importance": "high"
            })
        
        return {
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "missing_info": missing_topics,
            "completeness_reasoning": f"Analysis based on answer length ({answer_length} chars), context availability ({has_context}), specificity ({has_specific_info}), uncertainty indicators: {uncertainty_count}, missing indicators: {missing_count}"
        }
    
    async def _generate_enrichment_suggestions(self, query: str, missing_info: List[Dict[str, Any]], documents: List[DocumentInfo]) -> List[EnrichmentSuggestion]:
        """Generate enrichment suggestions based on missing information."""
        try:
            # Prepare document list
            doc_list = [f"- {doc.filename} ({doc.document_type.value})" for doc in documents]
            
            # Format the prompt
            formatted_prompt = self.enrichment_prompt.format(
                query=query,
                missing_info=json.dumps(missing_info),
                available_documents="\n".join(doc_list)
            )
            
            # Generate suggestions using ChatOpenAI
            if hasattr(self.llm, 'ainvoke'):
                # Modern ChatOpenAI
                result = await self.llm.ainvoke(formatted_prompt)
                if hasattr(result, 'content'):
                    response_text = result.content.strip()
                else:
                    response_text = str(result).strip()
            else:
                # Fallback for MockLLM
                result = self.llm.arun(query=query, missing_info=json.dumps(missing_info), available_documents="\n".join(doc_list))
                response_text = result.strip()
            
            # Parse JSON response
            try:
                suggestions_data = json.loads(response_text)
                suggestions = []
                
                for suggestion_data in suggestions_data.get("suggestions", []):
                    suggestions.append(EnrichmentSuggestion(
                        type=EnrichmentType(suggestion_data["type"]),
                        description=suggestion_data["description"],
                        missing_info=[MissingInfo(**info) for info in suggestion_data["missing_info"]],
                        suggested_actions=suggestion_data["suggested_actions"],
                        confidence=suggestion_data["confidence"],
                        auto_enrichable=suggestion_data.get("auto_enrichable", False)
                    ))
                
                return suggestions
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                return self._fallback_enrichment_suggestions(missing_info)
                
        except Exception as e:
            return self._fallback_enrichment_suggestions(missing_info)
    
    def _fallback_enrichment_suggestions(self, missing_info: List[Dict[str, Any]]) -> List[EnrichmentSuggestion]:
        """Generate fallback enrichment suggestions."""
        suggestions = []
        
        for info in missing_info:
            importance = info.get("importance", "medium")
            topic = info.get("topic", "Additional information")
            description = info.get("description", "More details needed")
            
            # Create suggestion based on importance
            if importance == "high":
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.DOCUMENT,
                    description=f"Upload additional documents about {topic}",
                    missing_info=[MissingInfo(**info)],
                    suggested_actions=[
                        f"Search for documents related to {topic}",
                        f"Upload relevant files about {topic}",
                        "Consider web search for recent information",
                        "Look for official documentation or research papers"
                    ],
                    confidence=0.8,
                    auto_enrichable=False
                ))
                
                # Add web search suggestion for high importance
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.WEB_SEARCH,
                    description=f"Search the web for recent information about {topic}",
                    missing_info=[MissingInfo(**info)],
                    suggested_actions=[
                        f"Search for '{topic}' on trusted sources",
                        "Look for recent articles or documentation",
                        "Check official websites or repositories"
                    ],
                    confidence=0.6,
                    auto_enrichable=True
                ))
                
            elif importance == "medium":
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.DOCUMENT,
                    description=f"Add more information about {topic}",
                    missing_info=[MissingInfo(**info)],
                    suggested_actions=[
                        f"Look for additional documents covering {topic}",
                        "Consider user-provided information",
                        "Search for supplementary materials"
                    ],
                    confidence=0.6,
                    auto_enrichable=False
                ))
            else:
                suggestions.append(EnrichmentSuggestion(
                    type=EnrichmentType.USER_INPUT,
                    description=f"Request user input about {topic}",
                    missing_info=[MissingInfo(**info)],
                    suggested_actions=[
                        f"Ask user for more details about {topic}",
                        "Request clarification or additional context",
                        "Gather user-specific information"
                    ],
                    confidence=0.5,
                    auto_enrichable=False
                ))
        
        return suggestions
    
    async def add_rating(self, query: str, answer: str, rating: int, feedback: Optional[str] = None):
        """Add a rating for an answer to improve the pipeline."""
        rating_data = {
            "query": query,
            "answer": answer,
            "rating": rating,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        self.ratings.append(rating_data)
        
        # In production, save to database
        # For now, just store in memory
        print(f"Rating added: {rating}/5 for query: {query[:50]}...")
    
    def get_ratings(self) -> List[Dict[str, Any]]:
        """Get all ratings for analysis."""
        return self.ratings.copy()

class MockLLM:
    """Mock LLM for demonstration when OpenAI API key is not available."""
    
    def arun(self, **kwargs) -> str:
        """Mock async run method."""
        query = kwargs.get("query", "")
        context = kwargs.get("context", "")
        
        # Simple mock response based on context
        if "completeness" in str(kwargs.get("template", "")):
            return json.dumps({
                "confidence_score": 0.7,
                "confidence_level": "medium",
                "missing_info": [
                    {
                        "topic": "Additional context",
                        "description": "More detailed information could be helpful",
                        "importance": "medium"
                    }
                ],
                "completeness_reasoning": "Answer is reasonably complete but could benefit from more context"
            })
        elif "enrichment" in str(kwargs.get("template", "")):
            return json.dumps({
                "suggestions": [
                    {
                        "type": "document",
                        "description": "Upload additional documents for better coverage",
                        "missing_info": [{"topic": "Additional context", "description": "More detailed information", "importance": "medium"}],
                        "suggested_actions": ["Search for related documents", "Upload relevant files"],
                        "confidence": 0.6,
                        "auto_enrichable": False
                    }
                ]
            })
        else:
            # Generate answer
            if context and len(context.strip()) > 50:
                # Extract key information from context
                context_lower = context.lower()
                query_lower = query.lower()
                
                # Clean and split context into sentences
                context_clean = context.replace('\n', ' ').replace('\r', ' ')
                sentences = [s.strip() for s in context_clean.split('.') if s.strip()]
                
                # Enhanced query analysis
                query_words = [word.lower() for word in query_lower.split() if len(word) > 2]
                
                # Look for specific patterns in the query
                is_name_query = any(word in query_lower for word in ['name', 'who', 'identity'])
                is_skill_query = any(word in query_lower for word in ['skill', 'ability', 'expertise', 'competence', 'capability'])
                is_education_query = any(word in query_lower for word in ['education', 'university', 'degree', 'school', 'college', 'study'])
                is_experience_query = any(word in query_lower for word in ['experience', 'work', 'job', 'career', 'background', 'history'])
                is_contact_query = any(word in query_lower for word in ['contact', 'email', 'phone', 'address', 'location'])
                
                # Find relevant sentences with enhanced matching
                relevant_sentences = []
                
                for sentence in sentences:
                    if len(sentence) > 15:  # Minimum sentence length
                        sentence_lower = sentence.lower()
                        
                        # Calculate relevance score
                        relevance_score = 0
                        
                        # Basic keyword matching
                        keyword_matches = sum(1 for word in query_words if word in sentence_lower)
                        relevance_score += keyword_matches * 2
                        
                        # Pattern-specific matching
                        if is_name_query and any(word in sentence_lower for word in ['name', 'i am', 'my name', 'called', 'known as']):
                            relevance_score += 5
                        elif is_skill_query and any(word in sentence_lower for word in ['skill', 'ability', 'expertise', 'proficient', 'experienced', 'knowledge']):
                            relevance_score += 5
                        elif is_education_query and any(word in sentence_lower for word in ['university', 'degree', 'education', 'studied', 'graduated', 'bachelor', 'master', 'phd']):
                            relevance_score += 5
                        elif is_experience_query and any(word in sentence_lower for word in ['experience', 'worked', 'job', 'position', 'role', 'career']):
                            relevance_score += 5
                        elif is_contact_query and any(word in sentence_lower for word in ['email', 'phone', 'contact', 'address', 'location']):
                            relevance_score += 5
                        
                        # Boost score for sentences with personal pronouns (likely about the person)
                        if any(word in sentence_lower for word in ['i', 'my', 'me', 'myself']):
                            relevance_score += 2
                        
                        # Boost score for sentences with specific details
                        if any(char in sentence for char in ['@', 'http', 'www', '2020', '2021', '2022', '2023', '2024', '2025']):
                            relevance_score += 1
                        
                        if relevance_score > 0:
                            relevant_sentences.append((sentence, relevance_score))
                
                # Sort by relevance score
                relevant_sentences.sort(key=lambda x: x[1], reverse=True)
                
                # Generate answer based on relevance and query type
                if relevant_sentences:
                    # Take top sentences based on relevance
                    top_sentences = [s[0] for s in relevant_sentences[:4]]
                    relevant_text = '. '.join(top_sentences) + '.'
                    
                    # Create a more natural answer based on query type
                    if is_name_query:
                        answer = f"Based on the uploaded documents, I can see that:\n\n{relevant_text}\n\nThis information is extracted from your uploaded resume/CV documents."
                    elif is_skill_query:
                        answer = f"According to the uploaded documents, the skills and expertise include:\n\n{relevant_text}\n\nThis information comes from your uploaded documents and shows your professional capabilities."
                    elif is_education_query:
                        answer = f"From the uploaded documents, the educational background shows:\n\n{relevant_text}\n\nThis educational information is extracted from your uploaded documents."
                    elif is_experience_query:
                        answer = f"Based on the uploaded documents, the work experience includes:\n\n{relevant_text}\n\nThis professional experience information comes from your uploaded resume/CV."
                    elif is_contact_query:
                        answer = f"From the uploaded documents, the contact information shows:\n\n{relevant_text}\n\nThis contact information is extracted from your uploaded documents."
                    else:
                        answer = f"Based on the uploaded documents, here's what I found regarding '{query}':\n\n{relevant_text}\n\nThis information is extracted from your uploaded documents."
                    
                    # Add additional context if available
                    if len(relevant_sentences) > 4:
                        additional_sentences = [s[0] for s in relevant_sentences[4:7]]
                        additional_text = '. '.join(additional_sentences) + '.'
                        answer += f"\n\nAdditional relevant information:\n{additional_text}"
                    
                    return answer
                else:
                    # If no specific matches, provide a more intelligent response
                    context_sentences = [s for s in sentences if len(s) > 20]
                    if context_sentences:
                        # Take first few sentences as they might contain relevant info
                        preview_sentences = context_sentences[:3]
                        preview_text = '. '.join(preview_sentences) + '.'
                        return f"Based on the uploaded documents, I found some relevant information:\n\n{preview_text}\n\nWhile this information is from your uploaded documents, it may not directly address your specific question about '{query}'. You might want to upload more specific documents or rephrase your question."
                    else:
                        return f"I found some information in the uploaded documents, but it may not directly address '{query}'. The available content appears to be limited. Please upload more relevant documents or provide additional context for a more specific answer."
            else:
                return f"I don't have enough information in the uploaded documents to fully answer '{query}'. Please upload more relevant documents or provide additional context."
