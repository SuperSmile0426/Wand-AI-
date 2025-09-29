from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import openai
import json
import asyncio
import subprocess
import requests
from datetime import datetime
from challenge1.models import AgentType, TaskStatus, AgentResponse, ToolCall, ClarificationRequest
from challenge1.config import OPENAI_API_KEY

class BaseAgent(ABC):
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.context = {}
        self.tool_calls_history = []
        self.max_retries = 3
        self.retry_delay = 1.0
        self.mock_mode = True  # Enable mock mode by default to avoid API quota issues
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @abstractmethod
    def process_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process the task and return results"""
        pass
    
    async def execute(self, task_id: str, task_description: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute the task and return a structured response"""
        try:
            self.context = context or {}
            result = await self.process_task(task_description, context)
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                progress=100
            )
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                progress=0
            )
    
    def ask_clarification(self, question: str) -> str:
        """Ask for clarification when task is ambiguous"""
        return f"Clarification needed: {question}"
    
    def update_progress(self, progress: int, message: str) -> None:
        """Update progress for the task"""
        # This would be used with WebSocket to send real-time updates
        pass
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Return list of available tools for this agent"""
        return [
            {
                "name": "python_executor",
                "description": "Execute Python code and return results",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                        "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30}
                    },
                    "required": ["code"]
                }
            },
            {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "description": "Maximum number of results", "default": 5}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "data_analysis",
                "description": "Perform statistical analysis on data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "array", "description": "Data to analyze"},
                        "analysis_type": {"type": "string", "description": "Type of analysis to perform"}
                    },
                    "required": ["data", "analysis_type"]
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> ToolCall:
        """Execute a tool and return the result"""
        tool_call = ToolCall(
            tool_name=tool_name,
            parameters=parameters,
            result=None,
            error=None
        )
        
        try:
            if tool_name == "python_executor":
                result = await self._execute_python_code(
                    parameters.get("code", ""),
                    parameters.get("timeout", 30)
                )
                tool_call.result = result
            elif tool_name == "web_search":
                result = await self._web_search(
                    parameters.get("query", ""),
                    parameters.get("max_results", 5)
                )
                tool_call.result = result
            elif tool_name == "data_analysis":
                result = await self._analyze_data(
                    parameters.get("data", []),
                    parameters.get("analysis_type", "basic")
                )
                tool_call.result = result
            else:
                tool_call.error = f"Unknown tool: {tool_name}"
        except Exception as e:
            tool_call.error = str(e)
        
        self.tool_calls_history.append(tool_call)
        return tool_call
    
    async def _execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            # Create a safe execution environment
            safe_globals = {
                "__builtins__": {
                    "print": print,
                    "len": len,
                    "sum": sum,
                    "max": max,
                    "min": min,
                    "abs": abs,
                    "round": round,
                    "sorted": sorted,
                    "list": list,
                    "dict": dict,
                    "tuple": tuple,
                    "set": set,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "type": type,
                    "isinstance": isinstance,
                    "enumerate": enumerate,
                    "zip": zip,
                    "range": range,
                    "map": map,
                    "filter": filter,
                    "any": any,
                    "all": all,
                }
            }
            safe_locals = {}
            
            # Execute the code
            exec(code, safe_globals, safe_locals)
            
            return {
                "success": True,
                "result": safe_locals,
                "output": "Code executed successfully"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": f"Execution failed: {str(e)}"
            }
    
    async def _web_search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Perform web search (mock implementation)"""
        # In a real implementation, this would use a search API like Google Custom Search
        # For now, return mock results
        return {
            "query": query,
            "results": [
                {
                    "title": f"Search result 1 for: {query}",
                    "snippet": f"This is a mock search result for the query: {query}",
                    "url": "https://example.com/result1"
                },
                {
                    "title": f"Search result 2 for: {query}",
                    "snippet": f"Another mock search result for: {query}",
                    "url": "https://example.com/result2"
                }
            ],
            "total_results": 2
        }
    
    async def _analyze_data(self, data: List[Any], analysis_type: str) -> Dict[str, Any]:
        """Perform data analysis"""
        try:
            if not data:
                return {"error": "No data provided for analysis"}
            
            if analysis_type == "basic":
                return {
                    "count": len(data),
                    "mean": sum(data) / len(data) if all(isinstance(x, (int, float)) for x in data) else "N/A",
                    "min": min(data) if all(isinstance(x, (int, float)) for x in data) else "N/A",
                    "max": max(data) if all(isinstance(x, (int, float)) for x in data) else "N/A"
                }
            elif analysis_type == "statistical":
                # More advanced statistical analysis
                numeric_data = [x for x in data if isinstance(x, (int, float))]
                if not numeric_data:
                    return {"error": "No numeric data found for statistical analysis"}
                
                mean = sum(numeric_data) / len(numeric_data)
                variance = sum((x - mean) ** 2 for x in numeric_data) / len(numeric_data)
                std_dev = variance ** 0.5
                
                return {
                    "count": len(numeric_data),
                    "mean": mean,
                    "median": sorted(numeric_data)[len(numeric_data) // 2],
                    "std_deviation": std_dev,
                    "variance": variance,
                    "min": min(numeric_data),
                    "max": max(numeric_data)
                }
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}
    
    def needs_clarification(self, task_description: str, context: Dict[str, Any] = None) -> Optional[ClarificationRequest]:
        """Check if the task needs clarification"""
        # Simple heuristic - look for ambiguous terms
        ambiguous_terms = ["analyze", "review", "evaluate", "assess", "improve", "optimize", "create"]
        vague_terms = ["some", "several", "many", "few", "recent", "latest"]
        
        task_lower = task_description.lower()
        
        # Check for ambiguous terms without specific context
        for term in ambiguous_terms:
            if term in task_lower and not any(specific in task_lower for specific in ["data", "revenue", "profit", "sales", "customers", "financial", "quarterly", "performance", "trends", "metrics", "comprehensive", "detailed", "analysis", "report", "chart", "visualization"]):
                return ClarificationRequest(
                    question=f"Could you be more specific about what you'd like me to {term}? Please provide more details about the data, timeframe, or specific metrics you're interested in.",
                    context=context,
                    required=True
                )
        
        # Check for vague terms
        for term in vague_terms:
            if term in task_lower:
                return ClarificationRequest(
                    question=f"You mentioned '{term}' - could you provide more specific details? For example, what specific data, timeframe, or metrics are you referring to?",
                    context=context,
                    required=True
                )
        
        return None
    
    def enable_mock_mode(self):
        """Enable mock mode to avoid OpenAI API calls"""
        self.mock_mode = True
    
    def disable_mock_mode(self):
        """Disable mock mode"""
        self.mock_mode = False
    
    async def execute_with_retry(self, task_id: str, task_description: str, context: Dict[str, Any] = None) -> AgentResponse:
        """Execute task with retry logic to handle LLM failures"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await self.execute(task_id, task_description, context)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's an OpenAI quota error
                if "quota" in error_str or "insufficient_quota" in error_str:
                    print(f"OpenAI quota exceeded, enabling mock mode for {self.agent_type}")
                    self.enable_mock_mode()
                    return await self.execute(task_id, task_description, context)
                
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    continue
                else:
                    break
        
        # All retries failed
        return AgentResponse(
            agent_type=self.agent_type,
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=f"Task failed after {self.max_retries} attempts. Last error: {str(last_error)}",
            progress=0
        )
