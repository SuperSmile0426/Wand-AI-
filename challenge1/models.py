from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentType(str, Enum):
    PLANNER = "planner"
    FINANCIAL_ANALYST = "financial_analyst"
    DATA_ANALYST = "data_analyst"
    CHART_GENERATOR = "chart_generator"
    AGGREGATOR = "aggregator"

class TaskRequest(BaseModel):
    user_request: str
    session_id: Optional[str] = None

class Subtask(BaseModel):
    id: str
    agent_type: AgentType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    dependencies: List[str] = []

class AgentResponse(BaseModel):
    agent_type: AgentType
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: int = 0

class TaskExecution(BaseModel):
    session_id: str
    main_request: str
    subtasks: List[Subtask]
    status: TaskStatus = TaskStatus.PENDING
    final_result: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str

class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None

class ClarificationRequest(BaseModel):
    question: str
    context: Optional[Dict[str, Any]] = None
    required: bool = True

class ConversationMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str
    session_id: str

class ProgressUpdate(BaseModel):
    session_id: str
    task_id: str
    agent_type: AgentType
    status: TaskStatus
    progress: int
    message: str
    result: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[ToolCall]] = None
    clarification_needed: Optional[ClarificationRequest] = None
