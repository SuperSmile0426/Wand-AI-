"""
Tests for Challenge 1 models.
"""

import pytest
from datetime import datetime
from challenge1.models import (
    TaskStatus, AgentType, TaskRequest, Subtask, AgentResponse, 
    TaskExecution, ToolCall, ClarificationRequest, ConversationMessage, 
    ProgressUpdate
)


class TestTaskStatus:
    """Test TaskStatus enum."""
    
    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING == "pending"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.COMPLETED == "completed"
        assert TaskStatus.FAILED == "failed"
    
    def test_task_status_membership(self):
        """Test TaskStatus membership."""
        assert "pending" in TaskStatus.__members__.values()
        assert "in_progress" in TaskStatus.__members__.values()
        assert "completed" in TaskStatus.__members__.values()
        assert "failed" in TaskStatus.__members__.values()


class TestAgentType:
    """Test AgentType enum."""
    
    def test_agent_type_values(self):
        """Test AgentType enum values."""
        assert AgentType.PLANNER == "planner"
        assert AgentType.FINANCIAL_ANALYST == "financial_analyst"
        assert AgentType.DATA_ANALYST == "data_analyst"
        assert AgentType.CHART_GENERATOR == "chart_generator"
        assert AgentType.AGGREGATOR == "aggregator"
    
    def test_agent_type_membership(self):
        """Test AgentType membership."""
        assert "planner" in AgentType.__members__.values()
        assert "financial_analyst" in AgentType.__members__.values()
        assert "data_analyst" in AgentType.__members__.values()
        assert "chart_generator" in AgentType.__members__.values()
        assert "aggregator" in AgentType.__members__.values()


class TestTaskRequest:
    """Test TaskRequest model."""
    
    def test_task_request_creation(self):
        """Test TaskRequest creation."""
        request = TaskRequest(user_request="Test request")
        assert request.user_request == "Test request"
        assert request.session_id is None
    
    def test_task_request_with_session_id(self):
        """Test TaskRequest with session ID."""
        request = TaskRequest(
            user_request="Test request",
            session_id="test_session_123"
        )
        assert request.user_request == "Test request"
        assert request.session_id == "test_session_123"
    
    def test_task_request_validation(self):
        """Test TaskRequest validation."""
        # Valid request
        request = TaskRequest(user_request="Valid request")
        assert request.user_request == "Valid request"
        
        # Empty request should still be valid (validation happens at business logic level)
        request = TaskRequest(user_request="")
        assert request.user_request == ""


class TestSubtask:
    """Test Subtask model."""
    
    def test_subtask_creation(self):
        """Test Subtask creation."""
        subtask = Subtask(
            id="test_task_1",
            agent_type=AgentType.FINANCIAL_ANALYST,
            description="Test task description"
        )
        assert subtask.id == "test_task_1"
        assert subtask.agent_type == AgentType.FINANCIAL_ANALYST
        assert subtask.description == "Test task description"
        assert subtask.status == TaskStatus.PENDING
        assert subtask.result is None
        assert subtask.error is None
        assert subtask.dependencies == []
    
    def test_subtask_with_dependencies(self):
        """Test Subtask with dependencies."""
        subtask = Subtask(
            id="test_task_2",
            agent_type=AgentType.DATA_ANALYST,
            description="Test task with dependencies",
            dependencies=["task_1", "task_2"]
        )
        assert subtask.dependencies == ["task_1", "task_2"]
    
    def test_subtask_with_result(self):
        """Test Subtask with result."""
        result = {"analysis": "test result"}
        subtask = Subtask(
            id="test_task_3",
            agent_type=AgentType.CHART_GENERATOR,
            description="Test task with result",
            result=result
        )
        assert subtask.result == result


class TestAgentResponse:
    """Test AgentResponse model."""
    
    def test_agent_response_creation(self):
        """Test AgentResponse creation."""
        response = AgentResponse(
            agent_type=AgentType.PLANNER,
            task_id="test_task",
            status=TaskStatus.COMPLETED,
            result={"test": "data"}
        )
        assert response.agent_type == AgentType.PLANNER
        assert response.task_id == "test_task"
        assert response.status == TaskStatus.COMPLETED
        assert response.result == {"test": "data"}
        assert response.error is None
        assert response.progress == 0
    
    def test_agent_response_with_error(self):
        """Test AgentResponse with error."""
        response = AgentResponse(
            agent_type=AgentType.FINANCIAL_ANALYST,
            task_id="test_task",
            status=TaskStatus.FAILED,
            error="Test error message"
        )
        assert response.status == TaskStatus.FAILED
        assert response.error == "Test error message"
        assert response.result is None
    
    def test_agent_response_with_progress(self):
        """Test AgentResponse with progress."""
        response = AgentResponse(
            agent_type=AgentType.DATA_ANALYST,
            task_id="test_task",
            status=TaskStatus.IN_PROGRESS,
            progress=50
        )
        assert response.progress == 50


class TestTaskExecution:
    """Test TaskExecution model."""
    
    def test_task_execution_creation(self):
        """Test TaskExecution creation."""
        now = datetime.now().isoformat()
        execution = TaskExecution(
            session_id="test_session",
            main_request="Test request",
            subtasks=[],
            created_at=now,
            updated_at=now
        )
        assert execution.session_id == "test_session"
        assert execution.main_request == "Test request"
        assert execution.subtasks == []
        assert execution.status == TaskStatus.PENDING
        assert execution.final_result is None
        assert execution.created_at == now
        assert execution.updated_at == now
    
    def test_task_execution_with_subtasks(self):
        """Test TaskExecution with subtasks."""
        subtasks = [
            Subtask(
                id="task_1",
                agent_type=AgentType.PLANNER,
                description="Plan task"
            )
        ]
        execution = TaskExecution(
            session_id="test_session",
            main_request="Test request",
            subtasks=subtasks,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        assert len(execution.subtasks) == 1
        assert execution.subtasks[0].id == "task_1"


class TestToolCall:
    """Test ToolCall model."""
    
    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        tool_call = ToolCall(
            tool_name="python_executor",
            parameters={"code": "print('hello')"}
        )
        assert tool_call.tool_name == "python_executor"
        assert tool_call.parameters == {"code": "print('hello')"}
        assert tool_call.result is None
        assert tool_call.error is None
    
    def test_tool_call_with_result(self):
        """Test ToolCall with result."""
        tool_call = ToolCall(
            tool_name="web_search",
            parameters={"query": "test"},
            result={"results": ["test1", "test2"]}
        )
        assert tool_call.result == {"results": ["test1", "test2"]}
    
    def test_tool_call_with_error(self):
        """Test ToolCall with error."""
        tool_call = ToolCall(
            tool_name="data_analysis",
            parameters={"data": []},
            error="No data provided"
        )
        assert tool_call.error == "No data provided"


class TestClarificationRequest:
    """Test ClarificationRequest model."""
    
    def test_clarification_request_creation(self):
        """Test ClarificationRequest creation."""
        request = ClarificationRequest(
            question="What specific data do you need?",
            context={"current_task": "financial_analysis"}
        )
        assert request.question == "What specific data do you need?"
        assert request.context == {"current_task": "financial_analysis"}
        assert request.required is True
    
    def test_clarification_request_optional(self):
        """Test ClarificationRequest with optional fields."""
        request = ClarificationRequest(
            question="Optional clarification?",
            required=False
        )
        assert request.question == "Optional clarification?"
        assert request.context is None
        assert request.required is False


class TestConversationMessage:
    """Test ConversationMessage model."""
    
    def test_conversation_message_creation(self):
        """Test ConversationMessage creation."""
        now = datetime.now().isoformat()
        message = ConversationMessage(
            role="user",
            content="Hello, how can you help?",
            timestamp=now,
            session_id="test_session"
        )
        assert message.role == "user"
        assert message.content == "Hello, how can you help?"
        assert message.timestamp == now
        assert message.session_id == "test_session"
    
    def test_conversation_message_assistant(self):
        """Test ConversationMessage for assistant."""
        message = ConversationMessage(
            role="assistant",
            content="I can help with financial analysis",
            timestamp=datetime.now().isoformat(),
            session_id="test_session"
        )
        assert message.role == "assistant"
        assert message.content == "I can help with financial analysis"


class TestProgressUpdate:
    """Test ProgressUpdate model."""
    
    def test_progress_update_creation(self):
        """Test ProgressUpdate creation."""
        update = ProgressUpdate(
            session_id="test_session",
            task_id="test_task",
            agent_type=AgentType.PLANNER,
            status=TaskStatus.IN_PROGRESS,
            progress=50,
            message="Processing task..."
        )
        assert update.session_id == "test_session"
        assert update.task_id == "test_task"
        assert update.agent_type == AgentType.PLANNER
        assert update.status == TaskStatus.IN_PROGRESS
        assert update.progress == 50
        assert update.message == "Processing task..."
        assert update.result is None
        assert update.tool_calls is None
        assert update.clarification_needed is None
    
    def test_progress_update_with_result(self):
        """Test ProgressUpdate with result."""
        result = {"analysis": "completed"}
        update = ProgressUpdate(
            session_id="test_session",
            task_id="test_task",
            agent_type=AgentType.FINANCIAL_ANALYST,
            status=TaskStatus.COMPLETED,
            progress=100,
            message="Task completed",
            result=result
        )
        assert update.result == result
    
    def test_progress_update_with_tool_calls(self):
        """Test ProgressUpdate with tool calls."""
        tool_calls = [
            ToolCall(tool_name="python_executor", parameters={"code": "print('test')"})
        ]
        update = ProgressUpdate(
            session_id="test_session",
            task_id="test_task",
            agent_type=AgentType.DATA_ANALYST,
            status=TaskStatus.IN_PROGRESS,
            progress=25,
            message="Using tools...",
            tool_calls=tool_calls
        )
        assert update.tool_calls == tool_calls
    
    def test_progress_update_with_clarification(self):
        """Test ProgressUpdate with clarification needed."""
        clarification = ClarificationRequest(
            question="What specific metrics do you need?",
            required=True
        )
        update = ProgressUpdate(
            session_id="test_session",
            task_id="test_task",
            agent_type=AgentType.PLANNER,
            status=TaskStatus.PENDING,
            progress=0,
            message="Clarification needed",
            clarification_needed=clarification
        )
        assert update.clarification_needed == clarification


class TestModelSerialization:
    """Test model serialization and deserialization."""
    
    def test_task_request_serialization(self):
        """Test TaskRequest serialization."""
        request = TaskRequest(user_request="Test request")
        data = request.model_dump()
        assert data["user_request"] == "Test request"
        assert data["session_id"] is None
    
    def test_subtask_serialization(self):
        """Test Subtask serialization."""
        subtask = Subtask(
            id="test_task",
            agent_type=AgentType.PLANNER,
            description="Test description"
        )
        data = subtask.model_dump()
        assert data["id"] == "test_task"
        assert data["agent_type"] == "planner"
        assert data["description"] == "Test description"
        assert data["status"] == "pending"
    
    def test_agent_response_serialization(self):
        """Test AgentResponse serialization."""
        response = AgentResponse(
            agent_type=AgentType.FINANCIAL_ANALYST,
            task_id="test_task",
            status=TaskStatus.COMPLETED,
            result={"test": "data"},
            progress=100
        )
        data = response.model_dump()
        assert data["agent_type"] == "financial_analyst"
        assert data["task_id"] == "test_task"
        assert data["status"] == "completed"
        assert data["result"] == {"test": "data"}
        assert data["progress"] == 100
    
    def test_model_deserialization(self):
        """Test model deserialization from dict."""
        data = {
            "user_request": "Test request",
            "session_id": "test_session"
        }
        request = TaskRequest(**data)
        assert request.user_request == "Test request"
        assert request.session_id == "test_session"
    
    def test_model_json_serialization(self):
        """Test model JSON serialization."""
        request = TaskRequest(user_request="Test request")
        json_str = request.model_dump_json()
        assert "Test request" in json_str
        assert "user_request" in json_str
