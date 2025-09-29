"""
Tests for Challenge 1 TaskOrchestrator class.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from challenge1.orchestrator import TaskOrchestrator
from challenge1.models import (
    TaskStatus, AgentType, TaskExecution, Subtask, ProgressUpdate, 
    ConversationMessage, ClarificationRequest
)


class TestTaskOrchestrator:
    """Test TaskOrchestrator class."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create a TaskOrchestrator instance for testing."""
        return TaskOrchestrator()
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test TaskOrchestrator initialization."""
        assert len(orchestrator.agents) == 5
        assert AgentType.PLANNER in orchestrator.agents
        assert AgentType.FINANCIAL_ANALYST in orchestrator.agents
        assert AgentType.DATA_ANALYST in orchestrator.agents
        assert AgentType.CHART_GENERATOR in orchestrator.agents
        assert AgentType.AGGREGATOR in orchestrator.agents
        assert orchestrator.active_executions == {}
        assert orchestrator.progress_callbacks == []
        assert orchestrator.conversations == {}
        assert orchestrator.pending_clarifications == {}
    
    def test_add_progress_callback(self, orchestrator):
        """Test add_progress_callback method."""
        callback = Mock()
        orchestrator.add_progress_callback(callback)
        assert callback in orchestrator.progress_callbacks
    
    @pytest.mark.asyncio
    async def test_notify_progress(self, orchestrator):
        """Test _notify_progress method."""
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        orchestrator.add_progress_callback(callback1)
        orchestrator.add_progress_callback(callback2)
        
        update = ProgressUpdate(
            session_id="test_session",
            task_id="test_task",
            agent_type=AgentType.PLANNER,
            status=TaskStatus.IN_PROGRESS,
            progress=50,
            message="Test progress"
        )
        
        await orchestrator._notify_progress(update)
        
        callback1.assert_called_once_with(update)
        callback2.assert_called_once_with(update)
    
    @pytest.mark.asyncio
    async def test_notify_progress_with_exception(self, orchestrator):
        """Test _notify_progress with callback exception."""
        callback = AsyncMock(side_effect=Exception("Callback error"))
        orchestrator.add_progress_callback(callback)
        
        update = ProgressUpdate(
            session_id="test_session",
            task_id="test_task",
            agent_type=AgentType.PLANNER,
            status=TaskStatus.IN_PROGRESS,
            progress=50,
            message="Test progress"
        )
        
        # Should not raise exception
        await orchestrator._notify_progress(update)
        callback.assert_called_once_with(update)
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, orchestrator):
        """Test successful task execution."""
        # Mock planner agent to return subtasks
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.COMPLETED
        mock_planner_result.result = {
            "subtasks": [
                {
                    "id": "financial_analysis",
                    "agent_type": "financial_analyst",
                    "description": "Perform financial analysis",
                    "dependencies": []
                }
            ],
            "clarification_needed": None
        }
        
        # Mock financial agent
        mock_financial_result = Mock()
        mock_financial_result.status = TaskStatus.COMPLETED
        mock_financial_result.result = {"analysis": "Financial analysis complete"}
        
        # Mock aggregator agent
        mock_aggregator_result = Mock()
        mock_aggregator_result.status = TaskStatus.COMPLETED
        mock_aggregator_result.result = {"final_report": "Comprehensive report"}
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result), \
             patch.object(orchestrator.agents[AgentType.FINANCIAL_ANALYST], 'execute_with_retry', return_value=mock_financial_result), \
             patch.object(orchestrator.agents[AgentType.AGGREGATOR], 'execute_with_retry', return_value=mock_aggregator_result):
            
            session_id = await orchestrator.execute_task("analyze quarterly performance")
        
        assert session_id is not None
        assert session_id in orchestrator.active_executions
        
        execution = orchestrator.active_executions[session_id]
        assert execution.status == TaskStatus.COMPLETED
        assert execution.final_result == {"final_report": "Comprehensive report"}
        assert len(execution.subtasks) == 1
        assert execution.subtasks[0].id == "financial_analysis"
    
    @pytest.mark.asyncio
    async def test_execute_task_with_clarification(self, orchestrator):
        """Test task execution requiring clarification."""
        # Mock planner agent to return clarification request
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.COMPLETED
        mock_planner_result.result = {
            "subtasks": [],
            "clarification_needed": "What specific data do you need?",
            "clarification_context": {"current_task": "planning"}
        }
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result):
            session_id = await orchestrator.execute_task("analyze data")
        
        assert session_id is not None
        assert session_id in orchestrator.active_executions
        
        execution = orchestrator.active_executions[session_id]
        assert execution.status == TaskStatus.PENDING
        assert session_id in orchestrator.pending_clarifications
        
        clarification = orchestrator.pending_clarifications[session_id]
        assert clarification.question == "What specific data do you need?"
    
    @pytest.mark.asyncio
    async def test_execute_task_planning_failure(self, orchestrator):
        """Test task execution with planning failure."""
        # Mock planner agent to fail
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.FAILED
        mock_planner_result.error = "Planning failed"
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result):
            with pytest.raises(Exception) as exc_info:
                await orchestrator.execute_task("analyze data")
            
            assert "Planning failed: Planning failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_task_invalid_subtasks(self, orchestrator):
        """Test task execution with invalid subtasks format."""
        # Mock planner agent to return invalid subtasks
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.COMPLETED
        mock_planner_result.result = {
            "subtasks": "invalid_format",  # Should be a list
            "clarification_needed": None
        }
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result):
            with pytest.raises(Exception) as exc_info:
                await orchestrator.execute_task("analyze data")
            
            assert "Invalid subtasks format from planner" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_task_circular_dependencies(self, orchestrator):
        """Test task execution with circular dependencies."""
        # Mock planner agent to return subtasks with circular dependencies
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.COMPLETED
        mock_planner_result.result = {
            "subtasks": [
                {
                    "id": "task1",
                    "agent_type": "financial_analyst",
                    "description": "Task 1",
                    "dependencies": ["task2"]
                },
                {
                    "id": "task2",
                    "agent_type": "data_analyst",
                    "description": "Task 2",
                    "dependencies": ["task1"]
                }
            ],
            "clarification_needed": None
        }
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result):
            with pytest.raises(Exception) as exc_info:
                await orchestrator.execute_task("analyze data")
            
            assert "Circular dependencies detected in subtasks" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_task_deadlock(self, orchestrator):
        """Test task execution with deadlock."""
        # Mock planner agent to return subtasks with unmet dependencies
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.COMPLETED
        mock_planner_result.result = {
            "subtasks": [
                {
                    "id": "task1",
                    "agent_type": "financial_analyst",
                    "description": "Task 1",
                    "dependencies": ["nonexistent_task"]
                }
            ],
            "clarification_needed": None
        }
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result):
            with pytest.raises(Exception) as exc_info:
                await orchestrator.execute_task("analyze data")
            
            assert "Task execution deadlock" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execute_task_aggregation_failure(self, orchestrator):
        """Test task execution with aggregation failure."""
        # Mock planner and financial agents to succeed
        mock_planner_result = Mock()
        mock_planner_result.status = TaskStatus.COMPLETED
        mock_planner_result.result = {
            "subtasks": [
                {
                    "id": "financial_analysis",
                    "agent_type": "financial_analyst",
                    "description": "Perform financial analysis",
                    "dependencies": []
                }
            ],
            "clarification_needed": None
        }
        
        mock_financial_result = Mock()
        mock_financial_result.status = TaskStatus.COMPLETED
        mock_financial_result.result = {"analysis": "Financial analysis complete"}
        
        # Mock aggregator agent to fail
        mock_aggregator_result = Mock()
        mock_aggregator_result.status = TaskStatus.FAILED
        mock_aggregator_result.error = "Aggregation failed"
        
        with patch.object(orchestrator.agents[AgentType.PLANNER], 'execute_with_retry', return_value=mock_planner_result), \
             patch.object(orchestrator.agents[AgentType.FINANCIAL_ANALYST], 'execute_with_retry', return_value=mock_financial_result), \
             patch.object(orchestrator.agents[AgentType.AGGREGATOR], 'execute_with_retry', return_value=mock_aggregator_result):
            
            with pytest.raises(Exception) as exc_info:
                await orchestrator.execute_task("analyze quarterly performance")
            
            assert "Aggregation failed: Aggregation failed" in str(exc_info.value)
    
    def test_get_execution(self, orchestrator):
        """Test get_execution method."""
        # Test with non-existent session
        result = orchestrator.get_execution("nonexistent_session")
        assert result is None
        
        # Test with existing session
        execution = TaskExecution(
            session_id="test_session",
            main_request="Test request",
            subtasks=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        orchestrator.active_executions["test_session"] = execution
        
        result = orchestrator.get_execution("test_session")
        assert result == execution
    
    def test_get_all_executions(self, orchestrator):
        """Test get_all_executions method."""
        # Test with no executions
        result = orchestrator.get_all_executions()
        assert result == []
        
        # Test with executions
        execution1 = TaskExecution(
            session_id="session1",
            main_request="Request 1",
            subtasks=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        execution2 = TaskExecution(
            session_id="session2",
            main_request="Request 2",
            subtasks=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        orchestrator.active_executions["session1"] = execution1
        orchestrator.active_executions["session2"] = execution2
        
        result = orchestrator.get_all_executions()
        assert len(result) == 2
        assert execution1 in result
        assert execution2 in result
    
    @pytest.mark.asyncio
    async def test_start_conversation(self, orchestrator):
        """Test start_conversation method."""
        session_id = "test_session"
        message = "Hello, can you help me?"
        
        with patch.object(orchestrator, '_process_conversation_message', return_value="Hello! How can I help you?"):
            response = await orchestrator.start_conversation(session_id, message)
        
        assert response == "Hello! How can I help you?"
        assert session_id in orchestrator.conversations
        assert len(orchestrator.conversations[session_id]) == 2  # User message + assistant response
        
        # Check user message
        user_message = orchestrator.conversations[session_id][0]
        assert user_message.role == "user"
        assert user_message.content == message
        assert user_message.session_id == session_id
        
        # Check assistant response
        assistant_message = orchestrator.conversations[session_id][1]
        assert assistant_message.role == "assistant"
        assert assistant_message.content == "Hello! How can I help you?"
        assert assistant_message.session_id == session_id
    
    @pytest.mark.asyncio
    async def test_continue_conversation(self, orchestrator):
        """Test continue_conversation method."""
        session_id = "test_session"
        message = "Can you analyze some data?"
        
        # Test with non-existent session
        response = await orchestrator.continue_conversation("nonexistent_session", message)
        assert response == "No active conversation found. Please start a new conversation."
        
        # Test with existing session - create the session first
        orchestrator.conversations[session_id] = []
        with patch.object(orchestrator, '_process_conversation_message', return_value="I can help you analyze data!"):
            response = await orchestrator.continue_conversation(session_id, message)
        
        assert response == "I can help you analyze data!"
        assert session_id in orchestrator.conversations
        assert len(orchestrator.conversations[session_id]) == 2  # User message + assistant response
    
    @pytest.mark.asyncio
    async def test_process_conversation_message_task_modification(self, orchestrator):
        """Test _process_conversation_message with task modification."""
        session_id = "test_session"
        message = "modify the previous analysis to include more data"
        
        # Mock execution
        execution = TaskExecution(
            session_id=session_id,
            main_request="Original request",
            subtasks=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        orchestrator.active_executions[session_id] = execution
        
        with patch.object(orchestrator, 'execute_task', return_value="new_session_id"):
            response = await orchestrator._process_conversation_message(session_id, message)
        
        assert "new task with your modifications" in response
        assert "new_session_id" in response
    
    @pytest.mark.asyncio
    async def test_process_conversation_message_clarification_response(self, orchestrator):
        """Test _process_conversation_message with clarification response."""
        session_id = "test_session"
        message = "I need revenue data for Q1-Q4 2023"
        
        # Mock pending clarification
        clarification = ClarificationRequest(
            question="What specific data do you need?",
            required=True
        )
        orchestrator.pending_clarifications[session_id] = clarification
        
        # Mock execution
        execution = TaskExecution(
            session_id=session_id,
            main_request="Original request",
            subtasks=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        orchestrator.active_executions[session_id] = execution
        
        with patch.object(orchestrator, 'execute_task', return_value=session_id):
            response = await orchestrator._process_conversation_message(session_id, message)
        
        assert "restarted the task with your additional information" in response
        assert session_id not in orchestrator.pending_clarifications
    
    @pytest.mark.asyncio
    async def test_process_conversation_message_new_task_request(self, orchestrator):
        """Test _process_conversation_message with new task request."""
        session_id = "test_session"
        message = "analyze the financial performance"
        
        with patch.object(orchestrator, 'execute_task', return_value=session_id):
            response = await orchestrator._process_conversation_message(session_id, message)
        
        assert "started processing your request" in response
    
    @pytest.mark.asyncio
    async def test_process_conversation_message_general_conversation(self, orchestrator):
        """Test _process_conversation_message with general conversation."""
        session_id = "test_session"
        message = "Hello, how are you?"
        
        with patch.object(orchestrator, '_handle_general_conversation', return_value="I'm doing well, thank you!"):
            response = await orchestrator._process_conversation_message(session_id, message)
        
        assert response == "I'm doing well, thank you!"
    
    def test_get_conversation_history(self, orchestrator):
        """Test get_conversation_history method."""
        session_id = "test_session"
        
        # Test with non-existent session
        history = orchestrator.get_conversation_history("nonexistent_session")
        assert history == []
        
        # Test with existing session
        message1 = ConversationMessage(
            role="user",
            content="Hello",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        message2 = ConversationMessage(
            role="assistant",
            content="Hi there!",
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        
        orchestrator.conversations[session_id] = [message1, message2]
        
        history = orchestrator.get_conversation_history(session_id)
        assert len(history) == 2
        assert message1 in history
        assert message2 in history
    
    def test_clear_conversation(self, orchestrator):
        """Test clear_conversation method."""
        session_id = "test_session"
        
        # Test with non-existent session
        result = orchestrator.clear_conversation("nonexistent_session")
        assert result is False
        
        # Test with existing session
        orchestrator.conversations[session_id] = [Mock()]
        
        result = orchestrator.clear_conversation(session_id)
        assert result is True
        assert session_id not in orchestrator.conversations
    
    @pytest.mark.asyncio
    async def test_provide_clarification(self, orchestrator):
        """Test provide_clarification method."""
        session_id = "test_session"
        clarification_text = "I need revenue data for Q1-Q4 2023"
        
        # Test with non-existent session
        response = await orchestrator.provide_clarification("nonexistent_session", clarification_text)
        assert response == "No pending clarification found for this session."
        
        # Test with existing session
        execution = TaskExecution(
            session_id=session_id,
            main_request="Original request",
            subtasks=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        orchestrator.active_executions[session_id] = execution
        
        clarification = ClarificationRequest(
            question="What specific data do you need?",
            required=True
        )
        orchestrator.pending_clarifications[session_id] = clarification
        
        with patch.object(orchestrator, 'execute_task', return_value=session_id):
            response = await orchestrator.provide_clarification(session_id, clarification_text)
        
        assert response == "Task restarted with your clarification."
        assert session_id not in orchestrator.pending_clarifications
    
    def test_has_circular_dependencies(self, orchestrator):
        """Test _has_circular_dependencies method."""
        # Test with no circular dependencies
        subtasks = [
            Subtask(id="task1", agent_type=AgentType.PLANNER, description="Task 1", dependencies=[]),
            Subtask(id="task2", agent_type=AgentType.FINANCIAL_ANALYST, description="Task 2", dependencies=["task1"])
        ]
        assert not orchestrator._has_circular_dependencies(subtasks)
        
        # Test with circular dependencies
        subtasks = [
            Subtask(id="task1", agent_type=AgentType.PLANNER, description="Task 1", dependencies=["task2"]),
            Subtask(id="task2", agent_type=AgentType.FINANCIAL_ANALYST, description="Task 2", dependencies=["task1"])
        ]
        assert orchestrator._has_circular_dependencies(subtasks)
        
        # Test with complex circular dependencies
        subtasks = [
            Subtask(id="task1", agent_type=AgentType.PLANNER, description="Task 1", dependencies=["task3"]),
            Subtask(id="task2", agent_type=AgentType.FINANCIAL_ANALYST, description="Task 2", dependencies=["task1"]),
            Subtask(id="task3", agent_type=AgentType.DATA_ANALYST, description="Task 3", dependencies=["task2"])
        ]
        assert orchestrator._has_circular_dependencies(subtasks)
        
        # Test with no dependencies
        subtasks = [
            Subtask(id="task1", agent_type=AgentType.PLANNER, description="Task 1", dependencies=[]),
            Subtask(id="task2", agent_type=AgentType.FINANCIAL_ANALYST, description="Task 2", dependencies=[])
        ]
        assert not orchestrator._has_circular_dependencies(subtasks)
