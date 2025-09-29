"""
Tests for Challenge 1 BaseAgent class.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from challenge1.agents.base_agent import BaseAgent
from challenge1.models import AgentType, TaskStatus, AgentResponse, ToolCall, ClarificationRequest


class TestBaseAgent:
    """Test BaseAgent abstract class."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        class MockAgent(BaseAgent):
            def get_system_prompt(self):
                return "Mock system prompt"
            
            async def process_task(self, task_description, context=None):
                return {"result": "mock_result"}
        
        return MockAgent(AgentType.PLANNER)
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.agent_type == AgentType.PLANNER
        assert mock_agent.context == {}
        assert mock_agent.tool_calls_history == []
        assert mock_agent.max_retries == 3
        assert mock_agent.retry_delay == 1.0
        assert mock_agent.mock_mode is True
    
    def test_get_system_prompt(self, mock_agent):
        """Test get_system_prompt method."""
        prompt = mock_agent.get_system_prompt()
        assert prompt == "Mock system prompt"
    
    @pytest.mark.asyncio
    async def test_process_task(self, mock_agent):
        """Test process_task method."""
        result = await mock_agent.process_task("test task")
        assert result == {"result": "mock_result"}
    
    @pytest.mark.asyncio
    async def test_process_task_with_context(self, mock_agent):
        """Test process_task with context."""
        context = {"key": "value"}
        result = await mock_agent.process_task("test task", context)
        assert result == {"result": "mock_result"}
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_agent):
        """Test successful task execution."""
        response = await mock_agent.execute("test_task", "test description")
        
        assert isinstance(response, AgentResponse)
        assert response.agent_type == AgentType.PLANNER
        assert response.task_id == "test_task"
        assert response.status == TaskStatus.COMPLETED
        assert response.result == {"result": "mock_result"}
        assert response.progress == 100
        assert response.error is None
    
    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_agent):
        """Test task execution with context."""
        context = {"previous_result": "data"}
        response = await mock_agent.execute("test_task", "test description", context)
        
        assert response.status == TaskStatus.COMPLETED
        assert response.result == {"result": "mock_result"}
    
    @pytest.mark.asyncio
    async def test_execute_failure(self, mock_agent):
        """Test task execution failure."""
        # Mock process_task to raise an exception
        mock_agent.process_task = Mock(side_effect=Exception("Test error"))
        
        response = await mock_agent.execute("test_task", "test description")
        
        assert response.status == TaskStatus.FAILED
        assert response.error == "Test error"
        assert response.progress == 0
        assert response.result is None
    
    def test_ask_clarification(self, mock_agent):
        """Test ask_clarification method."""
        question = "What specific data do you need?"
        result = mock_agent.ask_clarification(question)
        assert result == f"Clarification needed: {question}"
    
    def test_update_progress(self, mock_agent):
        """Test update_progress method."""
        # Should not raise an exception
        mock_agent.update_progress(50, "Half done")
    
    def test_get_available_tools(self, mock_agent):
        """Test get_available_tools method."""
        tools = mock_agent.get_available_tools()
        
        assert isinstance(tools, list)
        assert len(tools) == 3
        
        tool_names = [tool["name"] for tool in tools]
        assert "python_executor" in tool_names
        assert "web_search" in tool_names
        assert "data_analysis" in tool_names
        
        # Check python_executor tool structure
        python_tool = next(tool for tool in tools if tool["name"] == "python_executor")
        assert "description" in python_tool
        assert "parameters" in python_tool
        assert python_tool["parameters"]["type"] == "object"
        assert "code" in python_tool["parameters"]["properties"]
        assert "timeout" in python_tool["parameters"]["properties"]
    
    @pytest.mark.asyncio
    async def test_execute_tool_python_executor(self, mock_agent):
        """Test execute_tool with python_executor."""
        tool_call = await mock_agent.execute_tool("python_executor", {
            "code": "print('hello world')",
            "timeout": 30
        })
        
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "python_executor"
        assert tool_call.parameters == {"code": "print('hello world')", "timeout": 30}
        assert tool_call.result is not None
        assert tool_call.error is None
        assert tool_call.result["success"] is True
    
    @pytest.mark.asyncio
    async def test_execute_tool_web_search(self, mock_agent):
        """Test execute_tool with web_search."""
        tool_call = await mock_agent.execute_tool("web_search", {
            "query": "test search",
            "max_results": 5
        })
        
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "web_search"
        assert tool_call.parameters == {"query": "test search", "max_results": 5}
        assert tool_call.result is not None
        assert tool_call.error is None
        assert "query" in tool_call.result
        assert "results" in tool_call.result
    
    @pytest.mark.asyncio
    async def test_execute_tool_data_analysis(self, mock_agent):
        """Test execute_tool with data_analysis."""
        data = [1, 2, 3, 4, 5]
        tool_call = await mock_agent.execute_tool("data_analysis", {
            "data": data,
            "analysis_type": "basic"
        })
        
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "data_analysis"
        assert tool_call.parameters == {"data": data, "analysis_type": "basic"}
        assert tool_call.result is not None
        assert tool_call.error is None
        assert "count" in tool_call.result
        assert "mean" in tool_call.result
    
    @pytest.mark.asyncio
    async def test_execute_tool_unknown_tool(self, mock_agent):
        """Test execute_tool with unknown tool."""
        tool_call = await mock_agent.execute_tool("unknown_tool", {})
        
        assert isinstance(tool_call, ToolCall)
        assert tool_call.tool_name == "unknown_tool"
        assert tool_call.error == "Unknown tool: unknown_tool"
        assert tool_call.result is None
    
    @pytest.mark.asyncio
    async def test_execute_tool_exception(self, mock_agent):
        """Test execute_tool with exception."""
        # Mock _execute_python_code to raise an exception
        mock_agent._execute_python_code = AsyncMock(side_effect=Exception("Test error"))
        
        tool_call = await mock_agent.execute_tool("python_executor", {
            "code": "invalid code"
        })
        
        assert tool_call.error == "Test error"
        assert tool_call.result is None
    
    @pytest.mark.asyncio
    async def test_execute_python_code_success(self, mock_agent):
        """Test _execute_python_code success."""
        result = await mock_agent._execute_python_code("x = 1 + 1")
        
        assert result["success"] is True
        assert "x" in result["result"]
        assert result["result"]["x"] == 2
        assert result["output"] == "Code executed successfully"
    
    @pytest.mark.asyncio
    async def test_execute_python_code_failure(self, mock_agent):
        """Test _execute_python_code failure."""
        result = await mock_agent._execute_python_code("invalid syntax (")
        
        assert result["success"] is False
        assert "error" in result
        assert "output" in result
        assert "Execution failed" in result["output"]
    
    @pytest.mark.asyncio
    async def test_execute_python_code_safe_environment(self, mock_agent):
        """Test _execute_python_code with safe environment."""
        # Test that dangerous operations are not available
        result = await mock_agent._execute_python_code("import os")
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_web_search(self, mock_agent):
        """Test _web_search method."""
        result = await mock_agent._web_search("test query", 3)
        
        assert "query" in result
        assert "results" in result
        assert "total_results" in result
        assert result["query"] == "test query"
        assert len(result["results"]) == 2  # Mock returns 2 results
        assert result["total_results"] == 2
    
    @pytest.mark.asyncio
    async def test_analyze_data_basic(self, mock_agent):
        """Test _analyze_data with basic analysis."""
        data = [1, 2, 3, 4, 5]
        result = await mock_agent._analyze_data(data, "basic")
        
        assert "count" in result
        assert "mean" in result
        assert "min" in result
        assert "max" in result
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["min"] == 1
        assert result["max"] == 5
    
    @pytest.mark.asyncio
    async def test_analyze_data_statistical(self, mock_agent):
        """Test _analyze_data with statistical analysis."""
        data = [1, 2, 3, 4, 5]
        result = await mock_agent._analyze_data(data, "statistical")
        
        assert "count" in result
        assert "mean" in result
        assert "median" in result
        assert "std_deviation" in result
        assert "variance" in result
        assert result["count"] == 5
        assert result["mean"] == 3.0
        assert result["median"] == 3
        assert result["std_deviation"] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_data_empty(self, mock_agent):
        """Test _analyze_data with empty data."""
        result = await mock_agent._analyze_data([], "basic")
        
        assert "error" in result
        assert result["error"] == "No data provided for analysis"
    
    @pytest.mark.asyncio
    async def test_analyze_data_non_numeric(self, mock_agent):
        """Test _analyze_data with non-numeric data."""
        data = ["a", "b", "c"]
        result = await mock_agent._analyze_data(data, "statistical")
        
        assert "error" in result
        assert "No numeric data found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_analyze_data_unknown_type(self, mock_agent):
        """Test _analyze_data with unknown analysis type."""
        data = [1, 2, 3]
        result = await mock_agent._analyze_data(data, "unknown")
        
        assert "error" in result
        assert "Unknown analysis type" in result["error"]
    
    def test_needs_clarification_ambiguous_terms(self, mock_agent):
        """Test needs_clarification with ambiguous terms."""
        # Test with ambiguous term without specific context
        clarification = mock_agent.needs_clarification("analyze some data")
        assert clarification is not None
        assert isinstance(clarification, ClarificationRequest)
        assert "more specific" in clarification.question.lower()
    
    def test_needs_clarification_vague_terms(self, mock_agent):
        """Test needs_clarification with vague terms."""
        # Test with vague term
        clarification = mock_agent.needs_clarification("analyze some data")
        assert clarification is not None
        assert isinstance(clarification, ClarificationRequest)
        assert "some" in clarification.question.lower()
    
    def test_needs_clarification_specific_context(self, mock_agent):
        """Test needs_clarification with specific context."""
        # Test with specific context that should not trigger clarification
        clarification = mock_agent.needs_clarification("analyze the revenue data")
        assert clarification is None
    
    def test_needs_clarification_no_ambiguous_terms(self, mock_agent):
        """Test needs_clarification with no ambiguous terms."""
        clarification = mock_agent.needs_clarification("create a chart for revenue data")
        assert clarification is None
    
    def test_enable_mock_mode(self, mock_agent):
        """Test enable_mock_mode method."""
        mock_agent.enable_mock_mode()
        assert mock_agent.mock_mode is True
    
    def test_disable_mock_mode(self, mock_agent):
        """Test disable_mock_mode method."""
        mock_agent.disable_mock_mode()
        assert mock_agent.mock_mode is False
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, mock_agent):
        """Test execute_with_retry with successful execution."""
        response = await mock_agent.execute_with_retry("test_task", "test description")
        
        assert response.status == TaskStatus.COMPLETED
        assert response.result == {"result": "mock_result"}
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_after_retries(self, mock_agent):
        """Test execute_with_retry with failure after all retries."""
        # Mock execute to always raise an exception
        mock_agent.execute = AsyncMock(side_effect=Exception("Persistent error"))
        
        response = await mock_agent.execute_with_retry("test_task", "test description")
        
        assert response.status == TaskStatus.FAILED
        assert "Persistent error" in response.error
        assert response.progress == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_quota_error(self, mock_agent):
        """Test execute_with_retry with quota error enabling mock mode."""
        # Mock execute to raise quota error on first attempt, then succeed
        call_count = 0
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("insufficient_quota error")
            return AgentResponse(
                agent_type=AgentType.PLANNER,
                task_id="test_task",
                status=TaskStatus.COMPLETED,
                result={"result": "mock_result"}
            )
        
        mock_agent.execute = mock_execute
        
        response = await mock_agent.execute_with_retry("test_task", "test description")
        
        assert response.status == TaskStatus.COMPLETED
        assert mock_agent.mock_mode is True
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_exponential_backoff(self, mock_agent):
        """Test execute_with_retry with exponential backoff."""
        import time
        
        # Mock execute to fail twice, then succeed
        call_count = 0
        start_time = time.time()
        
        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Temporary error")
            return AgentResponse(
                agent_type=AgentType.PLANNER,
                task_id="test_task",
                status=TaskStatus.COMPLETED,
                result={"result": "mock_result"}
            )
        
        mock_agent.execute = mock_execute
        
        response = await mock_agent.execute_with_retry("test_task", "test description")
        
        # Check that it took some time due to backoff
        elapsed_time = time.time() - start_time
        assert elapsed_time > 0.1  # Should have some delay
        
        assert response.status == TaskStatus.COMPLETED
        assert call_count == 3  # Should have retried twice
    
    def test_tool_calls_history(self, mock_agent):
        """Test that tool calls are added to history."""
        assert len(mock_agent.tool_calls_history) == 0
        
        # Execute a tool call
        asyncio.run(mock_agent.execute_tool("python_executor", {"code": "print('test')"}))
        
        assert len(mock_agent.tool_calls_history) == 1
        assert mock_agent.tool_calls_history[0].tool_name == "python_executor"
