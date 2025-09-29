"""
Tests for Challenge 1 PlannerAgent class.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from challenge1.agents.planner_agent import PlannerAgent
from challenge1.models import AgentType, TaskStatus, ClarificationRequest


class TestPlannerAgent:
    """Test PlannerAgent class."""
    
    @pytest.fixture
    def planner_agent(self):
        """Create a PlannerAgent instance for testing."""
        return PlannerAgent()
    
    def test_agent_initialization(self, planner_agent):
        """Test PlannerAgent initialization."""
        assert planner_agent.agent_type == AgentType.PLANNER
        assert planner_agent.mock_mode is True
    
    def test_get_system_prompt(self, planner_agent):
        """Test get_system_prompt method."""
        prompt = planner_agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert "task planning agent" in prompt.lower()
        assert "breaks down complex business requests" in prompt.lower()
        assert "financial_analyst" in prompt
        assert "data_analyst" in prompt
        assert "chart_generator" in prompt
        assert "json" in prompt.lower()
        assert "subtasks" in prompt
        assert "clarification_needed" in prompt
    
    @pytest.mark.asyncio
    async def test_process_task_with_clarification_needed(self, planner_agent):
        """Test process_task when clarification is needed."""
        # Mock needs_clarification to return a clarification request
        clarification_request = ClarificationRequest(
            question="What specific data do you need?",
            context={"current_task": "planning"},
            required=True
        )
        planner_agent.needs_clarification = Mock(return_value=clarification_request)
        
        result = await planner_agent.process_task("analyze the data")
        
        assert "subtasks" in result
        assert "clarification_needed" in result
        assert "clarification_context" in result
        assert result["subtasks"] == []
        assert result["clarification_needed"] == "What specific data do you need?"
        assert result["clarification_context"] == {"current_task": "planning"}
    
    @pytest.mark.asyncio
    async def test_process_task_with_context(self, planner_agent):
        """Test process_task with context."""
        context = {
            "financial_analyst": Mock(result={"analysis": "financial data"}),
            "data_analyst": Mock(result={"data_analysis": {"summary_stats": {"mean": 100}}})
        }
        
        result = await planner_agent.process_task("create a comprehensive report", context)
        
        assert "subtasks" in result
        assert "clarification_needed" in result
        assert result["clarification_needed"] is None
        assert isinstance(result["subtasks"], list)
        assert len(result["subtasks"]) == 3  # Mock returns 3 subtasks
    
    @pytest.mark.asyncio
    async def test_process_task_mock_mode(self, planner_agent):
        """Test process_task in mock mode."""
        result = await planner_agent.process_task("analyze quarterly financial performance")
        
        assert "subtasks" in result
        assert "clarification_needed" in result
        # In mock mode, clarification_needed might be present if the task is ambiguous
        if result["clarification_needed"] is not None:
            assert isinstance(result["clarification_needed"], str)
        
        # Check subtasks structure
        subtasks = result["subtasks"]
        assert len(subtasks) == 3
        
        # Check each subtask has required fields
        for subtask in subtasks:
            assert "id" in subtask
            assert "agent_type" in subtask
            assert "description" in subtask
            assert "dependencies" in subtask
            assert isinstance(subtask["dependencies"], list)
        
        # Check agent types
        agent_types = [subtask["agent_type"] for subtask in subtasks]
        assert "financial_analyst" in agent_types
        assert "data_analyst" in agent_types
        assert "chart_generator" in agent_types
    
    @pytest.mark.asyncio
    async def test_process_task_with_openai_api(self, planner_agent):
        """Test process_task with OpenAI API (non-mock mode)."""
        planner_agent.mock_mode = False
        
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "subtasks": [
                {
                    "id": "plan_analysis",
                    "agent_type": "financial_analyst",
                    "description": "Plan financial analysis",
                    "dependencies": []
                }
            ],
            "clarification_needed": None
        })
        
        with patch.object(planner_agent.client.chat.completions, 'create', return_value=mock_response):
            result = await planner_agent.process_task("analyze financial data")
        
        assert "subtasks" in result
        assert "clarification_needed" in result
        assert result["clarification_needed"] is None
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["id"] == "plan_analysis"
    
    @pytest.mark.asyncio
    async def test_process_task_json_decode_error(self, planner_agent):
        """Test process_task with JSON decode error."""
        planner_agent.mock_mode = False
        
        # Mock OpenAI client to return invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        
        with patch.object(planner_agent.client.chat.completions, 'create', return_value=mock_response):
            result = await planner_agent.process_task("analyze data")
        
        # Should fallback to default subtasks
        assert "subtasks" in result
        assert "clarification_needed" in result
        assert result["clarification_needed"] is None
        assert len(result["subtasks"]) == 1
        assert result["subtasks"][0]["agent_type"] == "data_analyst"
    
    @pytest.mark.asyncio
    async def test_process_task_openai_exception(self, planner_agent):
        """Test process_task with OpenAI API exception."""
        planner_agent.mock_mode = False
        
        with patch.object(planner_agent.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception) as exc_info:
                await planner_agent.process_task("analyze data")
            
            assert "Planning failed" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_task_context_serialization(self, planner_agent):
        """Test process_task with complex context serialization."""
        # Create mock context with AgentResponse objects
        mock_agent_response = Mock()
        mock_agent_response.result = "test_data"
        mock_agent_response.model_dump.return_value = {"result": "test_data"}

        context = {
            "financial_analyst": mock_agent_response,
            "data_analyst": {"result": "simple_dict"}
        }

        result = await planner_agent.process_task("create report", context)
        
        assert "subtasks" in result
        assert result["clarification_needed"] is None
    
    @pytest.mark.asyncio
    async def test_process_task_context_serialization_error(self, planner_agent):
        """Test process_task with context serialization error."""
        # Create mock context that can't be serialized
        mock_agent_response = Mock()
        mock_agent_response.model_dump.side_effect = Exception("Serialization error")
        
        context = {
            "financial_analyst": mock_agent_response
        }
        
        result = await planner_agent.process_task("create report", context)
        
        assert "subtasks" in result
        assert result["clarification_needed"] is None
    
    def test_needs_clarification_inheritance(self, planner_agent):
        """Test that needs_clarification is inherited from BaseAgent."""
        # Test with ambiguous terms that should trigger clarification
        clarification = planner_agent.needs_clarification("analyze some data")
        assert clarification is not None
        assert isinstance(clarification, ClarificationRequest)
        
        # Test with specific context
        clarification = planner_agent.needs_clarification("analyze the revenue data")
        assert clarification is None
    
    @pytest.mark.asyncio
    async def test_process_task_empty_context(self, planner_agent):
        """Test process_task with empty context."""
        result = await planner_agent.process_task("test task", {})
        
        assert "subtasks" in result
        assert "clarification_needed" in result
        assert result["clarification_needed"] is None
        assert len(result["subtasks"]) == 3
    
    @pytest.mark.asyncio
    async def test_process_task_none_context(self, planner_agent):
        """Test process_task with None context."""
        result = await planner_agent.process_task("test task", None)
        
        assert "subtasks" in result
        assert "clarification_needed" in result
        assert result["clarification_needed"] is None
        assert len(result["subtasks"]) == 3
    
    @pytest.mark.asyncio
    async def test_process_task_mock_subtasks_structure(self, planner_agent):
        """Test the structure of mock subtasks."""
        result = await planner_agent.process_task("analyze quarterly performance")
        
        subtasks = result["subtasks"]
        
        # Check that all subtasks have no dependencies (to avoid deadlock)
        for subtask in subtasks:
            assert subtask["dependencies"] == []
        
        # Check that descriptions contain the task description
        for subtask in subtasks:
            assert "analyze quarterly performance" in subtask["description"]
    
    @pytest.mark.asyncio
    async def test_process_task_different_task_descriptions(self, planner_agent):
        """Test process_task with different task descriptions."""
        descriptions = [
            "create a financial dashboard",
            "analyze customer data trends",
            "generate quarterly reports",
            "build a data visualization"
        ]
        
        for description in descriptions:
            result = await planner_agent.process_task(description)
            
            assert "subtasks" in result
            assert "clarification_needed" in result
            assert result["clarification_needed"] is None
            assert len(result["subtasks"]) == 3
            
            # Check that descriptions are included in subtask descriptions
            for subtask in result["subtasks"]:
                assert description in subtask["description"]
    
    @pytest.mark.asyncio
    async def test_process_task_openai_temperature(self, planner_agent):
        """Test that OpenAI API is called with correct temperature."""
        planner_agent.mock_mode = False
        
        with patch.object(planner_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "subtasks": [],
                "clarification_needed": None
            })
            mock_create.return_value = mock_response
            
            await planner_agent.process_task("test task")
            
            # Check that create was called with temperature=0.3
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[1]["temperature"] == 0.3
    
    @pytest.mark.asyncio
    async def test_process_task_system_prompt_included(self, planner_agent):
        """Test that system prompt is included in OpenAI API call."""
        planner_agent.mock_mode = False
        
        with patch.object(planner_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = json.dumps({
                "subtasks": [],
                "clarification_needed": None
            })
            mock_create.return_value = mock_response
            
            await planner_agent.process_task("test task")
            
            # Check that system prompt is included
            call_args = mock_create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "task planning agent" in messages[0]["content"]
            assert messages[1]["role"] == "user"
