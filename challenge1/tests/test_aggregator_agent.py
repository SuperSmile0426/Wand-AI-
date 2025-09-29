"""
Tests for Challenge 1 AggregatorAgent class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from challenge1.agents.aggregator_agent import AggregatorAgent
from challenge1.models import AgentType, TaskStatus, AgentResponse


class TestAggregatorAgent:
    """Test AggregatorAgent class."""
    
    @pytest.fixture
    def aggregator_agent(self):
        """Create an AggregatorAgent instance for testing."""
        return AggregatorAgent()
    
    def test_agent_initialization(self, aggregator_agent):
        """Test AggregatorAgent initialization."""
        assert aggregator_agent.agent_type == AgentType.AGGREGATOR
        assert aggregator_agent.mock_mode is True
    
    def test_get_system_prompt(self, aggregator_agent):
        """Test get_system_prompt method."""
        prompt = aggregator_agent.get_system_prompt()
        
        assert isinstance(prompt, str)
        assert "aggregation agent" in prompt.lower()
        assert "combines results" in prompt.lower()
        assert "synthesize" in prompt.lower()
        assert "comprehensive summary" in prompt.lower()
        assert "final response" in prompt.lower()
        assert "well-structured" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_task_mock_mode(self, aggregator_agent):
        """Test process_task in mock mode."""
        # Mock context with agent results that can be serialized
        mock_financial = Mock()
        mock_financial.result = {
            "analysis": "Financial analysis complete",
            "key_metrics": {"revenue": 1000000, "profit": 150000}
        }
        mock_financial.model_dump.return_value = {
            "result": {
                "analysis": "Financial analysis complete",
                "key_metrics": {"revenue": 1000000, "profit": 150000}
            }
        }
        
        mock_data = Mock()
        mock_data.result = {
            "analysis": "Data analysis complete",
            "statistical_metrics": {"mean": 100, "std": 10}
        }
        mock_data.model_dump.return_value = {
            "result": {
                "analysis": "Data analysis complete",
                "statistical_metrics": {"mean": 100, "std": 10}
            }
        }
        
        mock_chart = Mock()
        mock_chart.result = {
            "analysis": "Chart generation complete",
            "chart_type": "line",
            "chart_html": "<div>Chart HTML</div>"
        }
        mock_chart.model_dump.return_value = {
            "result": {
                "analysis": "Chart generation complete",
                "chart_type": "line",
                "chart_html": "<div>Chart HTML</div>"
            }
        }
        
        context = {
            "financial_analyst": mock_financial,
            "data_analyst": mock_data,
            "chart_generator": mock_chart
        }
        
        result = await aggregator_agent.process_task("create a comprehensive financial report", context)
        
        assert "executive_summary" in result
        assert "detailed_analysis" in result
        assert "recommendations" in result
        assert "action_items" in result
        # In mock mode, clarification_needed might be present if the task is ambiguous
        if "clarification_needed" in result:
            assert result["clarification_needed"] is not None
        
        # Check executive_summary structure
        executive_summary = result["executive_summary"]
        assert isinstance(executive_summary, str)
        assert len(executive_summary) > 0
        
        # Check detailed_analysis structure
        detailed_analysis = result["detailed_analysis"]
        assert "financial_performance" in detailed_analysis
        assert "data_trends" in detailed_analysis
        assert "visualization_summary" in detailed_analysis
        
        # Check recommendations structure
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
        
        # Check action_items structure
        action_items = result["action_items"]
        assert isinstance(action_items, list)
        assert len(action_items) > 0
        assert all(isinstance(item, str) for item in action_items)
    
    @pytest.mark.asyncio
    async def test_process_task_with_clarification_needed(self, aggregator_agent):
        """Test process_task when clarification is needed."""
        # Mock needs_clarification to return a clarification request
        clarification_request = Mock()
        clarification_request.question = "What specific aspects should be emphasized in the final report?"
        aggregator_agent.needs_clarification = Mock(return_value=clarification_request)
        
        result = await aggregator_agent.process_task("create a report")
        
        assert "analysis" in result
        assert "clarification_needed" in result
        assert "What specific aspects should be emphasized in the final report?" in result["analysis"]
        assert result["clarification_needed"] == "What specific aspects should be emphasized in the final report?"
    
    @pytest.mark.asyncio
    async def test_process_task_with_empty_context(self, aggregator_agent):
        """Test process_task with empty context."""
        result = await aggregator_agent.process_task("create a comprehensive financial report", {})
        
        assert "analysis" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        assert "appendix" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_none_context(self, aggregator_agent):
        """Test process_task with None context."""
        result = await aggregator_agent.process_task("create a comprehensive financial report", None)
        
        assert "analysis" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        assert "appendix" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_openai_api(self, aggregator_agent):
        """Test process_task with OpenAI API (non-mock mode)."""
        aggregator_agent.mock_mode = False
        
        # Mock context
        context = {
            "financial_analyst": Mock(result={"analysis": "Financial data analyzed"}),
            "data_analyst": Mock(result={"analysis": "Data processed"})
        }
        
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI-generated comprehensive report"
        
        with patch.object(aggregator_agent.client.chat.completions, 'create', return_value=mock_response):
            result = await aggregator_agent.process_task("create a comprehensive financial report", context)
        
        assert "analysis" in result
        assert result["analysis"] == "AI-generated comprehensive report"
        assert "executive_summary" in result
        assert "key_findings" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_exception(self, aggregator_agent):
        """Test process_task with OpenAI API exception."""
        aggregator_agent.mock_mode = False

        with patch.object(aggregator_agent.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception) as exc_info:
                await aggregator_agent.process_task("create a comprehensive financial report")
            
            assert "API Error" in str(exc_info.value)
    
    def test_needs_clarification_inheritance(self, aggregator_agent):
        """Test that needs_clarification is inherited from BaseAgent."""
        # Test with ambiguous terms
        clarification = aggregator_agent.needs_clarification("create some data")
        assert clarification is not None
        
        # Test with specific context
        clarification = aggregator_agent.needs_clarification("create a comprehensive financial report")
        assert clarification is None
    
    @pytest.mark.asyncio
    async def test_process_task_context_serialization(self, aggregator_agent):
        """Test process_task with context serialization."""
        # Create mock context with AgentResponse objects
        mock_agent_response = Mock()
        mock_agent_response.model_dump.return_value = {"result": "test_data"}
        
        context = {
            "financial_analyst": mock_agent_response,
            "data_analyst": {"result": "simple_dict"}
        }
        
        result = await aggregator_agent.process_task("create a comprehensive financial report", context)
        
        assert "analysis" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        assert "appendix" in result
    
    @pytest.mark.asyncio
    async def test_process_task_context_serialization_error(self, aggregator_agent):
        """Test process_task with context serialization error."""
        # Create mock context that can't be serialized
        mock_agent_response = Mock()
        mock_agent_response.model_dump.side_effect = Exception("Serialization error")
        
        context = {
            "financial_analyst": mock_agent_response
        }
        
        result = await aggregator_agent.process_task("create a comprehensive financial report", context)
        
        assert "analysis" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        assert "appendix" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_temperature(self, aggregator_agent):
        """Test that OpenAI API is called with correct temperature."""
        aggregator_agent.mock_mode = False
        
        with patch.object(aggregator_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await aggregator_agent.process_task("test task")
            
            # Check that create was called with temperature=0.3
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[1]["temperature"] == 0.3
    
    @pytest.mark.asyncio
    async def test_process_task_system_prompt_included(self, aggregator_agent):
        """Test that system prompt is included in OpenAI API call."""
        aggregator_agent.mock_mode = False
        
        with patch.object(aggregator_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await aggregator_agent.process_task("test task")
            
            # Check that system prompt is included
            call_args = mock_create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "aggregation agent" in messages[0]["content"]
            assert messages[1]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_process_task_executive_summary_structure(self, aggregator_agent):
        """Test that executive summary has proper structure."""
        result = await aggregator_agent.process_task("create a comprehensive financial report")
        
        executive_summary = result["executive_summary"]
        
        # Check that executive summary is a string
        assert isinstance(executive_summary, str)
        assert len(executive_summary) > 0
    
    @pytest.mark.asyncio
    async def test_process_task_appendix_structure(self, aggregator_agent):
        """Test that appendix has proper structure."""
        result = await aggregator_agent.process_task("create a comprehensive financial report")
        
        appendix = result["appendix"]
        
        # Check that all required fields are present
        assert "agent_results" in appendix
        assert "data_sources" in appendix
        assert "methodology" in appendix
        
        # Check that fields are strings
        assert isinstance(appendix["agent_results"], str)
        assert isinstance(appendix["data_sources"], str)
        assert isinstance(appendix["methodology"], str)
        
        # Check that fields are not empty
        assert len(appendix["agent_results"]) > 0
        assert len(appendix["data_sources"]) > 0
        assert len(appendix["methodology"]) > 0
    
    @pytest.mark.asyncio
    async def test_process_task_key_findings_consistency(self, aggregator_agent):
        """Test that key findings are consistent."""
        result = await aggregator_agent.process_task("create a comprehensive financial report")
        
        key_findings = result["key_findings"]
        
        # Check that it's a list
        assert isinstance(key_findings, list)
        
        # Check that it has findings
        assert len(key_findings) > 0
        
        # Check that all findings are strings
        assert all(isinstance(finding, str) for finding in key_findings)
        
        # Check that findings are not empty
        assert all(len(finding) > 0 for finding in key_findings)
    
    @pytest.mark.asyncio
    async def test_process_task_recommendations_consistency(self, aggregator_agent):
        """Test that recommendations are consistent."""
        result = await aggregator_agent.process_task("create a comprehensive financial report")
        
        recommendations = result["recommendations"]
        
        # Check that it's a list
        assert isinstance(recommendations, list)
        
        # Check that it has recommendations
        assert len(recommendations) > 0
        
        # Check that all recommendations are strings
        assert all(isinstance(rec, str) for rec in recommendations)
        
        # Check that recommendations are not empty
        assert all(len(rec) > 0 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_process_task_different_contexts(self, aggregator_agent):
        """Test process_task with different context types."""
        contexts = [
            {},  # Empty context
            {"financial_analyst": Mock(result={"analysis": "Financial data"})},  # Single agent
            {  # Multiple agents
                "financial_analyst": Mock(result={"analysis": "Financial data"}),
                "data_analyst": Mock(result={"analysis": "Data analysis"}),
                "chart_generator": Mock(result={"analysis": "Chart created"})
            }
        ]
        
        for context in contexts:
            result = await aggregator_agent.process_task("create a comprehensive financial report", context)
            
            assert "analysis" in result
            assert "executive_summary" in result
            assert "key_findings" in result
            assert "recommendations" in result
            assert "appendix" in result
    
    @pytest.mark.asyncio
    async def test_process_task_context_with_agent_responses(self, aggregator_agent):
        """Test process_task with AgentResponse objects in context."""
        # Create mock AgentResponse objects
        financial_response = Mock()
        financial_response.result = {"analysis": "Financial analysis complete"}
        financial_response.model_dump.return_value = {"result": {"analysis": "Financial analysis complete"}}
        
        data_response = Mock()
        data_response.result = {"analysis": "Data analysis complete"}
        data_response.model_dump.return_value = {"result": {"analysis": "Data analysis complete"}}
        
        context = {
            "financial_analyst": financial_response,
            "data_analyst": data_response
        }
        
        result = await aggregator_agent.process_task("create a comprehensive financial report", context)
        
        assert "analysis" in result
        assert "executive_summary" in result
        assert "key_findings" in result
        assert "recommendations" in result
        assert "appendix" in result
