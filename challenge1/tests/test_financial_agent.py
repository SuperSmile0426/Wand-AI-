"""
Tests for Challenge 1 FinancialAgent class.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from challenge1.agents.financial_agent import FinancialAgent
from challenge1.models import AgentType, TaskStatus


class TestFinancialAgent:
    """Test FinancialAgent class."""
    
    @pytest.fixture
    def financial_agent(self):
        """Create a FinancialAgent instance for testing."""
        return FinancialAgent()
    
    def test_agent_initialization(self, financial_agent):
        """Test FinancialAgent initialization."""
        assert financial_agent.agent_type == AgentType.FINANCIAL_ANALYST
        assert financial_agent.mock_mode is True
    
    def test_get_system_prompt(self, financial_agent):
        """Test get_system_prompt method."""
        prompt = financial_agent.get_system_prompt()
        
        assert isinstance(prompt, str)
        assert "financial analysis agent" in prompt.lower()
        assert "financial data analysis" in prompt.lower()
        assert "trend identification" in prompt.lower()
        assert "financial reporting" in prompt.lower()
        assert "financial trends" in prompt.lower()
        assert "financial metrics" in prompt.lower()
        assert "financial ratios" in prompt.lower()
        assert "structured data" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_task_with_clarification_needed(self, financial_agent):
        """Test process_task when clarification is needed."""
        # Mock needs_clarification to return a clarification request
        clarification_request = Mock()
        clarification_request.question = "What specific financial data do you need?"
        financial_agent.needs_clarification = Mock(return_value=clarification_request)
        
        result = await financial_agent.process_task("analyze the data")
        
        assert "analysis" in result
        assert "clarification_needed" in result
        assert "What specific financial data do you need?" in result["analysis"]
        assert result["clarification_needed"] == "What specific financial data do you need?"
    
    @pytest.mark.asyncio
    async def test_process_task_mock_mode(self, financial_agent):
        """Test process_task in mock mode."""
        result = await financial_agent.process_task("analyze quarterly financial performance")
        
        assert "analysis" in result
        assert "financial_data" in result
        assert "key_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
        # In mock mode, clarification_needed might be present if the task is ambiguous
        if "clarification_needed" in result:
            assert result["clarification_needed"] is not None
        
        # Check financial_data structure
        financial_data = result["financial_data"]
        assert "quarters" in financial_data
        assert "revenue" in financial_data
        assert "profit" in financial_data
        assert "expenses" in financial_data
        assert "growth_rate" in financial_data
        assert "profit_margin" in financial_data
        assert "operating_cash_flow" in financial_data
        assert "debt_ratio" in financial_data
        
        # Check data consistency
        quarters = financial_data["quarters"]
        revenue = financial_data["revenue"]
        profit = financial_data["profit"]
        expenses = financial_data["expenses"]
        
        assert len(quarters) == 4
        assert len(revenue) == 4
        assert len(profit) == 4
        assert len(expenses) == 4
        
        # Check that all values are positive
        assert all(r > 0 for r in revenue)
        assert all(p > 0 for p in profit)
        assert all(e > 0 for e in expenses)
        
        # Check key_metrics structure
        key_metrics = result["key_metrics"]
        assert "average_growth_rate" in key_metrics
        assert "average_profit_margin" in key_metrics
        assert "total_revenue" in key_metrics
        assert "total_profit" in key_metrics
        assert "revenue_growth_yoy" in key_metrics
        assert "profit_growth_yoy" in key_metrics
        assert "operating_efficiency" in key_metrics
        
        # Check insights structure
        insights = result["insights"]
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
        
        # Check recommendations structure
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_process_task_with_web_search(self, financial_agent):
        """Test process_task with web search trigger."""
        # Mock execute_tool for web search
        financial_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="web_search",
            parameters={"query": "financial analysis test task"},
            result={"query": "financial analysis test task", "results": []}
        ))
        
        result = await financial_agent.process_task("analyze market trends")
        
        # Check that web search was called
        financial_agent.execute_tool.assert_called_with("web_search", {
            "query": "financial analysis analyze market trends",
            "max_results": 3
        })
        
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_data_analysis(self, financial_agent):
        """Test process_task with data analysis trigger."""
        # Mock context with data
        context = {
            "data_analyst": Mock(result={
                "data_analysis": {
                    "summary_stats": {"mean": 100, "std": 10}
                }
            })
        }
        
        # Mock execute_tool for data analysis
        financial_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [100, 100], "analysis_type": "statistical"},
            result={"count": 2, "mean": 100}
        ))
        
        result = await financial_agent.process_task("analyze financial data", context)
        
        # Check that data analysis was called
        financial_agent.execute_tool.assert_called_with("data_analysis", {
            "data": [100, 10],  # Extracted from context: mean=100, std=10
            "analysis_type": "statistical"
        })
        
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_financial_data_context(self, financial_agent):
        """Test process_task with financial data in context."""
        # Create a proper mock that can be serialized
        mock_result = Mock()
        mock_result.result = {
            "financial_data": {
                "revenue": [1000, 1100, 1200]
            }
        }
        mock_result.model_dump.return_value = {
            "result": {
                "financial_data": {
                    "revenue": [1000, 1100, 1200]
                }
            }
        }
        
        context = {
            "financial_analyst": mock_result
        }
        
        # Mock execute_tool for data analysis
        financial_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [1000, 1100, 1200], "analysis_type": "statistical"},
            result={"count": 3, "mean": 1100}
        ))
        
        result = await financial_agent.process_task("analyze revenue data", context)
        
        # Check that data analysis was called with revenue data
        financial_agent.execute_tool.assert_called_with("data_analysis", {
            "data": [1000, 1100, 1200],
            "analysis_type": "statistical"
        })
    
    @pytest.mark.asyncio
    async def test_process_task_context_serialization(self, financial_agent):
        """Test process_task with context serialization."""
        # Create mock context with AgentResponse objects
        mock_agent_response = Mock()
        mock_agent_response.model_dump.return_value = {"result": "test_data"}
        
        context = {
            "data_analyst": mock_agent_response,
            "financial_analyst": {"result": "simple_dict"}
        }
        
        result = await financial_agent.process_task("create financial report", context)
        
        assert "analysis" in result
        assert "financial_data" in result
    
    @pytest.mark.asyncio
    async def test_process_task_context_serialization_error(self, financial_agent):
        """Test process_task with context serialization error."""
        # Create mock context that can't be serialized
        mock_agent_response = Mock()
        mock_agent_response.model_dump.side_effect = Exception("Serialization error")
        
        context = {
            "data_analyst": mock_agent_response
        }
        
        result = await financial_agent.process_task("create report", context)
        
        assert "analysis" in result
        assert "financial_data" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_openai_api(self, financial_agent):
        """Test process_task with OpenAI API (non-mock mode)."""
        financial_agent.mock_mode = False
        
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI-generated financial analysis"
        
        with patch.object(financial_agent.client.chat.completions, 'create', return_value=mock_response):
            result = await financial_agent.process_task("analyze financial data")
        
        assert "analysis" in result
        assert result["analysis"] == "AI-generated financial analysis"
        assert "financial_data" in result
        assert "key_metrics" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_exception(self, financial_agent):
        """Test process_task with OpenAI API exception."""
        financial_agent.mock_mode = False
        
        with patch.object(financial_agent.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception) as exc_info:
                await financial_agent.process_task("analyze data")
            
            assert "Financial analysis failed" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_task_empty_context(self, financial_agent):
        """Test process_task with empty context."""
        result = await financial_agent.process_task("test task", {})
        
        assert "analysis" in result
        assert "financial_data" in result
        assert "key_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_process_task_none_context(self, financial_agent):
        """Test process_task with None context."""
        result = await financial_agent.process_task("test task", None)
        
        assert "analysis" in result
        assert "financial_data" in result
        assert "key_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_process_task_web_search_keywords(self, financial_agent):
        """Test process_task triggers web search for specific keywords."""
        keywords = ["market", "industry", "competitor", "trend"]
        
        financial_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="web_search",
            parameters={"query": "financial analysis test", "max_results": 3},
            result={"query": "financial analysis test", "results": []}
        ))
        
        for keyword in keywords:
            result = await financial_agent.process_task(f"analyze {keyword} data")
            
            # Check that web search was called
            financial_agent.execute_tool.assert_called_with("web_search", {
                "query": f"financial analysis analyze {keyword} data",
                "max_results": 3
            })
    
    @pytest.mark.asyncio
    async def test_process_task_data_analysis_keywords(self, financial_agent):
        """Test process_task triggers data analysis for data keywords."""
        context = {
            "data_analyst": Mock(result={
                "data_analysis": {
                    "summary_stats": {"mean": 100}
                }
            })
        }
        
        financial_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [100], "analysis_type": "statistical"},
            result={"count": 1, "mean": 100}
        ))
        
        result = await financial_agent.process_task("analyze data", context)
        
        # Check that data analysis was called
        financial_agent.execute_tool.assert_called_with("data_analysis", {
            "data": [100],
            "analysis_type": "statistical"
        })
    
    @pytest.mark.asyncio
    async def test_process_task_no_tools_triggered(self, financial_agent):
        """Test process_task when no tools are triggered."""
        result = await financial_agent.process_task("create a simple report")
        
        assert "analysis" in result
        assert "financial_data" in result
        assert "key_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_temperature(self, financial_agent):
        """Test that OpenAI API is called with correct temperature."""
        financial_agent.mock_mode = False
        
        with patch.object(financial_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await financial_agent.process_task("test task")
            
            # Check that create was called with temperature=0.3
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[1]["temperature"] == 0.3
    
    @pytest.mark.asyncio
    async def test_process_task_system_prompt_included(self, financial_agent):
        """Test that system prompt is included in OpenAI API call."""
        financial_agent.mock_mode = False
        
        with patch.object(financial_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await financial_agent.process_task("test task")
            
            # Check that system prompt is included
            call_args = mock_create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "financial analysis agent" in messages[0]["content"]
            assert messages[1]["role"] == "user"
    
    def test_needs_clarification_inheritance(self, financial_agent):
        """Test that needs_clarification is inherited from BaseAgent."""
        # Test with ambiguous terms that should trigger clarification
        clarification = financial_agent.needs_clarification("analyze some data")
        assert clarification is not None
        
        # Test with specific context
        clarification = financial_agent.needs_clarification("analyze the revenue data")
        assert clarification is None
    
    @pytest.mark.asyncio
    async def test_process_task_financial_data_consistency(self, financial_agent):
        """Test that financial data is internally consistent."""
        result = await financial_agent.process_task("analyze quarterly performance")
        
        # Check if clarification is needed first
        if "clarification_needed" in result:
            assert result["clarification_needed"] is not None
            return
            
        financial_data = result["financial_data"]
        
        # Check that revenue, profit, and expenses have same length
        assert len(financial_data["revenue"]) == len(financial_data["profit"])
        assert len(financial_data["revenue"]) == len(financial_data["expenses"])
        assert len(financial_data["revenue"]) == len(financial_data["quarters"])
        
        # Check that profit is less than revenue
        for i in range(len(financial_data["revenue"])):
            assert financial_data["profit"][i] < financial_data["revenue"][i]
        
        # Check that expenses + profit approximately equals revenue
        for i in range(len(financial_data["revenue"])):
            total = financial_data["expenses"][i] + financial_data["profit"][i]
            assert abs(total - financial_data["revenue"][i]) < 1000  # Allow small rounding differences
    
    @pytest.mark.asyncio
    async def test_process_task_key_metrics_calculation(self, financial_agent):
        """Test that key metrics are calculated correctly."""
        result = await financial_agent.process_task("analyze quarterly performance")
        
        # Check if clarification is needed first
        if "clarification_needed" in result:
            assert result["clarification_needed"] is not None
            return
            
        financial_data = result["financial_data"]
        key_metrics = result["key_metrics"]
        
        # Check total revenue calculation
        expected_total_revenue = sum(financial_data["revenue"])
        assert key_metrics["total_revenue"] == expected_total_revenue
        
        # Check total profit calculation
        expected_total_profit = sum(financial_data["profit"])
        assert key_metrics["total_profit"] == expected_total_profit
        
        # Check average growth rate calculation
        growth_rates = financial_data["growth_rate"]
        expected_avg_growth = sum(growth_rates) / len(growth_rates)
        assert abs(key_metrics["average_growth_rate"] - expected_avg_growth) < 0.01
        
        # Check average profit margin calculation
        profit_margins = financial_data["profit_margin"]
        expected_avg_margin = sum(profit_margins) / len(profit_margins)
        assert abs(key_metrics["average_profit_margin"] - expected_avg_margin) < 0.01
