"""
Tests for Challenge 1 DataAgent class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from challenge1.agents.data_agent import DataAgent
from challenge1.models import AgentType, TaskStatus


class TestDataAgent:
    """Test DataAgent class."""
    
    @pytest.fixture
    def data_agent(self):
        """Create a DataAgent instance for testing."""
        return DataAgent()
    
    def test_agent_initialization(self, data_agent):
        """Test DataAgent initialization."""
        assert data_agent.agent_type == AgentType.DATA_ANALYST
        assert data_agent.mock_mode is True
    
    def test_get_system_prompt(self, data_agent):
        """Test get_system_prompt method."""
        prompt = data_agent.get_system_prompt()
        
        assert isinstance(prompt, str)
        assert "data analysis agent" in prompt.lower()
        assert "statistical analysis" in prompt.lower()
        assert "data processing" in prompt.lower()
        assert "data interpretation" in prompt.lower()
        assert "prepare data for visualization" in prompt.lower()
        assert "structured data" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_task_mock_mode(self, data_agent):
        """Test process_task in mock mode."""
        result = await data_agent.process_task("analyze customer data")
        
        assert "analysis" in result
        assert "data_analysis" in result
        assert "recommendations" in result
        # In mock mode, clarification_needed might be present if the task is ambiguous
        if "clarification_needed" in result:
            assert result["clarification_needed"] is not None
        
        # Check data_analysis structure
        data_analysis = result["data_analysis"]
        assert "summary_stats" in data_analysis
        assert "trend_analysis" in data_analysis
        assert "correlations" in data_analysis
        assert "patterns" in data_analysis
        
        # Check summary_stats structure
        stats = data_analysis["summary_stats"]
        assert "mean" in stats
        assert "median" in stats
        assert "std_dev" in stats
        assert "variance" in stats
        assert "min" in stats
        assert "max" in stats
        assert "skewness" in stats
        
        # Check patterns structure
        patterns = data_analysis["patterns"]
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        assert all(isinstance(pattern, str) for pattern in patterns)
        
        # Check recommendations structure
        recommendations = result["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_process_task_with_clarification_needed(self, data_agent):
        """Test process_task when clarification is needed."""
        # Mock needs_clarification to return a clarification request
        clarification_request = Mock()
        clarification_request.question = "What specific data do you need to analyze?"
        data_agent.needs_clarification = Mock(return_value=clarification_request)
        
        result = await data_agent.process_task("analyze the data")
        
        assert "analysis" in result
        assert "clarification_needed" in result
        assert "What specific data do you need to analyze?" in result["analysis"]
        assert result["clarification_needed"] == "What specific data do you need to analyze?"
    
    @pytest.mark.asyncio
    async def test_process_task_with_python_execution(self, data_agent):
        """Test process_task with Python code execution."""
        # Mock execute_tool for python_executor
        data_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="python_executor",
            parameters={"code": "import pandas as pd; print('test')"},
            result={"success": True, "output": "test"}
        ))
        
        result = await data_agent.process_task("analyze data using Python")
        
        # Check that python executor was called
        data_agent.execute_tool.assert_called_with("python_executor", {
            "code": "\n# Data analysis code for: analyze data using Python\nimport pandas as pd\nimport numpy as np\nprint('Data analysis with Python')\n",
            "timeout": 30
        })
        
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_data_analysis_tool(self, data_agent):
        """Test process_task with data analysis tool."""
        # Mock context with data
        context = {
            "financial_analyst": Mock(result={
                "financial_data": {
                    "revenue": [1000, 1100, 1200, 1300]
                }
            })
        }
        
        # Mock execute_tool for data_analysis
        data_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [1000, 1100, 1200, 1300], "analysis_type": "statistical"},
            result={"count": 4, "mean": 1150, "std_deviation": 129.1}
        ))
        
        result = await data_agent.process_task("perform statistical analysis", context)
        
        # Check that data analysis was called
        data_agent.execute_tool.assert_called_with("data_analysis", {
            "data": [1000, 1100, 1200, 1300],
            "analysis_type": "statistical"
        })
        
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_openai_api(self, data_agent):
        """Test process_task with OpenAI API (non-mock mode)."""
        data_agent.mock_mode = False
        
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI-generated data analysis"
        
        with patch.object(data_agent.client.chat.completions, 'create', return_value=mock_response):
            result = await data_agent.process_task("analyze data")
        
        assert "analysis" in result
        assert result["analysis"] == "AI-generated data analysis"
        assert "data_summary" in result
        assert "statistical_metrics" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_exception(self, data_agent):
        """Test process_task with OpenAI API exception."""
        data_agent.mock_mode = False
        
        with patch.object(data_agent.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception) as exc_info:
                await data_agent.process_task("analyze data")
            
            assert "Data analysis failed" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_task_empty_context(self, data_agent):
        """Test process_task with empty context."""
        result = await data_agent.process_task("test task", {})
        
        assert "analysis" in result
        assert "data_summary" in result
        assert "statistical_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_process_task_none_context(self, data_agent):
        """Test process_task with None context."""
        result = await data_agent.process_task("test task", None)
        
        assert "analysis" in result
        assert "data_summary" in result
        assert "statistical_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
    
    def test_needs_clarification_inheritance(self, data_agent):
        """Test that needs_clarification is inherited from BaseAgent."""
        # Test with ambiguous terms that should trigger clarification
        clarification = data_agent.needs_clarification("analyze some data")
        assert clarification is not None
        
        # Test with specific context
        clarification = data_agent.needs_clarification("analyze the customer data")
        assert clarification is None
    
    @pytest.mark.asyncio
    async def test_process_task_python_keywords(self, data_agent):
        """Test process_task triggers Python execution for specific keywords."""
        keywords = ["python", "pandas", "numpy", "code", "script"]
        
        data_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="python_executor",
            parameters={"code": "import pandas as pd\nimport numpy as np\nprint('Data analysis with Python')", "timeout": 30},
            result={"success": True, "output": "test"}
        ))
        
        for keyword in keywords:
            result = await data_agent.process_task(f"calculate using {keyword}")
            
            # Check that python executor was called with the correct code
            expected_code = f"""
# Data analysis code for: calculate using {keyword}
import statistics

# Sample data (in real implementation, this would come from context)
data = [100, 120, 110, 140, 160, 150, 130, 170, 180, 165]

# Basic statistics
mean_val = statistics.mean(data)
median_val = statistics.median(data)
stdev_val = statistics.stdev(data)
variance_val = statistics.variance(data)

# Results
results = {{
    "mean": mean_val,
    "median": median_val,
    "standard_deviation": stdev_val,
    "variance": variance_val,
    "min": min(data),
    "max": max(data),
    "count": len(data)
}}

print("Analysis Results:", results)
"""
            data_agent.execute_tool.assert_called_with("python_executor", {
                "code": expected_code,
                "timeout": 30
            })
    
    @pytest.mark.asyncio
    async def test_process_task_data_analysis_keywords(self, data_agent):
        """Test process_task triggers data analysis for data keywords."""
        context = {
            "financial_analyst": Mock(result={
                "financial_data": {
                    "revenue": [1000, 1100, 1200]
                }
            })
        }
        
        data_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [1000, 1100, 1200], "analysis_type": "statistical"},
            result={"count": 3, "mean": 1100}
        ))
        
        result = await data_agent.process_task("perform statistical analysis", context)
        
        # Check that data analysis was called
        data_agent.execute_tool.assert_called_with("data_analysis", {
            "data": [1000, 1100, 1200],
            "analysis_type": "statistical"
        })
    
    @pytest.mark.asyncio
    async def test_process_task_no_tools_triggered(self, data_agent):
        """Test process_task when no tools are triggered."""
        result = await data_agent.process_task("create a simple data report")
        
        assert "analysis" in result
        assert "data_summary" in result
        assert "statistical_metrics" in result
        assert "insights" in result
        assert "recommendations" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_temperature(self, data_agent):
        """Test that OpenAI API is called with correct temperature."""
        data_agent.mock_mode = False
        
        with patch.object(data_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await data_agent.process_task("test task")
            
            # Check that create was called with temperature=0.3
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[1]["temperature"] == 0.3
    
    @pytest.mark.asyncio
    async def test_process_task_system_prompt_included(self, data_agent):
        """Test that system prompt is included in OpenAI API call."""
        data_agent.mock_mode = False
        
        with patch.object(data_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await data_agent.process_task("test task")
            
            # Check that system prompt is included
            call_args = mock_create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "data analysis agent" in messages[0]["content"]
            assert messages[1]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_process_task_data_summary_consistency(self, data_agent):
        """Test that data summary is internally consistent."""
        result = await data_agent.process_task("analyze customer data")
        
        data_summary = result["data_summary"]
        
        # Check that all values are non-negative
        assert data_summary["total_records"] >= 0
        assert data_summary["missing_values"] >= 0
        assert data_summary["outliers"] >= 0
        
        # Check that missing values don't exceed total records
        assert data_summary["missing_values"] <= data_summary["total_records"]
    
    @pytest.mark.asyncio
    async def test_process_task_statistical_metrics_consistency(self, data_agent):
        """Test that statistical metrics are internally consistent."""
        result = await data_agent.process_task("analyze customer data")
        
        stats = result["statistical_metrics"]
        
        # Check that min <= median <= max
        assert stats["min_value"] <= stats["median"]
        assert stats["median"] <= stats["max_value"]
        
        # Check that std_deviation is non-negative
        assert stats["std_deviation"] >= 0
        
        # Check that variance is non-negative
        assert stats["variance"] >= 0
        
        # Check quartiles are in order
        quartiles = stats["quartiles"]
        assert quartiles["q1"] <= quartiles["q2"]
        assert quartiles["q2"] <= quartiles["q3"]
        assert quartiles["q1"] <= quartiles["q3"]
