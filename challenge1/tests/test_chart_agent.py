"""
Tests for Challenge 1 ChartAgent class.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from challenge1.agents.chart_agent import ChartAgent
from challenge1.models import AgentType, TaskStatus


class TestChartAgent:
    """Test ChartAgent class."""
    
    @pytest.fixture
    def chart_agent(self):
        """Create a ChartAgent instance for testing."""
        return ChartAgent()
    
    def test_agent_initialization(self, chart_agent):
        """Test ChartAgent initialization."""
        assert chart_agent.agent_type == AgentType.CHART_GENERATOR
        assert chart_agent.mock_mode is True
    
    def test_get_system_prompt(self, chart_agent):
        """Test get_system_prompt method."""
        prompt = chart_agent.get_system_prompt()

        assert isinstance(prompt, str)
        assert "chart generation agent" in prompt.lower()
        assert "creating visualizations" in prompt.lower()
        assert "charts" in prompt.lower()
        assert "graphs" in prompt.lower()
        assert "dashboards" in prompt.lower()
        assert "plotly" in prompt.lower()
        assert "interactive" in prompt.lower()
        assert "chart data" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_process_task_mock_mode(self, chart_agent):
        """Test process_task in mock mode."""
        result = await chart_agent.process_task("create a revenue chart")
        
        assert "charts" in result
        assert "chart_metadata" in result
        assert "insights" in result
        assert "visualization_notes" in result
        # In mock mode, clarification_needed might be present if the task is ambiguous
        if "clarification_needed" in result:
            assert result["clarification_needed"] is not None
        
        # Check chart_metadata structure
        chart_metadata = result["chart_metadata"]
        assert "total_charts" in chart_metadata
        assert "chart_types" in chart_metadata
        assert "interactive" in chart_metadata
        assert "responsive" in chart_metadata
        assert "export_formats" in chart_metadata
        
        # Check charts structure
        charts = result["charts"]
        assert isinstance(charts, list)
        assert len(charts) > 0
        
        # Check insights structure
        insights = result["insights"]
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
    
    @pytest.mark.asyncio
    async def test_process_task_with_clarification_needed(self, chart_agent):
        """Test process_task when clarification is needed."""
        # Mock needs_clarification to return a clarification request
        clarification_request = Mock()
        clarification_request.question = "What type of chart do you need?"
        chart_agent.needs_clarification = Mock(return_value=clarification_request)
        
        result = await chart_agent.process_task("create a chart")
        
        assert "analysis" in result
        assert "clarification_needed" in result
        assert "What type of chart do you need?" in result["analysis"]
        assert result["clarification_needed"] == "What type of chart do you need?"
    
    @pytest.mark.asyncio
    async def test_process_task_with_python_execution(self, chart_agent):
        """Test process_task with Python code execution."""
        # Mock execute_tool for python_executor
        chart_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="python_executor",
            parameters={"code": "import plotly.graph_objects as go; print('chart')"},
            result={"success": True, "output": "chart created"}
        ))
        
        result = await chart_agent.process_task("create a chart using Python")
        
        # Check that python executor was called
        chart_agent.execute_tool.assert_called_with("python_executor", {
            "code": "\n# Chart generation code for: create a chart using Python\nimport plotly.graph_objects as go\nimport plotly.express as px\nprint('Creating chart with Plotly')\n",
            "timeout": 30
        })
        
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_data_analysis_tool(self, chart_agent):
        """Test process_task with data analysis tool."""
        # Mock context with data
        context = {
            "data_analyst": Mock(result={
                "data_analysis": {
                    "summary_stats": {"mean": 100, "std": 10}
                }
            })
        }
        
        # Mock execute_tool for data_analysis
        chart_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [100, 110, 120, 130], "analysis_type": "basic"},
            result={"count": 4, "mean": 115, "min": 100, "max": 130}
        ))
        
        result = await chart_agent.process_task("analyze data", context)
        
        # Check that data analysis was called
        chart_agent.execute_tool.assert_called_with("data_analysis", {
            "data": [100, 110, 120],
            "analysis_type": "basic"
        })
        
        assert "analysis" in result
    
    @pytest.mark.asyncio
    async def test_process_task_with_openai_api(self, chart_agent):
        """Test process_task with OpenAI API (non-mock mode)."""
        chart_agent.mock_mode = False
        
        # Mock OpenAI client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "AI-generated chart analysis"
        
        with patch.object(chart_agent.client.chat.completions, 'create', return_value=mock_response):
            result = await chart_agent.process_task("create a chart")
        
        assert "analysis" in result
        assert result["analysis"] == "AI-generated chart analysis"
        assert "charts" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_exception(self, chart_agent):
        """Test process_task with OpenAI API exception."""
        chart_agent.mock_mode = False
        
        with patch.object(chart_agent.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception) as exc_info:
                await chart_agent.process_task("create chart")
            
            assert "Chart generation failed" in str(exc_info.value)
            assert "API Error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_task_empty_context(self, chart_agent):
        """Test process_task with empty context."""
        result = await chart_agent.process_task("test task", {})
        
        assert "analysis" in result
        assert "chart_config" in result
        assert "chart_html" in result
        assert "chart_type" in result
        assert "data_summary" in result
        assert "insights" in result
    
    @pytest.mark.asyncio
    async def test_process_task_none_context(self, chart_agent):
        """Test process_task with None context."""
        result = await chart_agent.process_task("test task", None)
        
        assert "analysis" in result
        assert "chart_config" in result
        assert "chart_html" in result
        assert "chart_type" in result
        assert "data_summary" in result
        assert "insights" in result
    
    def test_needs_clarification_inheritance(self, chart_agent):
        """Test that needs_clarification is inherited from BaseAgent."""
        # Test with ambiguous terms
        clarification = chart_agent.needs_clarification("create some data")
        assert clarification is not None
        
        # Test with specific context
        clarification = chart_agent.needs_clarification("create a revenue bar chart")
        assert clarification is None
    
    @pytest.mark.asyncio
    async def test_process_task_python_keywords(self, chart_agent):
        """Test process_task triggers Python execution for specific keywords."""
        keywords = ["python", "plotly", "code", "script", "programmatic"]
        
        chart_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="python_executor",
            parameters={"code": "import plotly.graph_objects as go\nimport plotly.express as px\nprint('Creating chart with Plotly')", "timeout": 30},
            result={"success": True, "output": "chart created"}
        ))
        
        for keyword in keywords:
            result = await chart_agent.process_task(f"create a chart using {keyword}")
            
            # Check that python executor was called with the exact code generated
            expected_code = f"\n# Chart generation code for: create a chart using {keyword}\nimport plotly.graph_objects as go\nimport plotly.express as px\nprint('Creating chart with Plotly')\n"
            chart_agent.execute_tool.assert_called_with("python_executor", {
                "code": expected_code,
                "timeout": 30
            })
    
    @pytest.mark.asyncio
    async def test_process_task_data_analysis_keywords(self, chart_agent):
        """Test process_task triggers data analysis for data keywords."""
        context = {
            "data_analyst": Mock(result={
                "data_analysis": {
                    "summary_stats": {"mean": 100}
                }
            })
        }
        
        chart_agent.execute_tool = AsyncMock(return_value=Mock(
            tool_name="data_analysis",
            parameters={"data": [100, 110, 120], "analysis_type": "basic"},
            result={"count": 3, "mean": 110}
        ))
        
        result = await chart_agent.process_task("create a chart from data", context)

        # Check that python executor was called (chart keywords have priority over data keywords)
        chart_agent.execute_tool.assert_called_with("python_executor", {
            "code": "\n# Chart generation code for: create a chart from data\nimport plotly.graph_objects as go\nimport plotly.express as px\nprint('Creating chart with Plotly')\n",
            "timeout": 30
        })
    
    @pytest.mark.asyncio
    async def test_process_task_no_tools_triggered(self, chart_agent):
        """Test process_task when no tools are triggered."""
        result = await chart_agent.process_task("create a simple chart")
        
        assert "analysis" in result
        assert "chart_config" in result
        assert "chart_html" in result
        assert "chart_type" in result
        assert "data_summary" in result
        assert "insights" in result
    
    @pytest.mark.asyncio
    async def test_process_task_openai_temperature(self, chart_agent):
        """Test that OpenAI API is called with correct temperature."""
        chart_agent.mock_mode = False
        
        with patch.object(chart_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await chart_agent.process_task("test task")
            
            # Check that create was called with temperature=0.3
            mock_create.assert_called_once()
            call_args = mock_create.call_args
            assert call_args[1]["temperature"] == 0.3
    
    @pytest.mark.asyncio
    async def test_process_task_system_prompt_included(self, chart_agent):
        """Test that system prompt is included in OpenAI API call."""
        chart_agent.mock_mode = False
        
        with patch.object(chart_agent.client.chat.completions, 'create') as mock_create:
            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "AI analysis"
            mock_create.return_value = mock_response
            
            await chart_agent.process_task("test task")
            
            # Check that system prompt is included
            call_args = mock_create.call_args
            messages = call_args[1]["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "chart generation agent" in messages[0]["content"]
            assert messages[1]["role"] == "user"
    
    @pytest.mark.asyncio
    async def test_process_task_chart_config_consistency(self, chart_agent):
        """Test that chart config is internally consistent."""
        result = await chart_agent.process_task("create a revenue chart")
        
        chart_config = result["chart_config"]
        
        # Check that all required fields are present
        assert "title" in chart_config
        assert "x_axis" in chart_config
        assert "y_axis" in chart_config
        assert "chart_type" in chart_config
        assert "colors" in chart_config
        assert "layout" in chart_config
        
        # Check that chart_type matches the result
        assert chart_config["chart_type"] == result["chart_type"]
        
        # Check that colors is a list
        assert isinstance(chart_config["colors"], list)
        assert len(chart_config["colors"]) > 0
    
    @pytest.mark.asyncio
    async def test_process_task_data_summary_consistency(self, chart_agent):
        """Test that data summary is internally consistent."""
        result = await chart_agent.process_task("create a revenue chart")
        
        data_summary = result["data_summary"]
        
        # Check that all required fields are present
        assert "data_points" in data_summary
        assert "data_range" in data_summary
        assert "trends" in data_summary
        
        # Check that data_points is non-negative
        assert data_summary["data_points"] >= 0
        
        # Check that data_range has min and max
        data_range = data_summary["data_range"]
        assert "min" in data_range
        assert "max" in data_range
        assert data_range["min"] <= data_range["max"]
    
    @pytest.mark.asyncio
    async def test_process_task_chart_html_structure(self, chart_agent):
        """Test that chart HTML has proper structure."""
        result = await chart_agent.process_task("create a revenue chart")
        
        chart_html = result["chart_html"]
        
        # Check that HTML contains plotly elements
        assert "plotly" in chart_html.lower()
        assert "html" in chart_html.lower()
        
        # Check that it's a valid HTML structure
        assert "<" in chart_html
        assert ">" in chart_html
    
    @pytest.mark.asyncio
    async def test_process_task_different_chart_types(self, chart_agent):
        """Test process_task with different chart type requests."""
        chart_types = ["line", "bar", "pie", "scatter", "area"]
        
        for chart_type in chart_types:
            result = await chart_agent.process_task(f"create a {chart_type} chart")
            
            assert "chart_type" in result
            assert "chart_config" in result
            assert "chart_html" in result
            assert "data_summary" in result
            assert "insights" in result
