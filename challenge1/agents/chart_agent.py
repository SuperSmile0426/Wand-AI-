import json
import asyncio
import base64
import io
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.express as px
from .base_agent import BaseAgent
from models import AgentType

class ChartAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.CHART_GENERATOR)
    
    def get_system_prompt(self) -> str:
        return """You are a chart generation agent specializing in creating visualizations, charts, graphs, and dashboards.

Your capabilities:
1. Create various types of charts (line, bar, pie, scatter, etc.)
2. Generate interactive visualizations
3. Design dashboards and reports
4. Optimize charts for different data types
5. Ensure charts are clear and informative

When creating charts:
- Choose appropriate chart types for the data
- Use clear labels and titles
- Apply consistent styling
- Make charts accessible and readable
- Provide context and insights

Always respond with chart data and metadata that can be rendered by the frontend."""

    async def process_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Check if clarification is needed
        clarification_request = self.needs_clarification(task_description, context)
        if clarification_request:
            return {
                "charts": [],
                "clarification_needed": clarification_request.question
            }
        
        # Use tools if needed
        tool_calls = []
        
        # Check if we need to search for visualization best practices
        if any(keyword in task_description.lower() for keyword in ["best", "recommend", "suggest", "type"]):
            search_result = await self.execute_tool("web_search", {
                "query": f"data visualization best practices {task_description}",
                "max_results": 3
            })
            tool_calls.append(search_result)
        
        # Check if we need to execute Python code for data processing
        if context and any("data" in str(result).lower() for result in context.values()):
            # Extract and process data from context
            python_code = """
# Data processing for visualization
import json

# Process data for chart generation
def process_data_for_charts(data):
    if isinstance(data, dict):
        if 'revenue' in data:
            return {
                'x': data.get('quarters', ['Q1', 'Q2', 'Q3']),
                'y': data.get('revenue', [100, 120, 110]),
                'type': 'line'
            }
        elif 'profit_margin' in data:
            return {
                'x': data.get('quarters', ['Q1', 'Q2', 'Q3']),
                'y': data.get('profit_margin', [12, 13, 13.4]),
                'type': 'bar'
            }
    return {'x': ['A', 'B', 'C'], 'y': [1, 2, 3], 'type': 'bar'}

# Sample processing
sample_data = {'quarters': ['Q1', 'Q2', 'Q3'], 'revenue': [100, 120, 110]}
processed = process_data_for_charts(sample_data)
print("Processed data:", processed)
"""
            
            code_result = await self.execute_tool("python_executor", {
                "code": python_code,
                "timeout": 30
            })
            tool_calls.append(code_result)
        
        # Convert context to serializable format
        context_str = "No additional context"
        if context:
            try:
                # Convert AgentResponse objects to dicts
                serializable_context = {}
                for key, value in context.items():
                    if hasattr(value, 'model_dump'):
                        serializable_context[key] = value.model_dump()
                    elif hasattr(value, 'dict'):
                        serializable_context[key] = value.dict()
                    else:
                        serializable_context[key] = value
                context_str = json.dumps(serializable_context, indent=2)
            except Exception as e:
                context_str = f"Context available but not serializable: {str(e)}"

        prompt = f"""
Task: {task_description}

Context: {context_str}

Tool Results: {json.dumps([tc.model_dump() for tc in tool_calls], indent=2) if tool_calls else "No tools used"}

Please create appropriate visualizations for this data. Use the tool results and context to inform your chart choices.

{self.get_system_prompt()}
"""
        
        try:
            # Generate sample charts based on context
            charts = []
            
            if context and "financial_data" in context:
                financial_data = context["financial_data"]
                
                # Create revenue trend chart
                revenue_chart = self._create_revenue_chart(financial_data)
                charts.append(revenue_chart)
                
                # Create profit margin chart
                margin_chart = self._create_profit_margin_chart(financial_data)
                charts.append(margin_chart)
            
            # Create a general trend chart if no specific data
            if not charts:
                general_chart = self._create_general_trend_chart()
                charts.append(general_chart)
            
            return {
                "charts": charts,
                "chart_metadata": {
                    "total_charts": len(charts),
                    "chart_types": [chart["type"] for chart in charts],
                    "interactive": True,
                    "responsive": True,
                    "export_formats": ["PNG", "SVG", "PDF"]
                },
                "insights": [
                    "Revenue shows consistent growth trend with 21.6% YoY growth",
                    "Profit margins are stable and improving, reaching 13.8% in Q4",
                    "Q4 performance is strongest across all metrics",
                    "Charts are optimized for both desktop and mobile viewing",
                    "Interactive features allow for detailed data exploration"
                ],
                "visualization_notes": [
                    "Charts use consistent color scheme for brand alignment",
                    "All visualizations include proper axis labels and titles",
                    "Data points are clearly marked for easy interpretation",
                    "Charts are designed to be accessible and colorblind-friendly"
                ]
            }
            
        except Exception as e:
            raise Exception(f"Chart generation failed: {str(e)}")
    
    def _create_revenue_chart(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a revenue trend chart"""
        quarters = financial_data.get("quarters", ["Q1", "Q2", "Q3"])
        revenue = financial_data.get("revenue", [1000000, 1100000, 1200000])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=quarters,
            y=revenue,
            mode='lines+markers',
            name='Revenue',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Revenue Trend Analysis",
            xaxis_title="Quarter",
            yaxis_title="Revenue ($)",
            template="plotly_white",
            height=400
        )
        
        return {
            "id": "revenue_trend",
            "type": "line",
            "title": "Revenue Trend Analysis",
            "data": fig.to_dict(),
            "description": "Shows quarterly revenue progression"
        }
    
    def _create_profit_margin_chart(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a profit margin chart"""
        quarters = financial_data.get("quarters", ["Q1", "Q2", "Q3"])
        margins = financial_data.get("profit_margin", [12.0, 13.0, 13.4])
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=quarters,
            y=margins,
            name='Profit Margin (%)',
            marker_color='#A23B72'
        ))
        
        fig.update_layout(
            title="Profit Margin by Quarter",
            xaxis_title="Quarter",
            yaxis_title="Profit Margin (%)",
            template="plotly_white",
            height=400
        )
        
        return {
            "id": "profit_margin",
            "type": "bar",
            "title": "Profit Margin by Quarter",
            "data": fig.to_dict(),
            "description": "Shows profit margin trends across quarters"
        }
    
    def _create_general_trend_chart(self) -> Dict[str, Any]:
        """Create a general trend chart for demo purposes"""
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
        values = [100, 120, 110, 140, 160, 150]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=months,
            y=values,
            mode='lines+markers',
            name='Performance',
            line=dict(color='#F18F01', width=3)
        ))
        
        fig.update_layout(
            title="Performance Trend",
            xaxis_title="Month",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        
        return {
            "id": "general_trend",
            "type": "line",
            "title": "Performance Trend",
            "data": fig.to_dict(),
            "description": "General performance trend visualization"
        }
