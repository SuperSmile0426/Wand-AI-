import json
import asyncio
from typing import Dict, Any
from .base_agent import BaseAgent
from models import AgentType

class DataAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.DATA_ANALYST)
    
    def get_system_prompt(self) -> str:
        return """You are a data analysis agent specializing in data processing, statistical analysis, and data interpretation.

Your capabilities:
1. Process and clean datasets
2. Perform statistical analysis
3. Identify patterns and correlations
4. Generate data summaries and insights
5. Prepare data for visualization

When analyzing data:
- Apply appropriate statistical methods
- Identify significant patterns and trends
- Provide clear interpretations of findings
- Suggest actionable insights based on data

Always respond with structured data that can be used by other agents or displayed to users."""

    async def process_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Check if clarification is needed
        clarification_request = self.needs_clarification(task_description, context)
        if clarification_request:
            return {
                "analysis": f"Clarification needed: {clarification_request.question}",
                "clarification_needed": clarification_request.question
            }
        
        # Use tools if needed
        tool_calls = []
        
        # Check if we need to execute Python code for analysis
        if any(keyword in task_description.lower() for keyword in ["calculate", "compute", "formula", "equation"]):
            # Generate Python code for the analysis
            python_code = f"""
# Data analysis code for: {task_description}
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
            
            code_result = await self.execute_tool("python_executor", {
                "code": python_code,
                "timeout": 30
            })
            tool_calls.append(code_result)
        
        # Check if we need to search for data analysis methods
        if any(keyword in task_description.lower() for keyword in ["method", "technique", "approach", "best practice"]):
            search_result = await self.execute_tool("web_search", {
                "query": f"data analysis methods {task_description}",
                "max_results": 3
            })
            tool_calls.append(search_result)
        
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

Please provide a comprehensive data analysis. Use the tool results and context to inform your analysis.

{self.get_system_prompt()}
"""
        
        try:
            if self.mock_mode:
                analysis = f"Mock data analysis for: {task_description}. This is a demonstration of the data analysis capabilities without requiring OpenAI API access."
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                analysis = response.choices[0].message.content
            
            # Generate sample data analysis for demonstration
            sample_analysis = {
                "summary_stats": {
                    "mean": 1400000,
                    "median": 1415000,
                    "std_dev": 115000,
                    "min": 1250000,
                    "max": 1520000,
                    "variance": 13225000000,
                    "skewness": -0.15,
                    "kurtosis": 1.8
                },
                "trend_analysis": {
                    "direction": "increasing",
                    "strength": "strong",
                    "consistency": "high",
                    "trend_slope": 90000,
                    "r_squared": 0.94
                },
                "correlations": {
                    "revenue_profit": 0.98,
                    "revenue_expenses": 0.92,
                    "profit_margin": 0.85,
                    "revenue_cash_flow": 0.96,
                    "expenses_profit": -0.45
                },
                "patterns": [
                    "Consistent quarter-over-quarter growth averaging 7.75%",
                    "Strong positive correlation between revenue and profit (0.98)",
                    "Operating efficiency improving over time",
                    "Seasonal patterns show Q4 as strongest quarter",
                    "Debt reduction trend indicates improving financial health"
                ],
                "statistical_tests": {
                    "normality_test": "Data follows normal distribution (p > 0.05)",
                    "stationarity": "Trend-stationary with strong upward movement",
                    "autocorrelation": "Significant positive autocorrelation (0.85)"
                }
            }
            
            return {
                "analysis": analysis,
                "data_analysis": sample_analysis,
                "recommendations": [
                    "Continue current growth strategy with focus on Q4 performance patterns",
                    "Monitor expense ratios closely to maintain profit margin growth",
                    "Leverage strong revenue-profit correlation for predictive modeling",
                    "Consider seasonal adjustments based on Q2-Q4 performance patterns",
                    "Implement data-driven decision making using statistical insights",
                    "Focus on maintaining operating efficiency above 85%"
                ]
            }
            
        except Exception as e:
            raise Exception(f"Data analysis failed: {str(e)}")
