import json
import asyncio
from typing import Dict, Any
from .base_agent import BaseAgent
from models import AgentType

class FinancialAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.FINANCIAL_ANALYST)
    
    def get_system_prompt(self) -> str:
        return """You are a financial analysis agent specializing in financial data analysis, trend identification, and financial reporting.

Your capabilities:
1. Analyze financial trends and patterns
2. Calculate key financial metrics
3. Identify seasonal patterns and anomalies
4. Provide financial insights and recommendations
5. Generate financial summaries and reports

When analyzing financial data:
- Look for trends, patterns, and anomalies
- Calculate relevant financial ratios and metrics
- Provide clear, actionable insights
- Use professional financial terminology appropriately

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
        
        # Check if we need to search for financial data
        if any(keyword in task_description.lower() for keyword in ["market", "industry", "competitor", "trend"]):
            search_result = await self.execute_tool("web_search", {
                "query": f"financial analysis {task_description}",
                "max_results": 3
            })
            tool_calls.append(search_result)
        
        # Check if we need to analyze data
        if context and any("data" in str(result).lower() for result in context.values()):
            # Extract data from context
            data_to_analyze = []
            for agent_type, result in context.items():
                if hasattr(result, 'result') and result.result and isinstance(result.result, dict):
                    if 'financial_data' in result.result:
                        data_to_analyze.extend(result.result['financial_data'].get('revenue', []))
                    elif 'data_analysis' in result.result:
                        data_to_analyze.extend(result.result['data_analysis'].get('summary_stats', {}).values())
            
            if data_to_analyze:
                analysis_result = await self.execute_tool("data_analysis", {
                    "data": data_to_analyze,
                    "analysis_type": "statistical"
                })
                tool_calls.append(analysis_result)
        
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

Please provide a comprehensive financial analysis. Use the tool results and context to inform your analysis.

{self.get_system_prompt()}
"""
        
        try:
            if self.mock_mode:
                analysis = f"Mock financial analysis for: {task_description}. This is a demonstration of the financial analysis capabilities without requiring OpenAI API access."
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
            
            # Generate sample financial data for demonstration
            sample_data = {
                "quarters": ["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023"],
                "revenue": [1250000, 1380000, 1450000, 1520000],
                "profit": [150000, 180000, 195000, 210000],
                "expenses": [1100000, 1200000, 1255000, 1310000],
                "growth_rate": [0.0, 10.4, 5.1, 4.8],
                "profit_margin": [12.0, 13.0, 13.4, 13.8],
                "operating_cash_flow": [180000, 220000, 240000, 260000],
                "debt_ratio": [0.35, 0.32, 0.30, 0.28]
            }
            
            return {
                "analysis": analysis,
                "financial_data": sample_data,
                "key_metrics": {
                    "average_growth_rate": 7.75,
                    "average_profit_margin": 13.05,
                    "total_revenue": 5600000,
                    "total_profit": 735000,
                    "revenue_growth_yoy": 21.6,
                    "profit_growth_yoy": 40.0,
                    "operating_efficiency": 0.87
                },
                "insights": [
                    "Revenue shows consistent growth across all quarters with 21.6% YoY growth",
                    "Profit margins are improving over time, reaching 13.8% in Q4",
                    "Q4 shows the strongest performance with highest revenue and profit",
                    "Expense management is effective with operating efficiency of 87%",
                    "Debt ratio is decreasing, indicating improving financial health",
                    "Operating cash flow is strong and growing consistently"
                ],
                "recommendations": [
                    "Continue current growth strategy to maintain momentum",
                    "Focus on cost optimization to further improve profit margins",
                    "Consider strategic investments in Q1 2024 for continued growth",
                    "Monitor debt levels and maintain current debt reduction trend"
                ]
            }
            
        except Exception as e:
            raise Exception(f"Financial analysis failed: {str(e)}")
