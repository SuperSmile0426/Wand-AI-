import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from .base_agent import BaseAgent
from challenge1.models import AgentType

class AggregatorAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.AGGREGATOR)
    
    def get_system_prompt(self) -> str:
        return """You are an aggregation agent that combines results from multiple specialized agents into a coherent final response.

Your role:
1. Synthesize results from different agents
2. Identify key insights and patterns across results
3. Create a comprehensive summary and report
4. Ensure consistency and coherence
5. Highlight the most important findings

When aggregating results:
- Look for common themes and patterns
- Resolve any conflicts between agent results
- Prioritize the most actionable insights
- Create a clear narrative flow
- Ensure all user requirements are addressed

Always provide a well-structured final response that addresses the original user request."""

    async def process_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Check if clarification is needed
        clarification_request = self.needs_clarification(task_description, context)
        if clarification_request:
            return {
                "analysis": f"Clarification needed: {clarification_request.question}",
                "clarification_needed": clarification_request.question
            }
        
        # Convert AgentResponse objects to dictionaries for JSON serialization
        context_dict = {}
        if context:
            for agent_type, result in context.items():
                try:
                    if hasattr(result, 'model_dump'):
                        context_dict[agent_type] = result.model_dump()
                    elif hasattr(result, 'dict'):
                        context_dict[agent_type] = result.dict()
                    elif hasattr(result, 'result'):
                        # Extract the actual result from Mock objects
                        if hasattr(result.result, '_mock_name'):
                            # If result.result is also a Mock, convert to string
                            context_dict[agent_type] = str(result.result)
                        else:
                            context_dict[agent_type] = result.result
                    elif hasattr(result, '_mock_name'):
                        # Handle Mock objects directly
                        context_dict[agent_type] = str(result)
                    else:
                        context_dict[agent_type] = str(result)
                except Exception as e:
                    context_dict[agent_type] = f"Error serializing result: {str(e)}"
        
        # Final check: Convert any remaining Mock objects to strings
        for key, value in context_dict.items():
            if hasattr(value, '_mock_name'):
                context_dict[key] = str(value)

        prompt = f"""
Original Request: {task_description}

Agent Results to Aggregate:
{json.dumps(context_dict, indent=2) if context_dict else "No context provided"}

Please synthesize these results into a comprehensive final response that addresses the original user request.

{self.get_system_prompt()}
"""
        
        try:
            if self.mock_mode:
                synthesis = f"Mock synthesis for: {task_description}. This is a demonstration of the aggregation capabilities without requiring OpenAI API access. The analysis shows positive trends with consistent growth patterns and strong performance indicators."
            else:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                synthesis = response.choices[0].message.content
                
        except Exception as e:
            # Re-raise the exception for testing purposes
            raise e
        
        # Extract key components from context
        financial_insights = []
        data_insights = []
        charts = []
        
        if context:
            for agent_type, result in context.items():
                # Handle AgentResponse objects
                if hasattr(result, 'result') and result.result:
                    result_data = result.result
                elif isinstance(result, dict) and "result" in result:
                    result_data = result["result"]
                else:
                    result_data = result
                
                if agent_type == "financial_analyst" and result_data and isinstance(result_data, dict):
                    financial_insights = result_data.get("insights", [])
                elif agent_type == "data_analyst" and result_data and isinstance(result_data, dict):
                    data_insights = result_data.get("recommendations", [])
                elif agent_type == "chart_generator" and result_data and isinstance(result_data, dict):
                    charts = result_data.get("charts", [])
        
        return {
                "analysis": synthesis,
                "executive_summary": synthesis,
                "key_findings": financial_insights + data_insights if financial_insights or data_insights else ["No specific findings available"],
                "appendix": {
                    "methodology": "Multi-agent analysis approach",
                    "data_sources": "Internal financial data and market research",
                    "assumptions": "Based on current market conditions and historical trends",
                    "agent_results": "Results from financial_analyst, data_analyst, and chart_generator agents"
                },
                "detailed_analysis": {
                    "financial_performance": self._extract_financial_performance(context),
                    "data_trends": self._extract_data_trends(context),
                    "visualization_summary": self._extract_visualization_summary(context)
                },
                "recommendations": self._generate_recommendations(financial_insights, data_insights),
                "action_items": [
                    "Review the detailed analysis and visualizations",
                    "Implement recommended cost optimization strategies",
                    "Set up quarterly monitoring dashboard",
                    "Schedule follow-up analysis in 3 months",
                    "Consider strategic investments based on growth trends"
                ],
                "metrics_to_track": [
                    "Quarterly revenue growth rate",
                    "Profit margin trends",
                    "Operating efficiency ratio",
                    "Debt-to-equity ratio",
                    "Cash flow consistency"
                ],
                "confidence_score": self._calculate_confidence_score(context),
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "analysis_period": "Q1-Q4 2023",
                    "data_sources": ["Financial records", "Statistical analysis", "Visualization tools"],
                    "methodology": "Multi-agent analysis with specialized AI agents"
                }
            }
    
    def _generate_recommendations(self, financial_insights: List[str], data_insights: List[str]) -> List[str]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        if financial_insights:
            recommendations.extend([
                "Continue monitoring financial performance trends",
                "Maintain current profit margin strategies",
                "Consider expanding successful revenue streams"
            ])
        
        if data_insights:
            recommendations.extend([
                "Implement data-driven decision making processes",
                "Regularly review and update analytical models",
                "Establish key performance indicators for tracking"
            ])
        
        if not recommendations:
            recommendations = [
                "Review the analysis results carefully",
                "Consider additional data collection if needed",
                "Implement regular monitoring and reporting"
            ]
        
        return recommendations
    
    def _calculate_confidence_score(self, context: Dict[str, Any]) -> float:
        """Calculate confidence score based on available data"""
        if not context:
            return 0.5
        
        score = 0.0
        total_agents = 0
        
        for agent_type, result in context.items():
            # Handle AgentResponse objects
            if hasattr(result, 'status') and hasattr(result, 'result'):
                if result.status == "completed" and result.result:
                    score += 1.0
            elif isinstance(result, dict):
                if result.get("status") == "completed" and result.get("result"):
                    score += 1.0
            total_agents += 1
        
        if total_agents == 0:
            return 0.5
        
        return min(score / total_agents, 1.0)
    
    def _extract_financial_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial performance metrics from context"""
        financial_data = {}
        if context:
            for agent_type, result in context.items():
                if agent_type == "financial_analyst" and hasattr(result, 'result') and result.result:
                    financial_data = result.result.get("key_metrics", {})
                    break
        return financial_data
    
    def _extract_data_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data trends from context"""
        data_trends = {}
        if context:
            for agent_type, result in context.items():
                if agent_type == "data_analyst" and hasattr(result, 'result') and result.result:
                    data_trends = result.result.get("data_analysis", {})
                    break
        return data_trends
    
    def _extract_visualization_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visualization summary from context"""
        viz_summary = {}
        if context:
            for agent_type, result in context.items():
                if agent_type == "chart_generator" and hasattr(result, 'result') and result.result:
                    viz_summary = {
                        "total_charts": len(result.result.get("charts", [])),
                        "chart_types": result.result.get("chart_metadata", {}).get("chart_types", []),
                        "insights": result.result.get("insights", [])
                    }
                    break
        return viz_summary
