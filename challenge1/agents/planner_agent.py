import json
import asyncio
from typing import Dict, Any, List
from .base_agent import BaseAgent
from models import AgentType, Subtask

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentType.PLANNER)
    
    def get_system_prompt(self) -> str:
        return """You are a task planning agent that breaks down complex business requests into subtasks for specialized agents.

Your role:
1. Analyze the user's business request
2. Identify which specialized agents are needed
3. Create a logical sequence of subtasks
4. Define dependencies between tasks
5. Ask clarifying questions if the request is ambiguous

Available agents:
- financial_analyst: For financial data analysis, trend analysis, financial reporting
- data_analyst: For data processing, statistical analysis, data interpretation
- chart_generator: For creating visualizations, charts, graphs, dashboards

Always respond with a JSON object containing:
{
    "subtasks": [
        {
            "id": "unique_task_id",
            "agent_type": "agent_name",
            "description": "clear task description",
            "dependencies": ["list_of_task_ids_this_depends_on"]
        }
    ],
    "clarification_needed": "question_if_ambiguous_or_null_if_clear"
}"""

    async def process_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        # Check if clarification is needed first
        clarification_request = self.needs_clarification(task_description, context)
        if clarification_request:
            return {
                "subtasks": [],
                "clarification_needed": clarification_request.question,
                "clarification_context": clarification_request.context
            }
        
        # Enhanced context sharing
        context_info = ""
        if context:
            context_info = f"\n\nAvailable Context:\n"
            for agent_type, result in context.items():
                if hasattr(result, 'result') and result.result:
                    context_info += f"- {agent_type}: {json.dumps(result.result, indent=2)}\n"
                elif isinstance(result, dict) and 'result' in result:
                    context_info += f"- {agent_type}: {json.dumps(result['result'], indent=2)}\n"
        
        prompt = f"""
User Request: {task_description}{context_info}

Please break this down into subtasks for the appropriate specialized agents. Consider the logical flow and dependencies.
Use any available context from previous agent results to inform your planning.

{self.get_system_prompt()}
"""
        
        try:
            if self.mock_mode:
                # Return mock planning result with no dependencies to avoid deadlock
                return {
                    "subtasks": [
                        {
                            "id": "financial_analysis",
                            "agent_type": "financial_analyst",
                            "description": f"Perform financial analysis for: {task_description}",
                            "dependencies": []
                        },
                        {
                            "id": "data_analysis",
                            "agent_type": "data_analyst", 
                            "description": f"Conduct statistical analysis for: {task_description}",
                            "dependencies": []
                        },
                        {
                            "id": "chart_generation",
                            "agent_type": "chart_generator",
                            "description": f"Create visualizations for: {task_description}",
                            "dependencies": []
                        }
                    ],
                    "clarification_needed": None
                }
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "subtasks": [
                    {
                        "id": "analyze_request",
                        "agent_type": "data_analyst",
                        "description": f"Analyze the request: {task_description}",
                        "dependencies": []
                    }
                ],
                "clarification_needed": None
            }
        except Exception as e:
            raise Exception(f"Planning failed: {str(e)}")
