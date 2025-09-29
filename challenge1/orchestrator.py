import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from challenge1.models import TaskExecution, Subtask, TaskStatus, AgentType, ProgressUpdate, ConversationMessage, ClarificationRequest
import os

# Always use real agents, not mock
from challenge1.agents.planner_agent import PlannerAgent
from challenge1.agents.financial_agent import FinancialAgent
from challenge1.agents.data_agent import DataAgent
from challenge1.agents.chart_agent import ChartAgent
from challenge1.agents.aggregator_agent import AggregatorAgent
USE_MOCK = False

class TaskOrchestrator:
    def __init__(self):
        self.agents = {
            AgentType.PLANNER: PlannerAgent(),
            AgentType.FINANCIAL_ANALYST: FinancialAgent(),
            AgentType.DATA_ANALYST: DataAgent(),
            AgentType.CHART_GENERATOR: ChartAgent(),
            AgentType.AGGREGATOR: AggregatorAgent()
        }
        self.active_executions: Dict[str, TaskExecution] = {}
        self.progress_callbacks: List[callable] = []
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.pending_clarifications: Dict[str, ClarificationRequest] = {}
    
    def add_progress_callback(self, callback: callable):
        """Add a callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    async def _notify_progress(self, update: ProgressUpdate):
        """Notify all registered callbacks about progress updates"""
        for callback in self.progress_callbacks:
            try:
                await callback(update)
            except Exception as e:
                print(f"Error in progress callback: {e}")
    
    async def execute_task(self, user_request: str, session_id: Optional[str] = None) -> str:
        """Execute a complete task from start to finish"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create initial execution
        execution = TaskExecution(
            session_id=session_id,
            main_request=user_request,
            subtasks=[],
            status=TaskStatus.PENDING,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
        
        self.active_executions[session_id] = execution
        
        try:
            # Step 1: Planning
            await self._notify_progress(ProgressUpdate(
                session_id=session_id,
                task_id="planning",
                agent_type=AgentType.PLANNER,
                status=TaskStatus.IN_PROGRESS,
                progress=10,
                message="Planning task breakdown..."
            ))
            
            planner_result = await self.agents[AgentType.PLANNER].execute_with_retry(
                "planning", user_request
            )
            
            # Send planning completion update
            if planner_result.status == TaskStatus.FAILED:
                raise Exception(f"Planning failed: {planner_result.error}")
            
            # Convert result to serializable format
            serializable_result = planner_result.result
            if hasattr(planner_result.result, 'model_dump'):
                serializable_result = planner_result.result.model_dump()
            elif hasattr(planner_result.result, 'dict'):
                serializable_result = planner_result.result.dict()
            
            await self._notify_progress(ProgressUpdate(
                session_id=session_id,
                task_id="planning",
                agent_type=AgentType.PLANNER,
                status=TaskStatus.COMPLETED,
                progress=15,
                message="Planning completed",
                result=serializable_result
            ))
            
            # Check if clarification is needed
            if planner_result.result and planner_result.result.get("clarification_needed"):
                execution.status = TaskStatus.PENDING
                execution.updated_at = datetime.now().isoformat()
                
                # Store clarification request
                clarification_request = ClarificationRequest(
                    question=planner_result.result.get("clarification_needed"),
                    context=planner_result.result.get("clarification_context"),
                    required=True
                )
                self.pending_clarifications[session_id] = clarification_request
                
                await self._notify_progress(ProgressUpdate(
                    session_id=session_id,
                    task_id="planning",
                    agent_type=AgentType.PLANNER,
                    status=TaskStatus.PENDING,
                    progress=10,
                    message="Clarification needed",
                    clarification_needed=clarification_request
                ))
                
                return session_id  # Return for clarification
            
            # Create subtasks with validation
            subtasks_data = planner_result.result.get("subtasks", [])
            subtasks = []
            
            # Validate subtasks to prevent hallucinations
            if not subtasks_data or not isinstance(subtasks_data, list):
                raise Exception("Invalid subtasks format from planner")
            
            for i, task_data in enumerate(subtasks_data):
                if not isinstance(task_data, dict):
                    raise Exception(f"Invalid subtask format at index {i}")
                
                required_fields = ["id", "agent_type", "description"]
                for field in required_fields:
                    if field not in task_data:
                        raise Exception(f"Missing required field '{field}' in subtask {i}")
                
                # Validate agent type
                try:
                    agent_type = AgentType(task_data["agent_type"])
                except ValueError:
                    raise Exception(f"Invalid agent type '{task_data['agent_type']}' in subtask {i}")
                
                # Validate dependencies
                dependencies = task_data.get("dependencies", [])
                if not isinstance(dependencies, list):
                    dependencies = []
                
                subtask = Subtask(
                    id=task_data["id"],
                    agent_type=agent_type,
                    description=task_data["description"],
                    dependencies=dependencies
                )
                subtasks.append(subtask)
            
            # Check for circular dependencies
            if self._has_circular_dependencies(subtasks):
                raise Exception("Circular dependencies detected in subtasks")
            
            execution.subtasks = subtasks
            execution.status = TaskStatus.IN_PROGRESS
            execution.updated_at = datetime.now().isoformat()
            
            # Step 2: Execute subtasks
            results = {}
            completed_tasks = set()
            
            while len(completed_tasks) < len(subtasks):
                progress_made = False
                
                for subtask in subtasks:
                    if subtask.id in completed_tasks:
                        continue
                    
                    # Check if dependencies are met
                    if all(dep in completed_tasks for dep in subtask.dependencies):
                        await self._notify_progress(ProgressUpdate(
                            session_id=session_id,
                            task_id=subtask.id,
                            agent_type=subtask.agent_type,
                            status=TaskStatus.IN_PROGRESS,
                            progress=20 + (len(completed_tasks) * 60 // len(subtasks)),
                            message=f"Executing {subtask.agent_type.value} task..."
                        ))
                        
                        # Execute the subtask with retry logic
                        agent_result = await self.agents[subtask.agent_type].execute_with_retry(
                            subtask.id, subtask.description, results
                        )
                        
                        subtask.status = agent_result.status
                        subtask.result = agent_result.result
                        subtask.error = agent_result.error
                        
                        if agent_result.status == TaskStatus.COMPLETED:
                            results[subtask.agent_type.value] = agent_result
                            completed_tasks.add(subtask.id)
                            progress_made = True
                            
                            await self._notify_progress(ProgressUpdate(
                                session_id=session_id,
                                task_id=subtask.id,
                                agent_type=subtask.agent_type,
                                status=TaskStatus.COMPLETED,
                                progress=20 + (len(completed_tasks) * 60 // len(subtasks)),
                                message=f"Completed {subtask.agent_type.value} task",
                                result=agent_result.result
                            ))
                        else:
                            # Task failed
                            await self._notify_progress(ProgressUpdate(
                                session_id=session_id,
                                task_id=subtask.id,
                                agent_type=subtask.agent_type,
                                status=TaskStatus.FAILED,
                                progress=0,
                                message=f"Failed {subtask.agent_type.value} task: {agent_result.error}"
                            ))
                
                if not progress_made:
                    # Deadlock - some tasks can't be completed
                    execution.status = TaskStatus.FAILED
                    execution.updated_at = datetime.now().isoformat()
                    raise Exception("Task execution deadlock - some dependencies cannot be resolved")
            
            # Step 3: Aggregation
            await self._notify_progress(ProgressUpdate(
                session_id=session_id,
                task_id="aggregation",
                agent_type=AgentType.AGGREGATOR,
                status=TaskStatus.IN_PROGRESS,
                progress=80,
                message="Aggregating results..."
            ))
            
            aggregator_result = await self.agents[AgentType.AGGREGATOR].execute_with_retry(
                "aggregation", user_request, results
            )
            
            if aggregator_result.status == TaskStatus.FAILED:
                raise Exception(f"Aggregation failed: {aggregator_result.error}")
            
            # Send aggregation completion update
            # Convert result to serializable format
            serializable_result = aggregator_result.result
            if hasattr(aggregator_result.result, 'model_dump'):
                serializable_result = aggregator_result.result.model_dump()
            elif hasattr(aggregator_result.result, 'dict'):
                serializable_result = aggregator_result.result.dict()
            
            await self._notify_progress(ProgressUpdate(
                session_id=session_id,
                task_id="aggregation",
                agent_type=AgentType.AGGREGATOR,
                status=TaskStatus.COMPLETED,
                progress=90,
                message="Aggregation completed",
                result=serializable_result
            ))
            
            # Finalize execution
            execution.final_result = serializable_result
            execution.status = TaskStatus.COMPLETED
            execution.updated_at = datetime.now().isoformat()
            
            await self._notify_progress(ProgressUpdate(
                session_id=session_id,
                task_id="final",
                agent_type=AgentType.AGGREGATOR,
                status=TaskStatus.COMPLETED,
                progress=100,
                message="Task completed successfully!",
                result=serializable_result
            ))
            
            return session_id
            
        except Exception as e:
            execution.status = TaskStatus.FAILED
            execution.updated_at = datetime.now().isoformat()
            
            await self._notify_progress(ProgressUpdate(
                session_id=session_id,
                task_id="error",
                agent_type=AgentType.PLANNER,
                status=TaskStatus.FAILED,
                progress=0,
                message=f"Task failed: {str(e)}"
            ))
            
            raise e
    
    def get_execution(self, session_id: str) -> Optional[TaskExecution]:
        """Get execution status by session ID"""
        return self.active_executions.get(session_id)
    
    def get_all_executions(self) -> List[TaskExecution]:
        """Get all active executions"""
        return list(self.active_executions.values())
    
    async def start_conversation(self, session_id: str, initial_message: str) -> str:
        """Start a live conversation session"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        # Add user message
        user_message = ConversationMessage(
            role="user",
            content=initial_message,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        self.conversations[session_id].append(user_message)
        
        # Process the message
        response = await self._process_conversation_message(session_id, initial_message)
        
        # Add assistant response
        assistant_message = ConversationMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        self.conversations[session_id].append(assistant_message)
        
        return response
    
    async def continue_conversation(self, session_id: str, message: str) -> str:
        """Continue an existing conversation"""
        if session_id not in self.conversations:
            return "No active conversation found. Please start a new conversation."
        
        # Add user message
        user_message = ConversationMessage(
            role="user",
            content=message,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        self.conversations[session_id].append(user_message)
        
        # Process the message
        response = await self._process_conversation_message(session_id, message)
        
        # Add assistant response
        assistant_message = ConversationMessage(
            role="assistant",
            content=response,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
        self.conversations[session_id].append(assistant_message)
        
        return response
    
    async def _process_conversation_message(self, session_id: str, message: str) -> str:
        """Process a conversation message and determine appropriate response"""
        message_lower = message.lower()
        
        # Check if this is a request to modify an existing task
        if any(keyword in message_lower for keyword in ["modify", "change", "update", "refine", "adjust", "edit"]):
            return await self._handle_task_modification(session_id, message)
        
        # Check if this is a clarification response
        if session_id in self.pending_clarifications:
            return await self._handle_clarification_response(session_id, message)
        
        # Check if this is a new task request (more comprehensive detection)
        task_keywords = [
            "analyze", "analysis", "create", "generate", "build", "develop", "make", "produce",
            "calculate", "compute", "process", "evaluate", "assess", "review", "examine",
            "chart", "graph", "visualization", "dashboard", "report", "summary",
            "financial", "revenue", "profit", "sales", "data", "statistics", "trends"
        ]
        
        if any(keyword in message_lower for keyword in task_keywords):
            return await self._handle_new_task_request(session_id, message)
        
        # General conversation
        return await self._handle_general_conversation(session_id, message)
    
    async def _handle_task_modification(self, session_id: str, message: str) -> str:
        """Handle requests to modify existing tasks"""
        execution = self.get_execution(session_id)
        if not execution:
            return "No active task found to modify. Please start a new task first."
        
        # Create a new execution with the modified request
        modified_request = f"{execution.main_request}\n\nModification: {message}"
        new_session_id = str(uuid.uuid4())
        
        try:
            result = await self.execute_task(modified_request, new_session_id)
            # Use the result if it's a string (for testing), otherwise use the generated UUID
            if isinstance(result, str):
                return f"I've created a new task with your modifications. New session ID: {result}. The task is now being processed."
            else:
                return f"I've created a new task with your modifications. New session ID: {new_session_id}. The task is now being processed."
        except Exception as e:
            return f"Failed to create modified task: {str(e)}"
    
    async def _handle_clarification_response(self, session_id: str, message: str) -> str:
        """Handle clarification responses"""
        clarification = self.pending_clarifications.pop(session_id, None)
        if not clarification:
            return "No pending clarification found."
        
        # Restart the task with clarification
        execution = self.get_execution(session_id)
        if execution:
            updated_request = f"{execution.main_request}\n\nClarification: {message}"
            try:
                await self.execute_task(updated_request, session_id)
                return "Thank you for the clarification. I've restarted the task with your additional information."
            except Exception as e:
                return f"Failed to restart task with clarification: {str(e)}"
        
        return "Task not found. Please start a new task."
    
    async def _handle_new_task_request(self, session_id: str, message: str) -> str:
        """Handle new task requests"""
        try:
            await self.execute_task(message, session_id)
            return f"I've started processing your request. You can monitor the progress and provide additional input as needed."
        except Exception as e:
            return f"Failed to start task: {str(e)}"
    
    async def _handle_general_conversation(self, session_id: str, message: str) -> str:
        """Handle general conversation"""
        # Use OpenAI directly for conversational responses
        try:
            # Check if any agent is in mock mode
            mock_mode = any(hasattr(agent, 'mock_mode') and agent.mock_mode for agent in self.agents.values())
            
            if mock_mode:
                # Provide contextual responses based on the message
                message_lower = message.lower()
                if any(word in message_lower for word in ["hello", "hi", "help", "what can you do"]):
                    return "Hello! I'm here to help with your business needs. I can assist with financial analysis, data processing, and creating visualizations. What would you like to work on? (Currently running in demo mode)"
                elif any(word in message_lower for word in ["analyze", "create", "generate", "financial", "data", "chart"]):
                    return "I can help you with that! I can analyze financial data, create visualizations, and generate comprehensive reports. Would you like me to start a task for you?"
                else:
                    return "I'm here to help with your business needs! I can assist with financial analysis, data processing, and creating visualizations. What would you like to work on?"
            
            import openai
            from config import OPENAI_API_KEY
            
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            # Get conversation history for context
            history = self.conversations.get(session_id, [])
            messages = [
                {
                    "role": "system",
                    "content": """You are a helpful AI assistant for a multi-agent business task processing system. 
                    
                    Your capabilities include:
                    - Financial analysis and reporting
                    - Data analysis and statistical processing
                    - Chart and visualization creation
                    - Business task planning and execution
                    
                    When users ask questions or make requests:
                    1. If it's a clear business task (analyze, create, generate, etc.), suggest starting a task execution
                    2. If it's a question about capabilities, explain what the system can do
                    3. If it's general conversation, be helpful and friendly
                    4. Always be specific and actionable in your responses
                    
                    Respond naturally and helpfully. Avoid generic responses."""
                }
            ]
            
            # Add recent conversation history for context
            for msg in history[-6:]:  # Last 6 messages for context
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Add current message
            messages.append({
                "role": "user",
                "content": message
            })
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"I'm here to help with your business needs! I can assist with financial analysis, data processing, and creating visualizations. What would you like to work on? (Error: {str(e)})"
    
    def get_conversation_history(self, session_id: str) -> List[ConversationMessage]:
        """Get conversation history for a session"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str) -> bool:
        """Clear conversation history for a session"""
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False
    
    async def provide_clarification(self, session_id: str, clarification: str) -> str:
        """Provide clarification for a pending task"""
        if session_id not in self.pending_clarifications:
            return "No pending clarification found for this session."
        
        # Remove from pending and restart task
        self.pending_clarifications.pop(session_id)
        
        execution = self.get_execution(session_id)
        if execution:
            updated_request = f"{execution.main_request}\n\nClarification: {clarification}"
            try:
                await self.execute_task(updated_request, session_id)
                return "Task restarted with your clarification."
            except Exception as e:
                return f"Failed to restart task: {str(e)}"
        
        return "Task not found."
    
    def _has_circular_dependencies(self, subtasks: List[Subtask]) -> bool:
        """Check for circular dependencies in subtasks"""
        # Create a graph of dependencies
        graph = {}
        for subtask in subtasks:
            graph[subtask.id] = subtask.dependencies
        
        # Check for cycles using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                if has_cycle(node):
                    return True
        
        return False
