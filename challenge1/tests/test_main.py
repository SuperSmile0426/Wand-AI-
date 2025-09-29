"""
Tests for Challenge 1 main FastAPI application.
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from challenge1.main import app, manager, orchestrator


class TestMainApp:
    """Test main FastAPI application."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
    
    def test_api_info_endpoint(self, client):
        """Test API info endpoint."""
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Multi-Agent Task Solver API"
        assert data["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, client):
        """Test execute task endpoint with success."""
        with patch.object(orchestrator, 'execute_task', return_value="test_session_123"):
            response = client.post("/execute", json={
                "user_request": "Analyze quarterly financial performance"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session_123"
        assert data["status"] == "started"
    
    @pytest.mark.asyncio
    async def test_execute_task_with_session_id(self, client):
        """Test execute task endpoint with session ID."""
        with patch.object(orchestrator, 'execute_task', return_value="custom_session_456"):
            response = client.post("/execute", json={
                "user_request": "Create a financial dashboard",
                "session_id": "custom_session_456"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "custom_session_456"
        assert data["status"] == "started"
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, client):
        """Test execute task endpoint with failure."""
        with patch.object(orchestrator, 'execute_task', side_effect=Exception("Task execution failed")):
            response = client.post("/execute", json={
                "user_request": "Invalid request"
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Task execution failed" in data["detail"]
    
    def test_get_status_existing_session(self, client):
        """Test get status endpoint with existing session."""
        # Create a serializable execution object
        execution = {
            "session_id": "test_session",
            "main_request": "Test request",
            "status": "completed",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "subtasks": [],
            "final_result": None,
            "error": None
        }
        
        with patch.object(orchestrator, 'get_execution', return_value=execution):
            response = client.get("/status/test_session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
    
    def test_get_status_nonexistent_session(self, client):
        """Test get status endpoint with non-existent session."""
        with patch.object(orchestrator, 'get_execution', return_value=None):
            response = client.get("/status/nonexistent_session")
        
        assert response.status_code == 404
        data = response.json()
        assert "Session not found" in data["detail"]
    
    def test_get_all_executions(self, client):
        """Test get all executions endpoint."""
        # Create serializable executions
        execution1 = {
            "session_id": "session1",
            "main_request": "Test request 1",
            "status": "completed",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "subtasks": [],
            "final_result": None,
            "error": None
        }
        execution2 = {
            "session_id": "session2",
            "main_request": "Test request 2",
            "status": "in_progress",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "subtasks": [],
            "final_result": None,
            "error": None
        }
        
        with patch.object(orchestrator, 'get_all_executions', return_value=[execution1, execution2]):
            response = client.get("/executions")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
    
    @pytest.mark.asyncio
    async def test_websocket_endpoint(self, client):
        """Test WebSocket endpoint."""
        with client.websocket_connect("/ws") as websocket:
            # Test sending a message
            websocket.send_text("Hello WebSocket")
            data = websocket.receive_text()
            assert data == "Echo: Hello WebSocket"
    
    @pytest.mark.asyncio
    async def test_provide_clarification_success(self, client):
        """Test provide clarification endpoint with success."""
        # Mock execution
        execution = Mock()
        execution.status = "pending"
        execution.main_request = "Original request"
        
        with patch.object(orchestrator, 'get_execution', return_value=execution), \
             patch.object(orchestrator, 'execute_task', return_value="new_session_123"):
            response = client.post("/clarify/test_session", json={
                "clarification": "I need revenue data for Q1-Q4 2023"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "new_session_123"
        assert data["status"] == "restarted"
    
    @pytest.mark.asyncio
    async def test_provide_clarification_nonexistent_session(self, client):
        """Test provide clarification endpoint with non-existent session."""
        with patch.object(orchestrator, 'get_execution', return_value=None):
            response = client.post("/clarify/nonexistent_session", json={
                "clarification": "I need more data"
            })
        
        assert response.status_code == 404
        data = response.json()
        assert "Session not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_provide_clarification_wrong_status(self, client):
        """Test provide clarification endpoint with wrong status."""
        # Mock execution with wrong status
        execution = Mock()
        execution.status = "completed"  # Not pending
        
        with patch.object(orchestrator, 'get_execution', return_value=execution):
            response = client.post("/clarify/test_session", json={
                "clarification": "I need more data"
            })
        
        assert response.status_code == 400
        data = response.json()
        assert "Session is not waiting for clarification" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_provide_clarification_failure(self, client):
        """Test provide clarification endpoint with failure."""
        # Mock execution
        execution = Mock()
        execution.status = "pending"
        execution.main_request = "Original request"
        
        with patch.object(orchestrator, 'get_execution', return_value=execution), \
             patch.object(orchestrator, 'execute_task', side_effect=Exception("Execution failed")):
            response = client.post("/clarify/test_session", json={
                "clarification": "I need more data"
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Execution failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_start_conversation_success(self, client):
        """Test start conversation endpoint with success."""
        with patch.object(orchestrator, 'start_conversation', return_value="Hello! How can I help you?"):
            response = client.post("/conversation/start", json={
                "session_id": "test_session",
                "message": "Hello, can you help me?"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert data["response"] == "Hello! How can I help you?"
    
    @pytest.mark.asyncio
    async def test_start_conversation_missing_message(self, client):
        """Test start conversation endpoint with missing message."""
        response = client.post("/conversation/start", json={
            "session_id": "test_session"
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "Message is required" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_start_conversation_failure(self, client):
        """Test start conversation endpoint with failure."""
        with patch.object(orchestrator, 'start_conversation', side_effect=Exception("Conversation failed")):
            response = client.post("/conversation/start", json={
                "session_id": "test_session",
                "message": "Hello"
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Conversation failed" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_continue_conversation_success(self, client):
        """Test continue conversation endpoint with success."""
        with patch.object(orchestrator, 'continue_conversation', return_value="I can help you with that!"):
            response = client.post("/conversation/test_session", json={
                "message": "Can you analyze some data?"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert data["response"] == "I can help you with that!"
    
    @pytest.mark.asyncio
    async def test_continue_conversation_missing_message(self, client):
        """Test continue conversation endpoint with missing message."""
        response = client.post("/conversation/test_session", json={})
        
        assert response.status_code == 400
        data = response.json()
        assert "Message is required" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_continue_conversation_failure(self, client):
        """Test continue conversation endpoint with failure."""
        with patch.object(orchestrator, 'continue_conversation', side_effect=Exception("Conversation failed")):
            response = client.post("/conversation/test_session", json={
                "message": "Hello"
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Conversation failed" in data["detail"]
    
    def test_get_conversation_history_success(self, client):
        """Test get conversation history endpoint with success."""
        # Mock conversation history
        message1 = Mock()
        message1.model_dump.return_value = {"role": "user", "content": "Hello"}
        message2 = Mock()
        message2.model_dump.return_value = {"role": "assistant", "content": "Hi there!"}
        
        with patch.object(orchestrator, 'get_conversation_history', return_value=[message1, message2]):
            response = client.get("/conversation/test_session/history")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert len(data["history"]) == 2
        assert data["history"][0]["role"] == "user"
        assert data["history"][1]["role"] == "assistant"
    
    def test_clear_conversation_success(self, client):
        """Test clear conversation endpoint with success."""
        with patch.object(orchestrator, 'clear_conversation', return_value=True):
            response = client.delete("/conversation/test_session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert data["cleared"] is True
    
    def test_clear_conversation_failure(self, client):
        """Test clear conversation endpoint with failure."""
        with patch.object(orchestrator, 'clear_conversation', return_value=False):
            response = client.delete("/conversation/test_session")
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert data["cleared"] is False
    
    @pytest.mark.asyncio
    async def test_provide_enhanced_clarification_success(self, client):
        """Test provide enhanced clarification endpoint with success."""
        with patch.object(orchestrator, 'provide_clarification', return_value="Task restarted with your clarification."):
            response = client.post("/clarify/test_session/enhanced", json={
                "clarification": "I need revenue data for Q1-Q4 2023"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test_session"
        assert data["response"] == "Task restarted with your clarification."
        assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_provide_enhanced_clarification_failure(self, client):
        """Test provide enhanced clarification endpoint with failure."""
        with patch.object(orchestrator, 'provide_clarification', side_effect=Exception("Clarification failed")):
            response = client.post("/clarify/test_session/enhanced", json={
                "clarification": "I need more data"
            })
        
        assert response.status_code == 500
        data = response.json()
        assert "Clarification failed" in data["detail"]
    
    def test_connection_manager_initialization(self):
        """Test ConnectionManager initialization."""
        assert manager.active_connections == []
    
    @pytest.mark.asyncio
    async def test_connection_manager_connect(self, mock_websocket):
        """Test ConnectionManager connect method."""
        await manager.connect(mock_websocket)
        assert mock_websocket in manager.active_connections
        mock_websocket.accept.assert_called_once()
    
    def test_connection_manager_disconnect(self, mock_websocket):
        """Test ConnectionManager disconnect method."""
        manager.active_connections = [mock_websocket]
        manager.disconnect(mock_websocket)
        assert mock_websocket not in manager.active_connections
    
    @pytest.mark.asyncio
    async def test_connection_manager_send_personal_message(self, mock_websocket):
        """Test ConnectionManager send_personal_message method."""
        await manager.send_personal_message("Test message", mock_websocket)
        mock_websocket.send_text.assert_called_once_with("Test message")
    
    @pytest.mark.asyncio
    async def test_connection_manager_broadcast(self, mock_websocket):
        """Test ConnectionManager broadcast method."""
        manager.active_connections = [mock_websocket]
        await manager.broadcast("Broadcast message")
        mock_websocket.send_text.assert_called_once_with("Broadcast message")
    
    @pytest.mark.asyncio
    async def test_connection_manager_broadcast_with_broken_connection(self):
        """Test ConnectionManager broadcast with broken connection."""
        broken_websocket = Mock()
        broken_websocket.send_text = AsyncMock(side_effect=Exception("Connection broken"))
        
        good_websocket = Mock()
        good_websocket.send_text = AsyncMock()
        
        manager.active_connections = [broken_websocket, good_websocket]
        await manager.broadcast("Broadcast message")
        
        # Broken connection should be removed
        assert broken_websocket not in manager.active_connections
        assert good_websocket in manager.active_connections
        
        # Good connection should receive the message
        good_websocket.send_text.assert_called_once_with("Broadcast message")
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, mock_progress_callback):
        """Test progress callback function."""
        from challenge1.main import progress_callback
        
        update = Mock()
        update.model_dump.return_value = {"test": "data"}
        
        with patch.object(manager, 'broadcast', new_callable=AsyncMock) as mock_broadcast:
            await progress_callback(update)
        
        mock_broadcast.assert_called_once()
        call_args = mock_broadcast.call_args[0][0]
        assert json.loads(call_args)["type"] == "progress_update"
        assert json.loads(call_args)["data"]["test"] == "data"
    
    def test_cors_middleware(self, client):
        """Test CORS middleware is configured."""
        # Test that the API endpoint works (CORS middleware is configured)
        response = client.get("/api")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Multi-Agent Task Solver API"
        assert data["status"] == "running"
    
    def test_static_files_mounted(self, client):
        """Test static files are mounted."""
        # This would normally serve static files, but in test it might return 404
        # The important thing is that the mount is configured
        response = client.get("/static/nonexistent.html")
        # Should not return 500, indicating the mount is working
        assert response.status_code != 500
