from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import json
import asyncio
import uuid
from typing import List, Dict, Any
from models import TaskRequest, TaskExecution, ProgressUpdate, ConversationMessage
from orchestrator import TaskOrchestrator
from config import PORT, HOST

app = FastAPI(title="Multi-Agent Task Solver", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize orchestrator
orchestrator = TaskOrchestrator()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

# Progress callback for WebSocket updates
async def progress_callback(update: ProgressUpdate):
    message = {
        "type": "progress_update",
        "data": update.model_dump()
    }
    await manager.broadcast(json.dumps(message))

# Add progress callback to orchestrator
orchestrator.add_progress_callback(progress_callback)

@app.get("/")
async def root():
    import os
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))

@app.get("/api")
async def api_info():
    return {"message": "Multi-Agent Task Solver API", "status": "running"}

@app.post("/execute")
async def execute_task(request: TaskRequest):
    """Execute a new task"""
    try:
        session_id = await orchestrator.execute_task(
            request.user_request, 
            request.session_id
        )
        return {"session_id": session_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """Get execution status"""
    execution = orchestrator.get_execution(session_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Session not found")
    return execution

@app.get("/executions")
async def get_all_executions():
    """Get all executions"""
    return orchestrator.get_all_executions()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for connection testing
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/clarify/{session_id}")
async def provide_clarification(session_id: str, clarification: Dict[str, str]):
    """Provide clarification for a task that needs it"""
    execution = orchestrator.get_execution(session_id)
    if not execution:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if execution.status != "pending":
        raise HTTPException(status_code=400, detail="Session is not waiting for clarification")
    
    # Restart execution with clarification
    try:
        updated_request = f"{execution.main_request}\n\nClarification: {clarification.get('clarification', '')}"
        new_session_id = await orchestrator.execute_task(updated_request)
        return {"session_id": new_session_id, "status": "restarted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Conversation endpoints
@app.post("/conversation/start")
async def start_conversation(request: Dict[str, str]):
    """Start a new conversation session"""
    session_id = request.get("session_id", str(uuid.uuid4()))
    message = request.get("message", "")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        response = await orchestrator.start_conversation(session_id, message)
        return {"session_id": session_id, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversation/{session_id}")
async def continue_conversation(session_id: str, request: Dict[str, str]):
    """Continue an existing conversation"""
    message = request.get("message", "")
    
    if not message:
        raise HTTPException(status_code=400, detail="Message is required")
    
    try:
        response = await orchestrator.continue_conversation(session_id, message)
        return {"session_id": session_id, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """Get conversation history for a session"""
    history = orchestrator.get_conversation_history(session_id)
    return {"session_id": session_id, "history": [msg.model_dump() for msg in history]}

@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history for a session"""
    success = orchestrator.clear_conversation(session_id)
    return {"session_id": session_id, "cleared": success}

# Enhanced clarification endpoint
@app.post("/clarify/{session_id}/enhanced")
async def provide_enhanced_clarification(session_id: str, clarification: Dict[str, str]):
    """Provide enhanced clarification for a task that needs it"""
    try:
        response = await orchestrator.provide_clarification(session_id, clarification.get('clarification', ''))
        return {"session_id": session_id, "response": response, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
