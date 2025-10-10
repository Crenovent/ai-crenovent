"""
Task 3.3.38: Voice Interface Service for Conversational Mode
- Speech-to-text integration for multi-modal input
- Logged like text with orchestrator integration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
import uuid
import logging
import base64

app = FastAPI(title="RBIA Voice Interface Service")
logger = logging.getLogger(__name__)

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    OGG = "ogg"
    WEBM = "webm"

class VoiceRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tenant_id: str
    user_id: str
    session_id: str
    
    # Audio data
    audio_data: str  # Base64 encoded audio
    audio_format: AudioFormat
    duration_seconds: float
    
    # Context
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VoiceResponse(BaseModel):
    request_id: str
    transcribed_text: str
    confidence_score: float
    
    # Processing details
    processing_time_ms: int
    audio_duration_seconds: float
    
    # Conversation integration
    conversation_message_id: str
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VoiceSession(BaseModel):
    session_id: str
    tenant_id: str
    user_id: str
    
    # Voice interactions
    voice_requests: List[VoiceRequest] = Field(default_factory=list)
    transcriptions: List[str] = Field(default_factory=list)
    
    # Session stats
    total_audio_duration: float = 0.0
    total_requests: int = 0
    avg_confidence: float = 0.0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)

# In-memory stores (replace with database in production)
voice_sessions_store: Dict[str, VoiceSession] = {}
voice_requests_store: Dict[str, VoiceRequest] = {}
voice_responses_store: Dict[str, VoiceResponse] = {}

def _simulate_speech_to_text(audio_data: str, audio_format: AudioFormat) -> tuple[str, float]:
    """Simulate speech-to-text conversion (replace with actual service in production)"""
    
    # Simulate different responses based on audio characteristics
    simulated_transcriptions = [
        ("I want to create a workflow for identifying at-risk customers", 0.95),
        ("Show me the pipeline summary for this quarter", 0.92),
        ("Can you help me analyze the conversion rates", 0.88),
        ("What are the top risk factors for deal closure", 0.91),
        ("Create a notification for deals over fifty thousand dollars", 0.89),
        ("I need to see the data quality metrics", 0.94),
        ("Help me build a workflow for lead scoring", 0.87),
        ("Show me the bias detection results", 0.93)
    ]
    
    # Simple hash-based selection for consistent simulation
    hash_val = hash(audio_data[:50]) % len(simulated_transcriptions)
    return simulated_transcriptions[hash_val]

@app.post("/voice/sessions", response_model=VoiceSession)
async def create_voice_session(tenant_id: str, user_id: str, session_id: str):
    """Create a new voice session for conversational mode"""
    
    voice_session = VoiceSession(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id
    )
    
    voice_sessions_store[session_id] = voice_session
    logger.info(f"Created voice session {session_id} for user {user_id}")
    return voice_session

@app.post("/voice/transcribe", response_model=VoiceResponse)
async def transcribe_voice(request: VoiceRequest):
    """Transcribe voice input to text for conversational mode"""
    
    # Store the request
    voice_requests_store[request.request_id] = request
    
    # Get or create voice session
    voice_session = voice_sessions_store.get(request.session_id)
    if not voice_session:
        voice_session = VoiceSession(
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            user_id=request.user_id
        )
        voice_sessions_store[request.session_id] = voice_session
    
    # Simulate speech-to-text processing
    start_time = datetime.utcnow()
    transcribed_text, confidence = _simulate_speech_to_text(
        request.audio_data, 
        request.audio_format
    )
    processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    
    # Create response
    response = VoiceResponse(
        request_id=request.request_id,
        transcribed_text=transcribed_text,
        confidence_score=confidence,
        processing_time_ms=int(processing_time),
        audio_duration_seconds=request.duration_seconds,
        conversation_message_id=str(uuid.uuid4())
    )
    
    # Store response
    voice_responses_store[response.request_id] = response
    
    # Update session
    voice_session.voice_requests.append(request)
    voice_session.transcriptions.append(transcribed_text)
    voice_session.total_audio_duration += request.duration_seconds
    voice_session.total_requests += 1
    voice_session.avg_confidence = (
        (voice_session.avg_confidence * (voice_session.total_requests - 1) + confidence) 
        / voice_session.total_requests
    )
    voice_session.last_activity = datetime.utcnow()
    
    # Log like text (as specified in task)
    logger.info(f"Voice transcription: '{transcribed_text}' (confidence: {confidence:.2f})")
    
    return response

@app.post("/voice/upload-file")
async def upload_voice_file(
    tenant_id: str,
    user_id: str,
    session_id: str,
    file: UploadFile = File(...)
):
    """Upload voice file for transcription"""
    
    # Read file content
    file_content = await file.read()
    
    # Determine audio format from file extension
    file_extension = file.filename.split('.')[-1].lower()
    audio_format = AudioFormat.WAV  # Default
    
    if file_extension in ['mp3']:
        audio_format = AudioFormat.MP3
    elif file_extension in ['ogg']:
        audio_format = AudioFormat.OGG
    elif file_extension in ['webm']:
        audio_format = AudioFormat.WEBM
    
    # Encode to base64
    audio_data = base64.b64encode(file_content).decode('utf-8')
    
    # Create voice request
    voice_request = VoiceRequest(
        tenant_id=tenant_id,
        user_id=user_id,
        session_id=session_id,
        audio_data=audio_data,
        audio_format=audio_format,
        duration_seconds=10.0  # Simulated duration
    )
    
    # Process transcription
    response = await transcribe_voice(voice_request)
    
    return {
        "file_name": file.filename,
        "file_size": len(file_content),
        "transcription": response
    }

@app.get("/voice/sessions/{session_id}", response_model=VoiceSession)
async def get_voice_session(session_id: str):
    """Get voice session details"""
    
    if session_id not in voice_sessions_store:
        raise HTTPException(status_code=404, detail="Voice session not found")
    
    return voice_sessions_store[session_id]

@app.get("/voice/sessions/{session_id}/history")
async def get_voice_history(session_id: str):
    """Get voice interaction history for a session"""
    
    if session_id not in voice_sessions_store:
        raise HTTPException(status_code=404, detail="Voice session not found")
    
    session = voice_sessions_store[session_id]
    
    # Get all responses for this session
    session_responses = [
        response for response in voice_responses_store.values()
        if any(req.session_id == session_id for req in voice_requests_store.values() 
               if req.request_id == response.request_id)
    ]
    
    return {
        "session_id": session_id,
        "total_interactions": session.total_requests,
        "total_audio_duration": session.total_audio_duration,
        "avg_confidence": session.avg_confidence,
        "transcription_history": session.transcriptions,
        "detailed_responses": session_responses
    }

@app.post("/voice/sessions/{session_id}/integrate-conversation")
async def integrate_with_conversation(
    session_id: str,
    conversation_service_url: str,
    transcribed_text: str
):
    """Integrate voice transcription with conversational orchestrator"""
    
    # This would integrate with the existing conversation agent
    # For now, just log the integration
    
    integration_data = {
        "voice_session_id": session_id,
        "transcribed_text": transcribed_text,
        "conversation_service": conversation_service_url,
        "integration_timestamp": datetime.utcnow().isoformat(),
        "logged_as_text": True  # As specified in task context
    }
    
    logger.info(f"Voice-to-conversation integration: {integration_data}")
    
    return {
        "status": "integrated",
        "voice_session_id": session_id,
        "conversation_message": transcribed_text,
        "logged_as_text": True
    }

@app.get("/voice/analytics/usage")
async def get_voice_usage_analytics(tenant_id: str):
    """Get voice interface usage analytics"""
    
    tenant_sessions = [
        session for session in voice_sessions_store.values()
        if session.tenant_id == tenant_id
    ]
    
    if not tenant_sessions:
        return {"message": "No voice sessions found for tenant"}
    
    total_duration = sum(session.total_audio_duration for session in tenant_sessions)
    total_requests = sum(session.total_requests for session in tenant_sessions)
    avg_confidence = sum(session.avg_confidence for session in tenant_sessions) / len(tenant_sessions)
    
    analytics = {
        "total_voice_sessions": len(tenant_sessions),
        "total_audio_duration_seconds": total_duration,
        "total_transcription_requests": total_requests,
        "average_confidence_score": avg_confidence,
        "avg_requests_per_session": total_requests / len(tenant_sessions) if tenant_sessions else 0,
        "avg_duration_per_session": total_duration / len(tenant_sessions) if tenant_sessions else 0
    }
    
    return analytics

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "RBIA Voice Interface Service", "task": "3.3.38"}
