"""
Agent Registry Service
Manages AI agents registration, lifecycle, and configuration
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import uuid
from datetime import datetime

app = FastAPI(
    title="Agent Registry Service",
    description="Manages AI agents registration, lifecycle, and configuration",
    version="1.0.0"
)

class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    RETIRED = "retired"
    MAINTENANCE = "maintenance"

class AgentCapability(str, Enum):
    CALENDAR_MANAGEMENT = "calendar_management"
    MEETING_TRANSCRIPTION = "meeting_transcription"
    SUMMARY_GENERATION = "summary_generation"
    ACTION_EXTRACTION = "action_extraction"
    CRUXX_CREATION = "cruxx_creation"
    OPPORTUNITY_VALIDATION = "opportunity_validation"

class AgentRegistrationRequest(BaseModel):
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: List[AgentCapability] = Field(..., description="Agent capabilities")
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Confidence threshold")
    trust_score: float = Field(0.5, ge=0.0, le=1.0, description="Initial trust score")
    tenant_id: str = Field(..., description="Tenant ID")
    region: str = Field(..., description="Data residency region")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration")

class AgentResponse(BaseModel):
    agent_id: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    status: AgentStatus
    confidence_threshold: float
    trust_score: float
    tenant_id: str
    region: str
    configuration: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    last_activity: Optional[datetime] = None

class AgentUpdateRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[List[AgentCapability]] = None
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    trust_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    configuration: Optional[Dict[str, Any]] = None

class AgentQueryRequest(BaseModel):
    tenant_id: str
    capabilities: Optional[List[AgentCapability]] = None
    status: Optional[AgentStatus] = None
    region: Optional[str] = None
    min_trust_score: Optional[float] = Field(None, ge=0.0, le=1.0)

# In-memory storage (replace with database in production)
agents_db: Dict[str, AgentResponse] = {}

@app.post("/agents/register", response_model=AgentResponse)
async def register_agent(request: AgentRegistrationRequest):
    """Register a new AI agent"""
    agent_id = str(uuid.uuid4())
    now = datetime.utcnow()
    
    agent = AgentResponse(
        agent_id=agent_id,
        name=request.name,
        description=request.description,
        capabilities=request.capabilities,
        status=AgentStatus.ACTIVE,
        confidence_threshold=request.confidence_threshold,
        trust_score=request.trust_score,
        tenant_id=request.tenant_id,
        region=request.region,
        configuration=request.configuration,
        created_at=now,
        updated_at=now
    )
    
    agents_db[agent_id] = agent
    return agent

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get agent by ID"""
    if agent_id not in agents_db:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agents_db[agent_id]

@app.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(agent_id: str, request: AgentUpdateRequest):
    """Update agent configuration"""
    if agent_id not in agents_db:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_db[agent_id]
    update_data = request.dict(exclude_unset=True)
    
    for field, value in update_data.items():
        setattr(agent, field, value)
    
    agent.updated_at = datetime.utcnow()
    agents_db[agent_id] = agent
    
    return agent

@app.post("/agents/{agent_id}/retire")
async def retire_agent(agent_id: str):
    """Retire an agent"""
    if agent_id not in agents_db:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agents_db[agent_id]
    agent.status = AgentStatus.RETIRED
    agent.updated_at = datetime.utcnow()
    agents_db[agent_id] = agent
    
    return {"message": "Agent retired successfully"}

@app.post("/agents/query", response_model=List[AgentResponse])
async def query_agents(request: AgentQueryRequest):
    """Query agents based on criteria"""
    results = []
    
    for agent in agents_db.values():
        # Filter by tenant
        if agent.tenant_id != request.tenant_id:
            continue
        
        # Filter by capabilities
        if request.capabilities:
            if not any(cap in agent.capabilities for cap in request.capabilities):
                continue
        
        # Filter by status
        if request.status and agent.status != request.status:
            continue
        
        # Filter by region
        if request.region and agent.region != request.region:
            continue
        
        # Filter by minimum trust score
        if request.min_trust_score and agent.trust_score < request.min_trust_score:
            continue
        
        results.append(agent)
    
    return results

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents(tenant_id: str):
    """List all agents for a tenant"""
    return [agent for agent in agents_db.values() if agent.tenant_id == tenant_id]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "agent-registry",
        "timestamp": datetime.utcnow().isoformat(),
        "agents_count": len(agents_db)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
