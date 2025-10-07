"""
AI Microservices Orchestrator
Main service that orchestrates all AI microservices
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import httpx
import asyncio
from datetime import datetime
import uuid

app = FastAPI(
    title="AI Microservices Orchestrator",
    description="Main service that orchestrates all AI microservices",
    version="1.0.0"
)

# Service endpoints
SERVICE_ENDPOINTS = {
    "agent_registry": "http://localhost:8002",
    "routing_orchestrator": "http://localhost:8003",
    "kpi_exporter": "http://localhost:8004",
    "confidence_thresholds": "http://localhost:8005",
    "model_audit": "http://localhost:8006",
    "calendar_automation": "http://localhost:8007",
    "letsmeet_automation": "http://localhost:8008",
    "cruxx_automation": "http://localhost:8009",
    "run_trace_schema": "http://localhost:8010",
    "dlq_replay_tooling": "http://localhost:8011",
    "metrics_exporter": "http://localhost:8012",
    "event_bus_schema_registry": "http://localhost:8013"
}

class OrchestrationRequest(BaseModel):
    tenant_id: str
    user_id: str
    session_id: str
    service_name: str
    operation_type: str
    input_data: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)

class OrchestrationResponse(BaseModel):
    request_id: str
    recommended_mode: str
    confidence_score: float
    trust_score: float
    ui_metadata: Dict[str, Any]
    agent_id: Optional[str] = None
    explanation: str
    timestamp: datetime

class ServiceHealth(BaseModel):
    service_name: str
    status: str
    endpoint: str
    response_time_ms: Optional[float] = None
    error: Optional[str] = None

async def check_service_health(service_name: str, endpoint: str) -> ServiceHealth:
    """Check health of a microservice"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            start_time = datetime.utcnow()
            response = await client.get(f"{endpoint}/health")
            end_time = datetime.utcnow()
            
            response_time = (end_time - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                return ServiceHealth(
                    service_name=service_name,
                    status="healthy",
                    endpoint=endpoint,
                    response_time_ms=response_time
                )
            else:
                return ServiceHealth(
                    service_name=service_name,
                    status="unhealthy",
                    endpoint=endpoint,
                    error=f"HTTP {response.status_code}"
                )
    except Exception as e:
        return ServiceHealth(
            service_name=service_name,
            status="unhealthy",
            endpoint=endpoint,
            error=str(e)
        )

@app.get("/health", response_model=Dict[str, ServiceHealth])
async def health_check():
    """Check health of all microservices"""
    health_checks = []
    
    for service_name, endpoint in SERVICE_ENDPOINTS.items():
        health_checks.append(check_service_health(service_name, endpoint))
    
    results = await asyncio.gather(*health_checks)
    
    return {result.service_name: result for result in results}

@app.post("/orchestrate", response_model=OrchestrationResponse)
async def orchestrate_request(request: OrchestrationRequest):
    """Orchestrate a request across AI microservices"""
    
    request_id = str(uuid.uuid4())
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 1: Get confidence and trust scores (simulated)
            confidence_score = 0.8  # This would come from actual AI model evaluation
            trust_score = 0.7       # This would come from user history
            
            # Step 2: Evaluate confidence thresholds
            confidence_request = {
                "tenant_id": request.tenant_id,
                "service_name": request.service_name,
                "operation_type": request.operation_type,
                "confidence_score": confidence_score,
                "trust_score": trust_score,
                "context": request.context
            }
            
            confidence_response = await client.post(
                f"{SERVICE_ENDPOINTS['confidence_thresholds']}/confidence/evaluate",
                json=confidence_request
            )
            
            if confidence_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Confidence evaluation failed")
            
            confidence_data = confidence_response.json()
            
            # Step 3: Get routing recommendation
            routing_request = {
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "service_name": request.service_name,
                "operation_type": request.operation_type,
                "confidence_score": confidence_score,
                "trust_score": trust_score,
                "context": request.context
            }
            
            routing_response = await client.post(
                f"{SERVICE_ENDPOINTS['routing_orchestrator']}/routing/route",
                json=routing_request
            )
            
            if routing_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Routing evaluation failed")
            
            routing_data = routing_response.json()
            
            # Step 4: Get appropriate agent if needed
            agent_id = None
            if routing_data["recommended_mode"] == "agent":
                agent_query_request = {
                    "tenant_id": request.tenant_id,
                    "capabilities": [request.service_name],
                    "status": "active",
                    "min_trust_score": trust_score
                }
                
                agent_response = await client.post(
                    f"{SERVICE_ENDPOINTS['agent_registry']}/agents/query",
                    json=agent_query_request
                )
                
                if agent_response.status_code == 200:
                    agents = agent_response.json()
                    if agents:
                        agent_id = agents[0]["agent_id"]
            
            # Step 5: Record KPI metrics
            kpi_request = {
                "metric_id": str(uuid.uuid4()),
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "kpi_type": "confidence_score",
                "value": confidence_score,
                "timestamp": datetime.utcnow().isoformat(),
                "context": {
                    "service_name": request.service_name,
                    "operation_type": request.operation_type,
                    "recommended_mode": routing_data["recommended_mode"]
                }
            }
            
            await client.post(
                f"{SERVICE_ENDPOINTS['kpi_exporter']}/kpis/metrics",
                json=kpi_request
            )
            
            # Step 6: Record model call audit
            audit_request = {
                "call_id": str(uuid.uuid4()),
                "tenant_id": request.tenant_id,
                "user_id": request.user_id,
                "session_id": request.session_id,
                "agent_id": agent_id,
                "model_name": "orchestrator",
                "model_version": "1.0.0",
                "prompt": f"Orchestrate {request.service_name}.{request.operation_type}",
                "response": f"Recommended mode: {routing_data['recommended_mode']}",
                "tokens_used": 100,
                "cost_usd": 0.001,
                "latency_ms": 50,
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {
                    "confidence_score": confidence_score,
                    "trust_score": trust_score,
                    "recommended_mode": routing_data["recommended_mode"]
                }
            }
            
            await client.post(
                f"{SERVICE_ENDPOINTS['model_audit']}/audit/model-call",
                json=audit_request
            )
            
            return OrchestrationResponse(
                request_id=request_id,
                recommended_mode=routing_data["recommended_mode"],
                confidence_score=confidence_score,
                trust_score=trust_score,
                ui_metadata=routing_data["ui_metadata"],
                agent_id=agent_id,
                explanation=routing_data["reasoning"],
                timestamp=datetime.utcnow()
            )
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Service timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Orchestration failed: {str(e)}")

@app.get("/services")
async def list_services():
    """List all available microservices"""
    return {
        "services": list(SERVICE_ENDPOINTS.keys()),
        "endpoints": SERVICE_ENDPOINTS,
        "total_services": len(SERVICE_ENDPOINTS)
    }

@app.post("/services/{service_name}/restart")
async def restart_service(service_name: str):
    """Restart a microservice (placeholder)"""
    if service_name not in SERVICE_ENDPOINTS:
        raise HTTPException(status_code=404, detail="Service not found")
    
    # In a real implementation, this would trigger service restart
    return {
        "message": f"Service {service_name} restart initiated",
        "service": service_name,
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
