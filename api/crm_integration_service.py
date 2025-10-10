"""
Task 4.4.47: Add integration with CRM
- Sync success metrics to CRM
- Push adoption data to CRM
- Bidirectional data flow
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid
import logging

app = FastAPI(title="RBIA CRM Integration Service")
logger = logging.getLogger(__name__)

class CRMProvider(str, Enum):
    SALESFORCE = "salesforce"
    HUBSPOT = "hubspot"
    DYNAMICS_365 = "dynamics_365"
    ZOHO = "zoho"

class SyncDirection(str, Enum):
    TO_CRM = "to_crm"
    FROM_CRM = "from_crm"
    BIDIRECTIONAL = "bidirectional"

class SyncStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"

class CRMConnection(BaseModel):
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # Connection details
    tenant_id: str
    crm_provider: CRMProvider
    
    # Authentication
    api_key: str
    instance_url: str
    
    # Configuration
    sync_direction: SyncDirection
    sync_frequency_hours: int = 24
    
    # Status
    is_active: bool = True
    last_sync: Optional[datetime] = None
    next_sync: datetime
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class MetricsSyncPayload(BaseModel):
    tenant_id: str
    
    # Success metrics to sync
    trust_score: float
    accuracy_score: float
    roi_percentage: float
    adoption_percentage: float
    forecast_accuracy: float
    governance_compliance: float
    
    # Account info
    account_id: str
    account_name: str
    
    sync_timestamp: datetime = Field(default_factory=datetime.utcnow)

class CRMSyncRecord(BaseModel):
    sync_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    connection_id: str
    tenant_id: str
    crm_provider: CRMProvider
    
    # Sync details
    sync_direction: SyncDirection
    records_synced: int
    
    # Status
    status: SyncStatus
    error_message: Optional[str] = None
    
    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

class CRMFieldMapping(BaseModel):
    mapping_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    connection_id: str
    
    # Field mappings (RBIA field -> CRM field)
    field_mappings: Dict[str, str] = Field(default_factory=dict)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

# In-memory storage
crm_connections_store: Dict[str, CRMConnection] = {}
sync_records_store: Dict[str, CRMSyncRecord] = {}
field_mappings_store: Dict[str, CRMFieldMapping] = {}

class CRMIntegrationEngine:
    def __init__(self):
        # Default field mappings for each CRM provider
        self.default_mappings = {
            CRMProvider.SALESFORCE: {
                "trust_score": "RBIA_Trust_Score__c",
                "accuracy_score": "RBIA_Accuracy__c",
                "roi_percentage": "RBIA_ROI__c",
                "adoption_percentage": "RBIA_Adoption__c",
                "forecast_accuracy": "RBIA_Forecast_Accuracy__c",
                "governance_compliance": "RBIA_Governance__c"
            },
            CRMProvider.HUBSPOT: {
                "trust_score": "rbia_trust_score",
                "accuracy_score": "rbia_accuracy",
                "roi_percentage": "rbia_roi",
                "adoption_percentage": "rbia_adoption",
                "forecast_accuracy": "rbia_forecast_accuracy",
                "governance_compliance": "rbia_governance"
            },
            CRMProvider.DYNAMICS_365: {
                "trust_score": "new_rbia_trustscore",
                "accuracy_score": "new_rbia_accuracy",
                "roi_percentage": "new_rbia_roi",
                "adoption_percentage": "new_rbia_adoption",
                "forecast_accuracy": "new_rbia_forecastaccuracy",
                "governance_compliance": "new_rbia_governance"
            },
            CRMProvider.ZOHO: {
                "trust_score": "RBIA_Trust_Score",
                "accuracy_score": "RBIA_Accuracy",
                "roi_percentage": "RBIA_ROI",
                "adoption_percentage": "RBIA_Adoption",
                "forecast_accuracy": "RBIA_Forecast_Accuracy",
                "governance_compliance": "RBIA_Governance"
            }
        }
    
    def create_connection(self, connection: CRMConnection) -> CRMConnection:
        """Create CRM connection"""
        
        crm_connections_store[connection.connection_id] = connection
        
        # Create default field mapping
        default_mapping = self.default_mappings.get(connection.crm_provider, {})
        
        field_mapping = CRMFieldMapping(
            connection_id=connection.connection_id,
            field_mappings=default_mapping
        )
        field_mappings_store[field_mapping.mapping_id] = field_mapping
        
        logger.info(f"✅ Created CRM connection for {connection.crm_provider.value}")
        return connection
    
    def sync_metrics_to_crm(
        self,
        connection_id: str,
        payload: MetricsSyncPayload
    ) -> CRMSyncRecord:
        """Sync metrics to CRM"""
        
        connection = crm_connections_store.get(connection_id)
        if not connection:
            raise ValueError("Connection not found")
        
        if not connection.is_active:
            raise ValueError("Connection is not active")
        
        # Find field mapping
        mapping = None
        for m in field_mappings_store.values():
            if m.connection_id == connection_id:
                mapping = m
                break
        
        if not mapping:
            raise ValueError("Field mapping not found")
        
        # Simulate CRM API call
        try:
            # In real implementation, this would call the actual CRM API
            crm_data = self._prepare_crm_data(payload, mapping)
            
            # Log sync
            sync_record = CRMSyncRecord(
                connection_id=connection_id,
                tenant_id=payload.tenant_id,
                crm_provider=connection.crm_provider,
                sync_direction=SyncDirection.TO_CRM,
                records_synced=1,
                status=SyncStatus.SUCCESS,
                completed_at=datetime.utcnow()
            )
            
            sync_records_store[sync_record.sync_id] = sync_record
            
            # Update connection last sync
            connection.last_sync = datetime.utcnow()
            
            logger.info(f"✅ Synced metrics to {connection.crm_provider.value} for tenant {payload.tenant_id}")
            return sync_record
            
        except Exception as e:
            # Log failed sync
            sync_record = CRMSyncRecord(
                connection_id=connection_id,
                tenant_id=payload.tenant_id,
                crm_provider=connection.crm_provider,
                sync_direction=SyncDirection.TO_CRM,
                records_synced=0,
                status=SyncStatus.FAILED,
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
            
            sync_records_store[sync_record.sync_id] = sync_record
            
            logger.error(f"❌ Failed to sync to CRM: {str(e)}")
            raise
    
    def _prepare_crm_data(
        self,
        payload: MetricsSyncPayload,
        mapping: CRMFieldMapping
    ) -> Dict[str, Any]:
        """Prepare data for CRM API"""
        
        crm_data = {
            "account_id": payload.account_id,
            "account_name": payload.account_name
        }
        
        # Map RBIA fields to CRM fields
        metric_values = {
            "trust_score": payload.trust_score,
            "accuracy_score": payload.accuracy_score,
            "roi_percentage": payload.roi_percentage,
            "adoption_percentage": payload.adoption_percentage,
            "forecast_accuracy": payload.forecast_accuracy,
            "governance_compliance": payload.governance_compliance
        }
        
        for rbia_field, crm_field in mapping.field_mappings.items():
            if rbia_field in metric_values:
                crm_data[crm_field] = metric_values[rbia_field]
        
        crm_data["last_updated"] = datetime.utcnow().isoformat()
        
        return crm_data
    
    def get_sync_history(self, connection_id: str) -> List[CRMSyncRecord]:
        """Get sync history for a connection"""
        
        history = [
            record for record in sync_records_store.values()
            if record.connection_id == connection_id
        ]
        
        # Sort by started_at descending
        history.sort(key=lambda x: x.started_at, reverse=True)
        
        return history

# Global CRM integration engine
crm_engine = CRMIntegrationEngine()

@app.post("/crm-integration/connection/create", response_model=CRMConnection)
async def create_crm_connection(connection: CRMConnection):
    """Create CRM connection"""
    
    connection = crm_engine.create_connection(connection)
    return connection

@app.post("/crm-integration/sync/metrics", response_model=CRMSyncRecord)
async def sync_metrics_to_crm(
    connection_id: str,
    payload: MetricsSyncPayload
):
    """Sync metrics to CRM"""
    
    try:
        sync_record = crm_engine.sync_metrics_to_crm(connection_id, payload)
        return sync_record
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@app.get("/crm-integration/connection/{connection_id}")
async def get_crm_connection(connection_id: str):
    """Get CRM connection"""
    
    connection = crm_connections_store.get(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    return connection

@app.get("/crm-integration/connections/tenant/{tenant_id}")
async def get_tenant_connections(tenant_id: str):
    """Get all CRM connections for a tenant"""
    
    connections = [
        conn for conn in crm_connections_store.values()
        if conn.tenant_id == tenant_id
    ]
    
    return {
        "tenant_id": tenant_id,
        "total_connections": len(connections),
        "connections": connections
    }

@app.put("/crm-integration/connection/{connection_id}/toggle")
async def toggle_connection(connection_id: str, is_active: bool):
    """Enable or disable CRM connection"""
    
    connection = crm_connections_store.get(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    connection.is_active = is_active
    
    logger.info(f"✅ Connection {connection_id} {'enabled' if is_active else 'disabled'}")
    return connection

@app.get("/crm-integration/sync/history/{connection_id}")
async def get_sync_history(connection_id: str):
    """Get sync history for a connection"""
    
    connection = crm_connections_store.get(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    history = crm_engine.get_sync_history(connection_id)
    
    return {
        "connection_id": connection_id,
        "total_syncs": len(history),
        "history": history
    }

@app.get("/crm-integration/sync/stats/{connection_id}")
async def get_sync_stats(connection_id: str):
    """Get sync statistics for a connection"""
    
    connection = crm_connections_store.get(connection_id)
    if not connection:
        raise HTTPException(status_code=404, detail="Connection not found")
    
    history = crm_engine.get_sync_history(connection_id)
    
    successful_syncs = sum(1 for r in history if r.status == SyncStatus.SUCCESS)
    failed_syncs = sum(1 for r in history if r.status == SyncStatus.FAILED)
    total_records_synced = sum(r.records_synced for r in history)
    
    return {
        "connection_id": connection_id,
        "crm_provider": connection.crm_provider.value,
        "total_syncs": len(history),
        "successful_syncs": successful_syncs,
        "failed_syncs": failed_syncs,
        "success_rate": (successful_syncs / len(history) * 100) if history else 0,
        "total_records_synced": total_records_synced,
        "last_sync": connection.last_sync.isoformat() if connection.last_sync else None
    }

@app.put("/crm-integration/mapping/{connection_id}")
async def update_field_mapping(
    connection_id: str,
    field_mappings: Dict[str, str]
):
    """Update field mapping for a connection"""
    
    # Find existing mapping
    mapping = None
    for m in field_mappings_store.values():
        if m.connection_id == connection_id:
            mapping = m
            break
    
    if not mapping:
        # Create new mapping
        mapping = CRMFieldMapping(
            connection_id=connection_id,
            field_mappings=field_mappings
        )
        field_mappings_store[mapping.mapping_id] = mapping
    else:
        # Update existing mapping
        mapping.field_mappings = field_mappings
    
    logger.info(f"✅ Updated field mapping for connection {connection_id}")
    return mapping

@app.get("/crm-integration/summary")
async def get_crm_integration_summary():
    """Get CRM integration service summary"""
    
    total_connections = len(crm_connections_store)
    active_connections = sum(1 for c in crm_connections_store.values() if c.is_active)
    total_syncs = len(sync_records_store)
    successful_syncs = sum(1 for s in sync_records_store.values() if s.status == SyncStatus.SUCCESS)
    
    # Count by provider
    by_provider = {}
    for provider in CRMProvider:
        count = sum(1 for c in crm_connections_store.values() if c.crm_provider == provider)
        by_provider[provider.value] = count
    
    return {
        "total_connections": total_connections,
        "active_connections": active_connections,
        "total_syncs": total_syncs,
        "successful_syncs": successful_syncs,
        "success_rate": (successful_syncs / total_syncs * 100) if total_syncs > 0 else 0,
        "connections_by_provider": by_provider,
        "service_status": "active"
    }

