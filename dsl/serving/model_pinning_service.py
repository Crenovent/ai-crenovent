"""
Task 6.3.74: Model-of-record pinning per tenant
Implement model-of-record pinning for predictable tenant behavior
"""

import logging
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelPin:
    """Model pin configuration for tenant"""
    tenant_id: str
    model_id: str
    model_version: str
    pinned_at: datetime
    pinned_by: str
    reason: str
    expires_at: Optional[datetime] = None

class ModelPinningService:
    """
    Model-of-record pinning service
    Task 6.3.74: Predictable behavior with CAB change control
    """
    
    def __init__(self):
        self.model_pins: Dict[str, ModelPin] = {}  # tenant_id -> ModelPin
    
    def pin_model(self, tenant_id: str, model_id: str, model_version: str, 
                  pinned_by: str, reason: str, expires_at: Optional[datetime] = None) -> bool:
        """Pin a model version for a tenant"""
        try:
            pin = ModelPin(
                tenant_id=tenant_id,
                model_id=model_id,
                model_version=model_version,
                pinned_at=datetime.utcnow(),
                pinned_by=pinned_by,
                reason=reason,
                expires_at=expires_at
            )
            
            self.model_pins[tenant_id] = pin
            logger.info(f"Pinned model {model_id}@{model_version} for tenant {tenant_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pin model: {e}")
            return False
    
    def get_pinned_model(self, tenant_id: str) -> Optional[ModelPin]:
        """Get pinned model for tenant"""
        pin = self.model_pins.get(tenant_id)
        
        if pin and pin.expires_at and datetime.utcnow() > pin.expires_at:
            # Pin expired, remove it
            del self.model_pins[tenant_id]
            return None
        
        return pin
    
    def unpin_model(self, tenant_id: str) -> bool:
        """Remove model pin for tenant"""
        if tenant_id in self.model_pins:
            del self.model_pins[tenant_id]
            logger.info(f"Unpinned model for tenant {tenant_id}")
            return True
        return False

# Global model pinning service
model_pinning_service = ModelPinningService()
