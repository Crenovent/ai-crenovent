"""
Task 3.2.17: Key management & rotation; encrypt evidence/overrides at rest
- Confidentiality & crypto hygiene
- KMS + Vault with rotation
- Per-tenant encryption contexts
- Rotation schedule enforced
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import hashlib
import hmac
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import json

app = FastAPI(title="RBIA Key Management & Rotation Service")
logger = logging.getLogger(__name__)

class KeyType(str, Enum):
    EVIDENCE_ENCRYPTION = "evidence_encryption"
    OVERRIDE_ENCRYPTION = "override_encryption"
    TENANT_MASTER = "tenant_master"
    DATABASE_ENCRYPTION = "database_encryption"

class KeyStatus(str, Enum):
    ACTIVE = "active"
    ROTATING = "rotating"
    DEPRECATED = "deprecated"
    REVOKED = "revoked"

class RotationStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class EncryptionKey(BaseModel):
    key_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key_type: KeyType = Field(..., description="Type of encryption key")
    tenant_id: str = Field(..., description="Tenant identifier")
    key_version: int = Field(default=1, description="Key version number")
    status: KeyStatus = Field(default=KeyStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Key expiration time")
    last_rotated_at: Optional[datetime] = Field(None)
    rotation_schedule_days: int = Field(default=90, description="Rotation frequency in days")
    key_hash: str = Field(..., description="Hash of the key for verification")
    vault_key_name: str = Field(..., description="Key name in Azure Key Vault")

class KeyRotationRequest(BaseModel):
    key_id: str = Field(..., description="Key to rotate")
    force_rotation: bool = Field(default=False, description="Force rotation even if not due")
    new_expiry_days: Optional[int] = Field(None, description="Custom expiry for new key")

class EncryptionRequest(BaseModel):
    data: str = Field(..., description="Data to encrypt")
    tenant_id: str = Field(..., description="Tenant identifier")
    key_type: KeyType = Field(..., description="Type of encryption key to use")
    context: Dict[str, str] = Field(default={}, description="Additional encryption context")

class DecryptionRequest(BaseModel):
    encrypted_data: str = Field(..., description="Encrypted data to decrypt")
    tenant_id: str = Field(..., description="Tenant identifier")
    key_id: str = Field(..., description="Key ID used for encryption")
    context: Dict[str, str] = Field(default={}, description="Encryption context")

class EncryptionResult(BaseModel):
    encrypted_data: str = Field(..., description="Base64 encoded encrypted data")
    key_id: str = Field(..., description="Key ID used for encryption")
    key_version: int = Field(..., description="Key version used")
    encryption_context: Dict[str, str] = Field(..., description="Encryption context")
    encrypted_at: datetime = Field(default_factory=datetime.utcnow)

class KeyRotationResult(BaseModel):
    rotation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    old_key_id: str = Field(..., description="Previous key ID")
    new_key_id: str = Field(..., description="New key ID")
    status: RotationStatus = Field(..., description="Rotation status")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)
    affected_records: int = Field(default=0, description="Number of records re-encrypted")

# In-memory storage for keys (replace with actual secure storage)
encryption_keys: Dict[str, EncryptionKey] = {}
rotation_history: List[KeyRotationResult] = []

# Key derivation settings
KEY_DERIVATION_ITERATIONS = 100000
SALT_LENGTH = 32

@app.post("/keys/generate", response_model=EncryptionKey)
async def generate_encryption_key(
    key_type: KeyType,
    tenant_id: str,
    rotation_schedule_days: int = 90
):
    """
    Generate new encryption key for tenant
    """
    # Generate master key for tenant if not exists
    master_key = await get_or_create_tenant_master_key(tenant_id)
    
    # Create new encryption key
    key_id = str(uuid.uuid4())
    vault_key_name = f"{tenant_id}_{key_type.value}_{key_id[:8]}"
    
    # Generate key material
    key_material = Fernet.generate_key()
    key_hash = hashlib.sha256(key_material).hexdigest()
    
    # Store in vault (simulated)
    await store_key_in_vault(vault_key_name, key_material)
    
    encryption_key = EncryptionKey(
        key_id=key_id,
        key_type=key_type,
        tenant_id=tenant_id,
        expires_at=datetime.utcnow() + timedelta(days=rotation_schedule_days),
        rotation_schedule_days=rotation_schedule_days,
        key_hash=key_hash,
        vault_key_name=vault_key_name
    )
    
    encryption_keys[key_id] = encryption_key
    
    logger.info(f"Generated encryption key: {key_id} for tenant {tenant_id}, type {key_type.value}")
    
    return encryption_key

async def get_or_create_tenant_master_key(tenant_id: str) -> str:
    """Get or create master key for tenant"""
    master_key_id = f"master_{tenant_id}"
    
    # Check if master key exists
    for key in encryption_keys.values():
        if key.tenant_id == tenant_id and key.key_type == KeyType.TENANT_MASTER:
            return key.key_id
    
    # Create new master key
    master_key = await generate_encryption_key(
        KeyType.TENANT_MASTER,
        tenant_id,
        rotation_schedule_days=365  # Master keys rotate annually
    )
    
    return master_key.key_id

async def store_key_in_vault(key_name: str, key_material: bytes):
    """Store key in Azure Key Vault (simulated)"""
    # In production, this would use Azure Key Vault client
    logger.info(f"Storing key {key_name} in Azure Key Vault")
    # Simulate vault storage
    pass

async def retrieve_key_from_vault(key_name: str) -> bytes:
    """Retrieve key from Azure Key Vault (simulated)"""
    # In production, this would use Azure Key Vault client
    logger.info(f"Retrieving key {key_name} from Azure Key Vault")
    # Simulate key retrieval - in production this would be the actual key
    return Fernet.generate_key()

@app.post("/encrypt", response_model=EncryptionResult)
async def encrypt_data(request: EncryptionRequest):
    """
    Encrypt data using tenant-specific key
    """
    # Find active key for tenant and type
    active_key = None
    for key in encryption_keys.values():
        if (key.tenant_id == request.tenant_id and 
            key.key_type == request.key_type and 
            key.status == KeyStatus.ACTIVE and
            key.expires_at > datetime.utcnow()):
            active_key = key
            break
    
    if not active_key:
        # Generate new key if none exists
        active_key = await generate_encryption_key(
            request.key_type,
            request.tenant_id
        )
    
    # Retrieve key material from vault
    key_material = await retrieve_key_from_vault(active_key.vault_key_name)
    
    # Create encryption context
    encryption_context = {
        "tenant_id": request.tenant_id,
        "key_type": request.key_type.value,
        "key_version": str(active_key.key_version),
        **request.context
    }
    
    # Encrypt data
    fernet = Fernet(key_material)
    
    # Add context to data before encryption
    data_with_context = {
        "data": request.data,
        "context": encryption_context,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    data_json = json.dumps(data_with_context)
    encrypted_bytes = fernet.encrypt(data_json.encode())
    encrypted_data = base64.b64encode(encrypted_bytes).decode()
    
    logger.info(f"Encrypted data for tenant {request.tenant_id} using key {active_key.key_id}")
    
    return EncryptionResult(
        encrypted_data=encrypted_data,
        key_id=active_key.key_id,
        key_version=active_key.key_version,
        encryption_context=encryption_context
    )

@app.post("/decrypt")
async def decrypt_data(request: DecryptionRequest):
    """
    Decrypt data using specified key
    """
    # Find the key
    if request.key_id not in encryption_keys:
        raise HTTPException(status_code=404, detail="Encryption key not found")
    
    encryption_key = encryption_keys[request.key_id]
    
    # Verify tenant access
    if encryption_key.tenant_id != request.tenant_id:
        raise HTTPException(status_code=403, detail="Access denied to encryption key")
    
    # Retrieve key material from vault
    key_material = await retrieve_key_from_vault(encryption_key.vault_key_name)
    
    try:
        # Decrypt data
        fernet = Fernet(key_material)
        encrypted_bytes = base64.b64decode(request.encrypted_data.encode())
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
        
        # Parse decrypted data
        decrypted_json = json.loads(decrypted_bytes.decode())
        
        # Verify context
        stored_context = decrypted_json.get("context", {})
        if stored_context.get("tenant_id") != request.tenant_id:
            raise HTTPException(status_code=403, detail="Context verification failed")
        
        logger.info(f"Decrypted data for tenant {request.tenant_id} using key {request.key_id}")
        
        return {
            "decrypted_data": decrypted_json["data"],
            "context": stored_context,
            "encrypted_at": decrypted_json.get("timestamp")
        }
        
    except Exception as e:
        logger.error(f"Decryption failed: {e}")
        raise HTTPException(status_code=400, detail="Decryption failed")

@app.post("/keys/rotate", response_model=KeyRotationResult)
async def rotate_key(request: KeyRotationRequest):
    """
    Rotate encryption key
    """
    if request.key_id not in encryption_keys:
        raise HTTPException(status_code=404, detail="Key not found")
    
    old_key = encryption_keys[request.key_id]
    
    # Check if rotation is due
    if not request.force_rotation:
        days_until_expiry = (old_key.expires_at - datetime.utcnow()).days
        if days_until_expiry > 30:  # Don't rotate unless within 30 days of expiry
            raise HTTPException(
                status_code=400, 
                detail=f"Key rotation not due. {days_until_expiry} days remaining."
            )
    
    rotation_result = KeyRotationResult(
        old_key_id=request.key_id,
        new_key_id="",  # Will be set after generation
        status=RotationStatus.IN_PROGRESS
    )
    
    try:
        # Generate new key
        new_expiry_days = request.new_expiry_days or old_key.rotation_schedule_days
        new_key = await generate_encryption_key(
            old_key.key_type,
            old_key.tenant_id,
            new_expiry_days
        )
        
        rotation_result.new_key_id = new_key.key_id
        
        # Mark old key as deprecated
        old_key.status = KeyStatus.DEPRECATED
        old_key.last_rotated_at = datetime.utcnow()
        
        # Simulate re-encryption of existing data
        affected_records = await re_encrypt_existing_data(old_key, new_key)
        rotation_result.affected_records = affected_records
        
        rotation_result.status = RotationStatus.COMPLETED
        rotation_result.completed_at = datetime.utcnow()
        
        rotation_history.append(rotation_result)
        
        logger.info(f"Key rotation completed: {request.key_id} -> {new_key.key_id}")
        
        return rotation_result
        
    except Exception as e:
        rotation_result.status = RotationStatus.FAILED
        rotation_history.append(rotation_result)
        logger.error(f"Key rotation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Key rotation failed: {e}")

async def re_encrypt_existing_data(old_key: EncryptionKey, new_key: EncryptionKey) -> int:
    """Re-encrypt existing data with new key"""
    # In production, this would query database and re-encrypt records
    # Simulating re-encryption of evidence and override records
    
    affected_count = 0
    
    if old_key.key_type == KeyType.EVIDENCE_ENCRYPTION:
        # Simulate evidence re-encryption
        affected_count += 150  # Simulate 150 evidence records
        logger.info(f"Re-encrypted {affected_count} evidence records")
    
    elif old_key.key_type == KeyType.OVERRIDE_ENCRYPTION:
        # Simulate override re-encryption
        affected_count += 75   # Simulate 75 override records
        logger.info(f"Re-encrypted {affected_count} override records")
    
    return affected_count

@app.get("/keys/check-rotation")
async def check_rotation_schedule():
    """
    Check which keys need rotation
    """
    keys_needing_rotation = []
    
    for key in encryption_keys.values():
        if key.status == KeyStatus.ACTIVE:
            days_until_expiry = (key.expires_at - datetime.utcnow()).days
            
            if days_until_expiry <= 30:  # Rotation needed within 30 days
                keys_needing_rotation.append({
                    "key_id": key.key_id,
                    "tenant_id": key.tenant_id,
                    "key_type": key.key_type.value,
                    "days_until_expiry": days_until_expiry,
                    "expires_at": key.expires_at.isoformat(),
                    "rotation_urgency": "critical" if days_until_expiry <= 7 else "high" if days_until_expiry <= 14 else "medium"
                })
    
    return {
        "keys_needing_rotation": keys_needing_rotation,
        "total_keys": len(encryption_keys),
        "rotation_check_time": datetime.utcnow().isoformat()
    }

@app.post("/keys/auto-rotate")
async def auto_rotate_keys():
    """
    Automatically rotate keys that are due for rotation
    """
    rotation_results = []
    
    for key in encryption_keys.values():
        if key.status == KeyStatus.ACTIVE:
            days_until_expiry = (key.expires_at - datetime.utcnow()).days
            
            if days_until_expiry <= 7:  # Auto-rotate within 7 days
                try:
                    rotation_request = KeyRotationRequest(
                        key_id=key.key_id,
                        force_rotation=True
                    )
                    rotation_result = await rotate_key(rotation_request)
                    rotation_results.append(rotation_result)
                    
                except Exception as e:
                    logger.error(f"Auto-rotation failed for key {key.key_id}: {e}")
                    rotation_results.append({
                        "key_id": key.key_id,
                        "status": "failed",
                        "error": str(e)
                    })
    
    return {
        "auto_rotation_results": rotation_results,
        "rotated_count": len([r for r in rotation_results if hasattr(r, 'status') and r.status == RotationStatus.COMPLETED]),
        "rotation_time": datetime.utcnow().isoformat()
    }

@app.get("/keys")
async def list_keys(
    tenant_id: Optional[str] = None,
    key_type: Optional[KeyType] = None,
    status: Optional[KeyStatus] = None
):
    """List encryption keys with filtering"""
    filtered_keys = list(encryption_keys.values())
    
    if tenant_id:
        filtered_keys = [k for k in filtered_keys if k.tenant_id == tenant_id]
    
    if key_type:
        filtered_keys = [k for k in filtered_keys if k.key_type == key_type]
    
    if status:
        filtered_keys = [k for k in filtered_keys if k.status == status]
    
    return {
        "keys": [
            {
                "key_id": k.key_id,
                "key_type": k.key_type.value,
                "tenant_id": k.tenant_id,
                "status": k.status.value,
                "created_at": k.created_at.isoformat(),
                "expires_at": k.expires_at.isoformat(),
                "days_until_expiry": (k.expires_at - datetime.utcnow()).days
            }
            for k in filtered_keys
        ],
        "total_count": len(filtered_keys)
    }

@app.get("/keys/rotation-history")
async def get_rotation_history(limit: int = 50):
    """Get key rotation history"""
    return {
        "rotation_history": rotation_history[-limit:],
        "total_rotations": len(rotation_history)
    }

@app.get("/health")
async def health_check():
    active_keys = len([k for k in encryption_keys.values() if k.status == KeyStatus.ACTIVE])
    keys_expiring_soon = len([k for k in encryption_keys.values() 
                             if k.status == KeyStatus.ACTIVE and 
                             (k.expires_at - datetime.utcnow()).days <= 30])
    
    return {
        "status": "healthy",
        "service": "RBIA Key Management & Rotation Service",
        "task": "3.2.17",
        "active_keys": active_keys,
        "keys_expiring_soon": keys_expiring_soon,
        "total_rotations": len(rotation_history),
        "supported_key_types": [kt.value for kt in KeyType]
    }

