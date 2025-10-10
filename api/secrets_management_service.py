"""
Task 3.2.18: Secrets management for model endpoints & connectors
- Least-privilege, no hard-coded secrets
- Vault + RBAC
- Short-lived tokens
"""

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from enum import Enum
import uuid
import logging
import hashlib
import secrets
import jwt
import base64
import json

app = FastAPI(title="RBIA Secrets Management Service")
logger = logging.getLogger(__name__)

class SecretType(str, Enum):
    MODEL_ENDPOINT = "model_endpoint"
    DATABASE_CONNECTOR = "database_connector"
    API_CONNECTOR = "api_connector"
    SERVICE_ACCOUNT = "service_account"
    WEBHOOK_SECRET = "webhook_secret"

class TokenType(str, Enum):
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    API_KEY = "api_key"
    SERVICE_TOKEN = "service_token"

class SecretStatus(str, Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ROTATING = "rotating"

class AccessLevel(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SERVICE = "service"

class ManagedSecret(BaseModel):
    secret_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    secret_type: SecretType = Field(..., description="Type of secret")
    name: str = Field(..., description="Secret name/identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    service_name: str = Field(..., description="Service that uses this secret")
    endpoint_url: Optional[str] = Field(None, description="Target endpoint URL")
    access_level: AccessLevel = Field(..., description="Access level for this secret")
    status: SecretStatus = Field(default=SecretStatus.ACTIVE)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Secret expiration time")
    last_rotated_at: Optional[datetime] = Field(None)
    rotation_interval_days: int = Field(default=90, description="Rotation frequency")
    vault_path: str = Field(..., description="Path in secret vault")
    allowed_services: List[str] = Field(default=[], description="Services allowed to access this secret")
    usage_count: int = Field(default=0, description="Number of times secret has been accessed")
    last_accessed_at: Optional[datetime] = Field(None)

class ShortLivedToken(BaseModel):
    token_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    token_type: TokenType = Field(..., description="Type of token")
    tenant_id: str = Field(..., description="Tenant identifier")
    service_name: str = Field(..., description="Service requesting token")
    target_resource: str = Field(..., description="Target resource/endpoint")
    access_level: AccessLevel = Field(..., description="Access level granted")
    issued_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime = Field(..., description="Token expiration time")
    is_revoked: bool = Field(default=False)
    usage_count: int = Field(default=0)
    max_usage_count: Optional[int] = Field(None, description="Maximum allowed uses")

class SecretRequest(BaseModel):
    secret_type: SecretType = Field(..., description="Type of secret to create")
    name: str = Field(..., description="Secret name")
    tenant_id: str = Field(..., description="Tenant identifier")
    service_name: str = Field(..., description="Service that will use this secret")
    endpoint_url: Optional[str] = Field(None, description="Target endpoint URL")
    access_level: AccessLevel = Field(..., description="Required access level")
    secret_value: str = Field(..., description="Secret value to store")
    rotation_interval_days: int = Field(default=90, description="Rotation frequency")
    allowed_services: List[str] = Field(default=[], description="Services allowed to access")

class TokenRequest(BaseModel):
    token_type: TokenType = Field(..., description="Type of token to generate")
    tenant_id: str = Field(..., description="Tenant identifier")
    service_name: str = Field(..., description="Service requesting token")
    target_resource: str = Field(..., description="Target resource/endpoint")
    access_level: AccessLevel = Field(..., description="Required access level")
    duration_minutes: int = Field(default=60, description="Token lifetime in minutes")
    max_usage_count: Optional[int] = Field(None, description="Maximum allowed uses")

class SecretAccessRequest(BaseModel):
    secret_id: str = Field(..., description="Secret to access")
    requesting_service: str = Field(..., description="Service requesting access")
    tenant_id: str = Field(..., description="Tenant identifier")
    access_reason: str = Field(..., description="Reason for accessing secret")

# In-memory storage (replace with actual secure vault)
managed_secrets: Dict[str, ManagedSecret] = {}
short_lived_tokens: Dict[str, ShortLivedToken] = {}
access_logs: List[Dict[str, Any]] = []

# RBAC permissions
SERVICE_PERMISSIONS = {
    "rbia_model_service": [SecretType.MODEL_ENDPOINT, SecretType.API_CONNECTOR],
    "rbia_data_service": [SecretType.DATABASE_CONNECTOR, SecretType.API_CONNECTOR],
    "rbia_governance_service": [SecretType.SERVICE_ACCOUNT, SecretType.WEBHOOK_SECRET],
    "rbia_orchestrator": [SecretType.MODEL_ENDPOINT, SecretType.DATABASE_CONNECTOR, SecretType.SERVICE_ACCOUNT]
}

JWT_SECRET = "rbia_jwt_secret_key"  # In production, use proper secret management

@app.post("/secrets", response_model=ManagedSecret)
async def create_secret(request: SecretRequest):
    """
    Create and store a new managed secret
    """
    # Validate service permissions
    if not validate_service_permissions(request.service_name, request.secret_type):
        raise HTTPException(
            status_code=403,
            detail=f"Service {request.service_name} not authorized for {request.secret_type.value} secrets"
        )
    
    # Generate vault path
    vault_path = f"secrets/{request.tenant_id}/{request.secret_type.value}/{request.name}"
    
    # Store secret in vault (simulated)
    await store_secret_in_vault(vault_path, request.secret_value)
    
    # Create managed secret record
    managed_secret = ManagedSecret(
        secret_type=request.secret_type,
        name=request.name,
        tenant_id=request.tenant_id,
        service_name=request.service_name,
        endpoint_url=request.endpoint_url,
        access_level=request.access_level,
        expires_at=datetime.utcnow() + timedelta(days=request.rotation_interval_days),
        rotation_interval_days=request.rotation_interval_days,
        vault_path=vault_path,
        allowed_services=request.allowed_services
    )
    
    managed_secrets[managed_secret.secret_id] = managed_secret
    
    # Log secret creation
    await log_secret_access(
        "secret_created",
        managed_secret.secret_id,
        request.service_name,
        request.tenant_id,
        f"Created {request.secret_type.value} secret: {request.name}"
    )
    
    logger.info(f"Created managed secret: {managed_secret.secret_id} for {request.service_name}")
    
    return managed_secret

def validate_service_permissions(service_name: str, secret_type: SecretType) -> bool:
    """Validate service has permission to access secret type"""
    allowed_types = SERVICE_PERMISSIONS.get(service_name, [])
    return secret_type in allowed_types

async def store_secret_in_vault(vault_path: str, secret_value: str):
    """Store secret in vault (simulated)"""
    # In production, this would use Azure Key Vault or HashiCorp Vault
    logger.info(f"Storing secret at vault path: {vault_path}")
    # Simulate secure storage
    pass

async def retrieve_secret_from_vault(vault_path: str) -> str:
    """Retrieve secret from vault (simulated)"""
    # In production, this would use actual vault client
    logger.info(f"Retrieving secret from vault path: {vault_path}")
    # Return simulated secret
    return "simulated_secret_value"

@app.post("/secrets/access")
async def access_secret(request: SecretAccessRequest):
    """
    Access a managed secret with RBAC checks
    """
    if request.secret_id not in managed_secrets:
        raise HTTPException(status_code=404, detail="Secret not found")
    
    secret = managed_secrets[request.secret_id]
    
    # Validate tenant access
    if secret.tenant_id != request.tenant_id:
        raise HTTPException(status_code=403, detail="Tenant access denied")
    
    # Validate service access
    if secret.allowed_services and request.requesting_service not in secret.allowed_services:
        raise HTTPException(status_code=403, detail="Service access denied")
    
    # Check if secret is expired
    if secret.expires_at and secret.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Secret has expired")
    
    # Check if secret is active
    if secret.status != SecretStatus.ACTIVE:
        raise HTTPException(status_code=403, detail=f"Secret is {secret.status.value}")
    
    # Retrieve secret value from vault
    secret_value = await retrieve_secret_from_vault(secret.vault_path)
    
    # Update access tracking
    secret.usage_count += 1
    secret.last_accessed_at = datetime.utcnow()
    
    # Log access
    await log_secret_access(
        "secret_accessed",
        request.secret_id,
        request.requesting_service,
        request.tenant_id,
        request.access_reason
    )
    
    logger.info(f"Secret accessed: {request.secret_id} by {request.requesting_service}")
    
    return {
        "secret_value": secret_value,
        "secret_type": secret.secret_type.value,
        "endpoint_url": secret.endpoint_url,
        "access_level": secret.access_level.value,
        "expires_at": secret.expires_at.isoformat() if secret.expires_at else None
    }

@app.post("/tokens/generate", response_model=ShortLivedToken)
async def generate_short_lived_token(request: TokenRequest):
    """
    Generate short-lived access token
    """
    # Calculate expiration
    expires_at = datetime.utcnow() + timedelta(minutes=request.duration_minutes)
    
    # Create token record
    token = ShortLivedToken(
        token_type=request.token_type,
        tenant_id=request.tenant_id,
        service_name=request.service_name,
        target_resource=request.target_resource,
        access_level=request.access_level,
        expires_at=expires_at,
        max_usage_count=request.max_usage_count
    )
    
    # Generate JWT token
    jwt_payload = {
        "token_id": token.token_id,
        "tenant_id": request.tenant_id,
        "service_name": request.service_name,
        "target_resource": request.target_resource,
        "access_level": request.access_level.value,
        "iat": int(token.issued_at.timestamp()),
        "exp": int(expires_at.timestamp())
    }
    
    jwt_token = jwt.encode(jwt_payload, JWT_SECRET, algorithm="HS256")
    
    short_lived_tokens[token.token_id] = token
    
    # Log token generation
    await log_secret_access(
        "token_generated",
        token.token_id,
        request.service_name,
        request.tenant_id,
        f"Generated {request.token_type.value} token for {request.target_resource}"
    )
    
    logger.info(f"Generated short-lived token: {token.token_id} for {request.service_name}")
    
    # Return token with JWT
    token_dict = token.dict()
    token_dict["jwt_token"] = jwt_token
    
    return token_dict

@app.post("/tokens/validate")
async def validate_token(token: str, target_resource: str, tenant_id: str):
    """
    Validate short-lived token
    """
    try:
        # Decode JWT token
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        token_id = payload["token_id"]
        
        if token_id not in short_lived_tokens:
            raise HTTPException(status_code=404, detail="Token not found")
        
        token_record = short_lived_tokens[token_id]
        
        # Validate token status
        if token_record.is_revoked:
            raise HTTPException(status_code=403, detail="Token has been revoked")
        
        # Validate expiration
        if token_record.expires_at < datetime.utcnow():
            raise HTTPException(status_code=401, detail="Token has expired")
        
        # Validate tenant
        if token_record.tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch")
        
        # Validate target resource
        if token_record.target_resource != target_resource:
            raise HTTPException(status_code=403, detail="Resource mismatch")
        
        # Check usage limits
        if (token_record.max_usage_count and 
            token_record.usage_count >= token_record.max_usage_count):
            raise HTTPException(status_code=403, detail="Token usage limit exceeded")
        
        # Update usage tracking
        token_record.usage_count += 1
        
        # Log token validation
        await log_secret_access(
            "token_validated",
            token_id,
            token_record.service_name,
            tenant_id,
            f"Token validated for {target_resource}"
        )
        
        return {
            "valid": True,
            "token_id": token_id,
            "service_name": token_record.service_name,
            "access_level": token_record.access_level.value,
            "expires_at": token_record.expires_at.isoformat(),
            "usage_count": token_record.usage_count,
            "max_usage_count": token_record.max_usage_count
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/tokens/revoke")
async def revoke_token(token_id: str, tenant_id: str, reason: str):
    """
    Revoke a short-lived token
    """
    if token_id not in short_lived_tokens:
        raise HTTPException(status_code=404, detail="Token not found")
    
    token = short_lived_tokens[token_id]
    
    # Validate tenant access
    if token.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="Tenant access denied")
    
    # Revoke token
    token.is_revoked = True
    
    # Log revocation
    await log_secret_access(
        "token_revoked",
        token_id,
        token.service_name,
        tenant_id,
        f"Token revoked: {reason}"
    )
    
    logger.info(f"Token revoked: {token_id} - Reason: {reason}")
    
    return {
        "token_id": token_id,
        "revoked": True,
        "reason": reason,
        "revoked_at": datetime.utcnow().isoformat()
    }

@app.post("/secrets/rotate/{secret_id}")
async def rotate_secret(secret_id: str, new_secret_value: str):
    """
    Rotate a managed secret
    """
    if secret_id not in managed_secrets:
        raise HTTPException(status_code=404, detail="Secret not found")
    
    secret = managed_secrets[secret_id]
    
    # Update secret status
    secret.status = SecretStatus.ROTATING
    
    try:
        # Store new secret value in vault
        await store_secret_in_vault(secret.vault_path, new_secret_value)
        
        # Update secret metadata
        secret.last_rotated_at = datetime.utcnow()
        secret.expires_at = datetime.utcnow() + timedelta(days=secret.rotation_interval_days)
        secret.status = SecretStatus.ACTIVE
        
        # Log rotation
        await log_secret_access(
            "secret_rotated",
            secret_id,
            secret.service_name,
            secret.tenant_id,
            f"Secret rotated for {secret.name}"
        )
        
        logger.info(f"Secret rotated: {secret_id}")
        
        return {
            "secret_id": secret_id,
            "rotated": True,
            "rotated_at": secret.last_rotated_at.isoformat(),
            "new_expires_at": secret.expires_at.isoformat()
        }
        
    except Exception as e:
        secret.status = SecretStatus.ACTIVE  # Rollback status
        logger.error(f"Secret rotation failed: {secret_id} - {e}")
        raise HTTPException(status_code=500, detail=f"Secret rotation failed: {e}")

async def log_secret_access(action: str, resource_id: str, service_name: str, tenant_id: str, details: str):
    """Log secret access for audit trail"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "resource_id": resource_id,
        "service_name": service_name,
        "tenant_id": tenant_id,
        "details": details,
        "log_id": str(uuid.uuid4())
    }
    
    access_logs.append(log_entry)
    logger.info(f"Secret access logged: {action} - {resource_id} by {service_name}")

@app.get("/secrets")
async def list_secrets(
    tenant_id: Optional[str] = None,
    service_name: Optional[str] = None,
    secret_type: Optional[SecretType] = None,
    status: Optional[SecretStatus] = None
):
    """List managed secrets with filtering"""
    filtered_secrets = list(managed_secrets.values())
    
    if tenant_id:
        filtered_secrets = [s for s in filtered_secrets if s.tenant_id == tenant_id]
    
    if service_name:
        filtered_secrets = [s for s in filtered_secrets if s.service_name == service_name]
    
    if secret_type:
        filtered_secrets = [s for s in filtered_secrets if s.secret_type == secret_type]
    
    if status:
        filtered_secrets = [s for s in filtered_secrets if s.status == status]
    
    return {
        "secrets": [
            {
                "secret_id": s.secret_id,
                "name": s.name,
                "secret_type": s.secret_type.value,
                "tenant_id": s.tenant_id,
                "service_name": s.service_name,
                "status": s.status.value,
                "created_at": s.created_at.isoformat(),
                "expires_at": s.expires_at.isoformat() if s.expires_at else None,
                "usage_count": s.usage_count,
                "last_accessed_at": s.last_accessed_at.isoformat() if s.last_accessed_at else None
            }
            for s in filtered_secrets
        ],
        "total_count": len(filtered_secrets)
    }

@app.get("/tokens")
async def list_tokens(
    tenant_id: Optional[str] = None,
    service_name: Optional[str] = None,
    active_only: bool = True
):
    """List short-lived tokens"""
    filtered_tokens = list(short_lived_tokens.values())
    
    if tenant_id:
        filtered_tokens = [t for t in filtered_tokens if t.tenant_id == tenant_id]
    
    if service_name:
        filtered_tokens = [t for t in filtered_tokens if t.service_name == service_name]
    
    if active_only:
        now = datetime.utcnow()
        filtered_tokens = [t for t in filtered_tokens if not t.is_revoked and t.expires_at > now]
    
    return {
        "tokens": [
            {
                "token_id": t.token_id,
                "token_type": t.token_type.value,
                "tenant_id": t.tenant_id,
                "service_name": t.service_name,
                "target_resource": t.target_resource,
                "issued_at": t.issued_at.isoformat(),
                "expires_at": t.expires_at.isoformat(),
                "is_revoked": t.is_revoked,
                "usage_count": t.usage_count,
                "max_usage_count": t.max_usage_count
            }
            for t in filtered_tokens
        ],
        "total_count": len(filtered_tokens)
    }

@app.get("/audit-logs")
async def get_audit_logs(
    tenant_id: Optional[str] = None,
    service_name: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 100
):
    """Get secret access audit logs"""
    filtered_logs = access_logs
    
    if tenant_id:
        filtered_logs = [log for log in filtered_logs if log.get("tenant_id") == tenant_id]
    
    if service_name:
        filtered_logs = [log for log in filtered_logs if log.get("service_name") == service_name]
    
    if action:
        filtered_logs = [log for log in filtered_logs if log.get("action") == action]
    
    # Sort by timestamp, most recent first
    filtered_logs.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "audit_logs": filtered_logs[:limit],
        "total_count": len(filtered_logs)
    }

@app.get("/health")
async def health_check():
    active_secrets = len([s for s in managed_secrets.values() if s.status == SecretStatus.ACTIVE])
    active_tokens = len([t for t in short_lived_tokens.values() 
                        if not t.is_revoked and t.expires_at > datetime.utcnow()])
    
    return {
        "status": "healthy",
        "service": "RBIA Secrets Management Service",
        "task": "3.2.18",
        "active_secrets": active_secrets,
        "active_tokens": active_tokens,
        "total_access_logs": len(access_logs),
        "supported_secret_types": [st.value for st in SecretType],
        "supported_token_types": [tt.value for tt in TokenType]
    }

