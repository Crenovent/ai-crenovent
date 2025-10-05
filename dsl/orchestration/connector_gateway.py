"""
RBA Connector Gateway - Task 10.1-T13, T14
Policy-aware connector gateway that serves as single choke point for all external system calls.
Implements rate limits, quotas, circuit-breakers, and governance-first design.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)

class GatewayStatus(Enum):
    """Gateway operational status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CIRCUIT_OPEN = "circuit_open"
    MAINTENANCE = "maintenance"

class PolicyDecision(Enum):
    """Policy enforcement decision"""
    ALLOW = "allow"
    DENY = "deny"
    THROTTLE = "throttle"
    AUDIT = "audit"

@dataclass
class ConnectorRequest:
    """Connector request context"""
    request_id: str
    tenant_id: str
    user_id: str
    connector_name: str
    operation: str
    endpoint: str
    method: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    data: Optional[Dict[str, Any]] = None
    
    # Governance context
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    policy_context: Dict[str, Any] = field(default_factory=dict)
    
    # Request metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/audit"""
        return {
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "connector_name": self.connector_name,
            "operation": self.operation,
            "endpoint": self.endpoint,
            "method": self.method,
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "source_ip": self.source_ip,
            "user_agent": self.user_agent
        }

@dataclass
class PolicyEnforcementResult:
    """Result of policy enforcement check"""
    decision: PolicyDecision
    allowed: bool
    reason: str
    
    # Rate limiting
    rate_limit_remaining: Optional[int] = None
    rate_limit_reset: Optional[datetime] = None
    
    # Quotas
    quota_remaining: Optional[int] = None
    quota_reset: Optional[datetime] = None
    
    # Additional context
    policy_violations: List[str] = field(default_factory=list)
    required_approvals: List[str] = field(default_factory=list)
    audit_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision": self.decision.value,
            "allowed": self.allowed,
            "reason": self.reason,
            "rate_limit_remaining": self.rate_limit_remaining,
            "rate_limit_reset": self.rate_limit_reset.isoformat() if self.rate_limit_reset else None,
            "quota_remaining": self.quota_remaining,
            "quota_reset": self.quota_reset.isoformat() if self.quota_reset else None,
            "policy_violations": self.policy_violations,
            "required_approvals": self.required_approvals,
            "audit_required": self.audit_required
        }

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for a connector"""
    connector_name: str
    tenant_id: str
    
    # State tracking
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    # Configuration
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open -> closed transition
    
    def should_allow_request(self) -> bool:
        """Check if request should be allowed based on circuit breaker state"""
        now = datetime.now(timezone.utc)
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time and (now - self.last_failure_time).total_seconds() > self.recovery_timeout:
                self.state = "half_open"
                self.success_count = 0
                return True
            return False
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful request"""
        self.last_success_time = datetime.now(timezone.utc)
        
        if self.state == "half_open":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "closed"
                self.failure_count = 0
        elif self.state == "closed":
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self):
        """Record failed request"""
        self.last_failure_time = datetime.now(timezone.utc)
        self.failure_count += 1
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            self.success_count = 0

class ConnectorGateway:
    """
    Policy-aware connector gateway - single choke point for all external system calls
    Implements governance-first design with rate limits, quotas, circuit breakers
    """
    
    def __init__(self, policy_engine=None, evidence_service=None, override_service=None, risk_service=None):
        self.logger = logging.getLogger(__name__)
        
        # Core services
        self.policy_engine = policy_engine
        self.evidence_service = evidence_service
        self.override_service = override_service
        self.risk_service = risk_service
        
        # Gateway state
        self.status = GatewayStatus.HEALTHY
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Rate limiting and quotas
        self.rate_limits: Dict[str, Dict[str, Any]] = {}
        self.quotas: Dict[str, Dict[str, Any]] = {}
        
        # Metrics and monitoring
        self.metrics = {
            "requests_total": 0,
            "requests_allowed": 0,
            "requests_denied": 0,
            "requests_throttled": 0,
            "policy_violations": 0,
            "circuit_breaker_trips": 0,
            "average_response_time": 0.0
        }
        
        # Request tracking
        self.active_requests: Dict[str, ConnectorRequest] = {}
        
        self.logger.info("🚪 Connector Gateway initialized with governance-first design")
    
    async def process_request(self, request: ConnectorRequest) -> Dict[str, Any]:
        """
        Process connector request through policy-aware gateway
        Main entry point for all external system calls
        """
        start_time = time.time()
        self.metrics["requests_total"] += 1
        
        try:
            # Track active request
            self.active_requests[request.request_id] = request
            
            self.logger.info(f"🚪 Processing connector request: {request.connector_name}/{request.operation}")
            
            # Step 1: Policy enforcement
            policy_result = await self._enforce_policies(request)
            
            if not policy_result.allowed:
                self.metrics["requests_denied"] += 1
                if policy_result.policy_violations:
                    self.metrics["policy_violations"] += 1
                
                await self._log_policy_violation(request, policy_result)
                
                return {
                    "success": False,
                    "error": "Policy violation",
                    "reason": policy_result.reason,
                    "policy_result": policy_result.to_dict(),
                    "request_id": request.request_id
                }
            
            # Step 2: Circuit breaker check
            circuit_key = f"{request.tenant_id}:{request.connector_name}"
            circuit_breaker = self._get_circuit_breaker(circuit_key, request.connector_name, request.tenant_id)
            
            if not circuit_breaker.should_allow_request():
                self.metrics["requests_denied"] += 1
                self.metrics["circuit_breaker_trips"] += 1
                
                await self._log_circuit_breaker_trip(request, circuit_breaker)
                
                return {
                    "success": False,
                    "error": "Circuit breaker open",
                    "reason": f"Circuit breaker is {circuit_breaker.state} for {request.connector_name}",
                    "circuit_breaker_state": circuit_breaker.state,
                    "request_id": request.request_id
                }
            
            # Step 3: Rate limiting and throttling
            if policy_result.decision == PolicyDecision.THROTTLE:
                self.metrics["requests_throttled"] += 1
                await asyncio.sleep(1.0)  # Simple throttling delay
            
            # Step 4: Evidence capture (pre-execution)
            evidence_id = await self._capture_pre_execution_evidence(request, policy_result)
            
            # Step 5: Execute the actual connector call
            # Note: This would delegate to the actual connector implementation
            # For now, we'll simulate the call
            execution_result = await self._execute_connector_call(request)
            
            # Step 6: Record success/failure for circuit breaker
            if execution_result.get("success", False):
                circuit_breaker.record_success()
                self.metrics["requests_allowed"] += 1
            else:
                circuit_breaker.record_failure()
            
            # Step 7: Evidence capture (post-execution)
            await self._capture_post_execution_evidence(request, execution_result, evidence_id)
            
            # Step 8: Risk assessment
            await self._assess_execution_risk(request, execution_result)
            
            # Update metrics
            duration = time.time() - start_time
            self._update_response_time_metric(duration)
            
            self.logger.info(f"✅ Connector request completed: {request.request_id} in {duration:.3f}s")
            
            return {
                **execution_result,
                "request_id": request.request_id,
                "evidence_id": evidence_id,
                "policy_result": policy_result.to_dict(),
                "duration": duration
            }
            
        except Exception as e:
            self.metrics["requests_denied"] += 1
            self.logger.error(f"❌ Gateway error processing request {request.request_id}: {e}")
            
            # Record failure for circuit breaker
            circuit_key = f"{request.tenant_id}:{request.connector_name}"
            if circuit_key in self.circuit_breakers:
                self.circuit_breakers[circuit_key].record_failure()
            
            return {
                "success": False,
                "error": "Gateway processing error",
                "reason": str(e),
                "request_id": request.request_id
            }
        
        finally:
            # Clean up active request tracking
            self.active_requests.pop(request.request_id, None)
    
    async def _enforce_policies(self, request: ConnectorRequest) -> PolicyEnforcementResult:
        """Enforce governance policies on connector request"""
        try:
            # Rate limiting check
            rate_limit_result = self._check_rate_limits(request)
            if not rate_limit_result["allowed"]:
                return PolicyEnforcementResult(
                    decision=PolicyDecision.DENY,
                    allowed=False,
                    reason="Rate limit exceeded",
                    rate_limit_remaining=rate_limit_result.get("remaining", 0),
                    rate_limit_reset=rate_limit_result.get("reset_time")
                )
            
            # Quota check
            quota_result = self._check_quotas(request)
            if not quota_result["allowed"]:
                return PolicyEnforcementResult(
                    decision=PolicyDecision.DENY,
                    allowed=False,
                    reason="Quota exceeded",
                    quota_remaining=quota_result.get("remaining", 0),
                    quota_reset=quota_result.get("reset_time")
                )
            
            # Data scope validation
            scope_violations = self._validate_data_scope(request)
            if scope_violations:
                return PolicyEnforcementResult(
                    decision=PolicyDecision.DENY,
                    allowed=False,
                    reason="Data scope violation",
                    policy_violations=scope_violations
                )
            
            # Residency compliance check
            residency_violations = self._check_residency_compliance(request)
            if residency_violations:
                return PolicyEnforcementResult(
                    decision=PolicyDecision.DENY,
                    allowed=False,
                    reason="Residency compliance violation",
                    policy_violations=residency_violations
                )
            
            # SoD (Segregation of Duties) check
            sod_violations = self._check_sod_compliance(request)
            if sod_violations:
                return PolicyEnforcementResult(
                    decision=PolicyDecision.DENY,
                    allowed=False,
                    reason="Segregation of Duties violation",
                    policy_violations=sod_violations,
                    required_approvals=["manager_approval", "compliance_approval"]
                )
            
            # Budget/cost check
            if self._exceeds_budget_limits(request):
                return PolicyEnforcementResult(
                    decision=PolicyDecision.THROTTLE,
                    allowed=True,
                    reason="Budget limits approached - throttling applied",
                    audit_required=True
                )
            
            # All checks passed
            return PolicyEnforcementResult(
                decision=PolicyDecision.ALLOW,
                allowed=True,
                reason="All policy checks passed",
                rate_limit_remaining=rate_limit_result.get("remaining"),
                quota_remaining=quota_result.get("remaining"),
                audit_required=self._requires_audit(request)
            )
            
        except Exception as e:
            self.logger.error(f"❌ Policy enforcement error: {e}")
            # Fail closed - deny on policy engine errors
            return PolicyEnforcementResult(
                decision=PolicyDecision.DENY,
                allowed=False,
                reason=f"Policy engine error: {str(e)}"
            )
    
    def _check_rate_limits(self, request: ConnectorRequest) -> Dict[str, Any]:
        """Check rate limits for tenant/connector combination"""
        rate_limit_key = f"{request.tenant_id}:{request.connector_name}"
        now = datetime.now(timezone.utc)
        
        if rate_limit_key not in self.rate_limits:
            # Initialize rate limit tracking
            self.rate_limits[rate_limit_key] = {
                "requests": [],
                "window_size": 3600,  # 1 hour window
                "max_requests": 1000  # Default limit
            }
        
        rate_limit = self.rate_limits[rate_limit_key]
        
        # Clean old requests outside window
        window_start = now - timedelta(seconds=rate_limit["window_size"])
        rate_limit["requests"] = [
            req_time for req_time in rate_limit["requests"] 
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(rate_limit["requests"]) >= rate_limit["max_requests"]:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": window_start + timedelta(seconds=rate_limit["window_size"])
            }
        
        # Record this request
        rate_limit["requests"].append(now)
        
        return {
            "allowed": True,
            "remaining": rate_limit["max_requests"] - len(rate_limit["requests"]),
            "reset_time": window_start + timedelta(seconds=rate_limit["window_size"])
        }
    
    def _check_quotas(self, request: ConnectorRequest) -> Dict[str, Any]:
        """Check quotas for tenant/connector combination"""
        quota_key = f"{request.tenant_id}:{request.connector_name}"
        now = datetime.now(timezone.utc)
        
        if quota_key not in self.quotas:
            # Initialize quota tracking
            self.quotas[quota_key] = {
                "used": 0,
                "limit": 10000,  # Default monthly limit
                "reset_date": now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)
            }
        
        quota = self.quotas[quota_key]
        
        # Check if quota period has reset
        if now >= quota["reset_date"]:
            quota["used"] = 0
            quota["reset_date"] = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0) + timedelta(days=32)
        
        # Check if quota exceeded
        if quota["used"] >= quota["limit"]:
            return {
                "allowed": False,
                "remaining": 0,
                "reset_time": quota["reset_date"]
            }
        
        # Record usage
        quota["used"] += 1
        
        return {
            "allowed": True,
            "remaining": quota["limit"] - quota["used"],
            "reset_time": quota["reset_date"]
        }
    
    def _validate_data_scope(self, request: ConnectorRequest) -> List[str]:
        """Validate data access scope for the request"""
        violations = []
        
        # Check if requesting sensitive data without proper authorization
        if "sensitive" in request.endpoint.lower() and not request.policy_context.get("sensitive_data_approved"):
            violations.append("Sensitive data access requires explicit approval")
        
        # Check bulk operations
        if request.method in ["DELETE", "PUT"] and "bulk" in request.operation.lower():
            if not request.policy_context.get("bulk_operations_approved"):
                violations.append("Bulk operations require manager approval")
        
        # Check cross-tenant data access
        if request.params.get("tenant_id") and request.params["tenant_id"] != request.tenant_id:
            violations.append("Cross-tenant data access not allowed")
        
        return violations
    
    def _check_residency_compliance(self, request: ConnectorRequest) -> List[str]:
        """Check data residency compliance"""
        violations = []
        
        # Simple residency check based on tenant region
        tenant_region = request.policy_context.get("tenant_region", "US")
        connector_region = request.policy_context.get("connector_region", "US")
        
        # GDPR compliance - EU data must stay in EU
        if tenant_region == "EU" and connector_region not in ["EU", "EEA"]:
            violations.append("GDPR violation: EU tenant data cannot be processed outside EU/EEA")
        
        # DPDP compliance - Indian data residency
        if tenant_region == "IN" and connector_region != "IN":
            violations.append("DPDP violation: Indian tenant data must be processed within India")
        
        return violations
    
    def _check_sod_compliance(self, request: ConnectorRequest) -> List[str]:
        """Check Segregation of Duties compliance"""
        violations = []
        
        # Check if user is trying to approve their own actions
        if request.operation == "approve" and request.policy_context.get("created_by") == request.user_id:
            violations.append("SoD violation: Cannot approve own actions")
        
        # Check financial operations
        if "payment" in request.operation.lower() or "disbursal" in request.operation.lower():
            if not request.policy_context.get("financial_approval"):
                violations.append("SoD violation: Financial operations require separate approval")
        
        return violations
    
    def _exceeds_budget_limits(self, request: ConnectorRequest) -> bool:
        """Check if request exceeds budget limits"""
        # Simple budget check - in production this would integrate with cost tracking
        estimated_cost = request.policy_context.get("estimated_cost", 0.01)
        budget_remaining = request.policy_context.get("budget_remaining", 1000.0)
        
        return estimated_cost > budget_remaining * 0.8  # Throttle when 80% budget used
    
    def _requires_audit(self, request: ConnectorRequest) -> bool:
        """Check if request requires audit logging"""
        # Audit high-risk operations
        high_risk_operations = ["delete", "update", "create", "approve", "disbursal"]
        return any(op in request.operation.lower() for op in high_risk_operations)
    
    def _get_circuit_breaker(self, circuit_key: str, connector_name: str, tenant_id: str) -> CircuitBreakerState:
        """Get or create circuit breaker for connector/tenant combination"""
        if circuit_key not in self.circuit_breakers:
            self.circuit_breakers[circuit_key] = CircuitBreakerState(
                connector_name=connector_name,
                tenant_id=tenant_id
            )
        return self.circuit_breakers[circuit_key]
    
    async def _execute_connector_call(self, request: ConnectorRequest) -> Dict[str, Any]:
        """Execute the actual connector call - placeholder for now"""
        # This would delegate to the actual connector implementation
        # For now, simulate a successful call
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "success": True,
            "data": {"message": f"Simulated {request.operation} on {request.connector_name}"},
            "status_code": 200,
            "response_time": 0.1
        }
    
    async def _capture_pre_execution_evidence(self, request: ConnectorRequest, policy_result: PolicyEnforcementResult) -> str:
        """Capture evidence before execution"""
        evidence_id = str(uuid.uuid4())
        
        if self.evidence_service:
            try:
                await self.evidence_service.capture_evidence({
                    "evidence_id": evidence_id,
                    "type": "connector_pre_execution",
                    "request": request.to_dict(),
                    "policy_result": policy_result.to_dict(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                self.logger.error(f"❌ Failed to capture pre-execution evidence: {e}")
        
        return evidence_id
    
    async def _capture_post_execution_evidence(self, request: ConnectorRequest, result: Dict[str, Any], evidence_id: str):
        """Capture evidence after execution"""
        if self.evidence_service:
            try:
                await self.evidence_service.capture_evidence({
                    "evidence_id": evidence_id,
                    "type": "connector_post_execution",
                    "request": request.to_dict(),
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                self.logger.error(f"❌ Failed to capture post-execution evidence: {e}")
    
    async def _assess_execution_risk(self, request: ConnectorRequest, result: Dict[str, Any]):
        """Assess risk of the executed operation"""
        if self.risk_service:
            try:
                risk_factors = []
                
                # Check for unusual patterns
                if result.get("response_time", 0) > 5.0:
                    risk_factors.append("Unusually slow response time")
                
                if not result.get("success", False):
                    risk_factors.append("Operation failed")
                
                if request.method == "DELETE":
                    risk_factors.append("Destructive operation")
                
                if risk_factors:
                    await self.risk_service.log_risk({
                        "request_id": request.request_id,
                        "tenant_id": request.tenant_id,
                        "connector_name": request.connector_name,
                        "risk_factors": risk_factors,
                        "severity": "medium" if len(risk_factors) > 1 else "low",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            except Exception as e:
                self.logger.error(f"❌ Failed to assess execution risk: {e}")
    
    async def _log_policy_violation(self, request: ConnectorRequest, policy_result: PolicyEnforcementResult):
        """Log policy violation for audit"""
        self.logger.warning(f"🚫 Policy violation: {request.request_id} - {policy_result.reason}")
        
        if self.override_service:
            try:
                await self.override_service.log_violation({
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "user_id": request.user_id,
                    "violation_type": "policy",
                    "reason": policy_result.reason,
                    "violations": policy_result.policy_violations,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                self.logger.error(f"❌ Failed to log policy violation: {e}")
    
    async def _log_circuit_breaker_trip(self, request: ConnectorRequest, circuit_breaker: CircuitBreakerState):
        """Log circuit breaker trip"""
        self.logger.warning(f"⚡ Circuit breaker trip: {request.connector_name} for tenant {request.tenant_id}")
        
        if self.risk_service:
            try:
                await self.risk_service.log_risk({
                    "request_id": request.request_id,
                    "tenant_id": request.tenant_id,
                    "connector_name": request.connector_name,
                    "risk_type": "circuit_breaker_trip",
                    "circuit_state": circuit_breaker.state,
                    "failure_count": circuit_breaker.failure_count,
                    "severity": "high",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            except Exception as e:
                self.logger.error(f"❌ Failed to log circuit breaker trip: {e}")
    
    def _update_response_time_metric(self, duration: float):
        """Update average response time metric"""
        if self.metrics["requests_total"] == 1:
            self.metrics["average_response_time"] = duration
        else:
            # Simple moving average
            self.metrics["average_response_time"] = (
                (self.metrics["average_response_time"] * (self.metrics["requests_total"] - 1) + duration) /
                self.metrics["requests_total"]
            )
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get gateway status and metrics"""
        return {
            "status": self.status.value,
            "metrics": self.metrics.copy(),
            "active_requests": len(self.active_requests),
            "circuit_breakers": {
                key: {
                    "state": cb.state,
                    "failure_count": cb.failure_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for key, cb in self.circuit_breakers.items()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform gateway health check"""
        try:
            # Check if critical services are available
            services_healthy = True
            service_status = {}
            
            if self.policy_engine:
                try:
                    # Simulate policy engine health check
                    service_status["policy_engine"] = "healthy"
                except:
                    service_status["policy_engine"] = "unhealthy"
                    services_healthy = False
            
            if self.evidence_service:
                try:
                    # Simulate evidence service health check
                    service_status["evidence_service"] = "healthy"
                except:
                    service_status["evidence_service"] = "unhealthy"
                    services_healthy = False
            
            # Update gateway status
            if services_healthy:
                self.status = GatewayStatus.HEALTHY
            else:
                self.status = GatewayStatus.DEGRADED
            
            return {
                "gateway_status": self.status.value,
                "services": service_status,
                "metrics": self.metrics.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
            self.status = GatewayStatus.DEGRADED
            return {
                "gateway_status": self.status.value,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

# Global gateway instance
connector_gateway = ConnectorGateway()
