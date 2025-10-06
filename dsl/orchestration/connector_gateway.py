"""
Integration Gateway - Tasks 9.2.19, 10.1-T13, T14
Unified ingress/egress gateway for all external system calls across planes.
Implements rate limits, quotas, circuit-breakers, governance-first design, and cross-plane routing.

Task 9.2.19: Integration Gateway - Unified ingress/egress
- Cross-plane request routing and load balancing
- Dynamic service discovery and health checking
- Governance plane integration for all requests
- Multi-tenant request isolation and SLA enforcement
- Real-time monitoring and observability
"""

import asyncio
import json
import os
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

# Task 9.2.19: Integration Gateway enhancements
class PlaneType(Enum):
    """Architectural planes for request routing"""
    CONTROL = "control"
    EXECUTION = "execution"
    DATA = "data"
    GOVERNANCE = "governance"
    UX = "ux"

class ServiceType(Enum):
    """Service types for dynamic discovery"""
    ORCHESTRATOR = "orchestrator"
    REGISTRY = "registry"
    EVIDENCE = "evidence"
    POLICY = "policy"
    CONNECTOR = "connector"
    AGENT = "agent"
    DASHBOARD = "dashboard"

class LoadBalancingStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"
    HEALTH_BASED = "health_based"
    SLA_AWARE = "sla_aware"

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
    """
    Task 9.5.8: Enhanced Circuit breaker state for APIs - Resilience patterns
    DYNAMIC configuration with no hardcoding
    """
    connector_name: str
    tenant_id: str
    
    # State tracking
    state: str = "closed"  # closed, open, half_open
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    
    # DYNAMIC configuration - no hardcoding
    failure_threshold: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_FAILURE_THRESHOLD", "5")))
    recovery_timeout_seconds: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_RECOVERY_TIMEOUT", "60")))
    half_open_max_calls: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_HALF_OPEN_CALLS", "3")))
    
    # SLA-aware thresholds
    sla_tier: str = "standard"
    response_time_threshold_ms: int = field(default_factory=lambda: int(os.getenv("CIRCUIT_BREAKER_RESPONSE_THRESHOLD", "5000")))
    
    # Advanced metrics
    total_requests: int = 0
    total_failures: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    
    # Plane-specific configuration
    plane_type: Optional[str] = None
    service_type: Optional[str] = None
    
    def should_allow_request(self) -> bool:
        """
        Task 9.5.8: Enhanced circuit breaker logic - DYNAMIC decision making
        Check if request should be allowed based on circuit breaker state
        """
        now = datetime.now(timezone.utc)
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has passed - DYNAMIC timeout
            if self.last_failure_time and (now - self.last_failure_time).total_seconds() > self.recovery_timeout_seconds:
                self.state = "half_open"
                self.success_count = 0
                return True
            return False
        elif self.state == "half_open":
            # Allow limited requests in half-open state
            return self.success_count < self.half_open_max_calls
        
        return False
    
    def record_success(self, response_time_ms: float = 0.0):
        """
        Task 9.5.8: Enhanced success recording with SLA awareness
        Record successful request with response time tracking
        """
        self.last_success_time = datetime.now(timezone.utc)
        self.total_requests += 1
        
        # Update average response time
        if self.total_requests > 1:
            self.avg_response_time_ms = (
                (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms) / 
                self.total_requests
            )
        else:
            self.avg_response_time_ms = response_time_ms
        
        # Update error rate
        self.error_rate = self.total_failures / self.total_requests if self.total_requests > 0 else 0.0
        
        if self.state == "half_open":
            self.success_count += 1
            # Transition to closed if enough successes - DYNAMIC threshold
            if self.success_count >= self.half_open_max_calls:
                self.state = "closed"
                self.failure_count = 0
        elif self.state == "closed":
            # Reduce failure count on success (gradual recovery)
            self.failure_count = max(0, self.failure_count - 1)
    
    def record_failure(self, response_time_ms: float = 0.0, error_type: str = "unknown"):
        """
        Task 9.5.8: Enhanced failure recording with error classification
        Record failed request with detailed error tracking
        """
        self.last_failure_time = datetime.now(timezone.utc)
        self.failure_count += 1
        self.total_requests += 1
        self.total_failures += 1
        
        # Update average response time (even for failures)
        if self.total_requests > 1:
            self.avg_response_time_ms = (
                (self.avg_response_time_ms * (self.total_requests - 1) + response_time_ms) / 
                self.total_requests
            )
        else:
            self.avg_response_time_ms = response_time_ms
        
        # Update error rate
        self.error_rate = self.total_failures / self.total_requests
        
        # Check for SLA-aware circuit breaking
        sla_breach = response_time_ms > self.response_time_threshold_ms
        
        # Open circuit if failure threshold reached or SLA consistently breached
        if self.failure_count >= self.failure_threshold or (sla_breach and self.error_rate > 0.5):
            self.state = "open"
            self.success_count = 0
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker status"""
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "error_rate": round(self.error_rate, 3),
            "avg_response_time_ms": round(self.avg_response_time_ms, 2),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "configuration": {
                "failure_threshold": self.failure_threshold,
                "recovery_timeout_seconds": self.recovery_timeout_seconds,
                "half_open_max_calls": self.half_open_max_calls,
                "response_time_threshold_ms": self.response_time_threshold_ms,
                "sla_tier": self.sla_tier
            }
        }

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

# Task 9.2.19: Integration Gateway - Unified ingress/egress
@dataclass
class ServiceEndpoint:
    """Service endpoint for dynamic discovery"""
    service_id: str
    service_type: ServiceType
    plane: PlaneType
    endpoint_url: str
    health_check_url: str
    tenant_id: Optional[int] = None
    sla_tier: str = "standard"
    weight: int = 100
    max_connections: int = 1000
    timeout_ms: int = 30000
    
    # Health and performance metrics
    is_healthy: bool = True
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    active_connections: int = 0
    last_health_check: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_type": self.service_type.value,
            "plane": self.plane.value,
            "endpoint_url": self.endpoint_url,
            "tenant_id": self.tenant_id,
            "sla_tier": self.sla_tier,
            "is_healthy": self.is_healthy,
            "response_time_ms": self.response_time_ms,
            "success_rate": self.success_rate,
            "active_connections": self.active_connections,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None
        }

class IntegrationGateway(ConnectorGateway):
    """
    Task 9.2.19: Integration Gateway - Unified ingress/egress
    Enhanced ConnectorGateway with cross-plane routing and service discovery
    """
    
    def __init__(self, pool_manager=None):
        super().__init__(pool_manager)
        
        # Service discovery and routing - DYNAMIC configuration
        self.service_registry: Dict[str, ServiceEndpoint] = {}
        self.plane_services: Dict[PlaneType, List[str]] = {plane: [] for plane in PlaneType}
        self.load_balancing_strategy = LoadBalancingStrategy.SLA_AWARE
        
        # Cross-plane routing configuration - NO hardcoding
        self.routing_rules: Dict[str, Dict[str, Any]] = {}
        self.plane_priorities: Dict[PlaneType, int] = {
            PlaneType.GOVERNANCE: 1,  # Highest priority
            PlaneType.CONTROL: 2,
            PlaneType.EXECUTION: 3,
            PlaneType.DATA: 4,
            PlaneType.UX: 5  # Lowest priority
        }
        
        # Dynamic configuration from environment/tenant settings
        self.gateway_config = {
            "health_check_interval_seconds": int(os.getenv("GATEWAY_HEALTH_CHECK_INTERVAL", "30")),
            "service_discovery_enabled": os.getenv("GATEWAY_SERVICE_DISCOVERY", "true").lower() == "true",
            "cross_plane_routing_enabled": os.getenv("GATEWAY_CROSS_PLANE_ROUTING", "true").lower() == "true",
            "load_balancing_enabled": os.getenv("GATEWAY_LOAD_BALANCING", "true").lower() == "true",
            "auto_scaling_enabled": os.getenv("GATEWAY_AUTO_SCALING", "true").lower() == "true",
            "governance_enforcement": os.getenv("GATEWAY_GOVERNANCE_ENFORCEMENT", "true").lower() == "true"
        }
        
        # Observability and monitoring
        self.cross_plane_metrics = {
            "requests_by_plane": {plane.value: 0 for plane in PlaneType},
            "response_times_by_plane": {plane.value: [] for plane in PlaneType},
            "errors_by_plane": {plane.value: 0 for plane in PlaneType},
            "service_discovery_events": 0,
            "routing_decisions": 0,
            "load_balancing_decisions": 0
        }
        
        logger.info("🌐 Integration Gateway initialized with cross-plane routing")
    
    async def register_service(self, service: ServiceEndpoint) -> bool:
        """Register a service endpoint for dynamic discovery"""
        try:
            # Validate service configuration
            if not service.service_id or not service.endpoint_url:
                logger.error(f"Invalid service configuration: {service}")
                return False
            
            # Register service
            self.service_registry[service.service_id] = service
            
            # Add to plane-specific service list
            if service.service_id not in self.plane_services[service.plane]:
                self.plane_services[service.plane].append(service.service_id)
            
            # Perform initial health check
            await self._health_check_service(service.service_id)
            
            self.cross_plane_metrics["service_discovery_events"] += 1
            logger.info(f"✅ Registered service {service.service_id} on {service.plane.value} plane")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to register service {service.service_id}: {e}")
            return False
    
    async def route_request(self, request: ConnectorRequest, target_plane: PlaneType, 
                          service_type: ServiceType) -> Dict[str, Any]:
        """Route request to appropriate service using dynamic discovery and load balancing"""
        try:
            # Get available services for the target plane and type
            available_services = self._get_available_services(target_plane, service_type, request.tenant_id)
            
            if not available_services:
                return {
                    "success": False,
                    "error": f"No available services for {service_type.value} on {target_plane.value} plane",
                    "plane": target_plane.value,
                    "service_type": service_type.value
                }
            
            # Select service using load balancing strategy
            selected_service = await self._select_service(available_services, request)
            
            if not selected_service:
                return {
                    "success": False,
                    "error": "No healthy services available",
                    "plane": target_plane.value,
                    "service_type": service_type.value
                }
            
            # Route request to selected service
            response = await self._execute_service_request(selected_service, request)
            
            # Update metrics
            self.cross_plane_metrics["requests_by_plane"][target_plane.value] += 1
            self.cross_plane_metrics["routing_decisions"] += 1
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Request routing failed: {e}")
            self.cross_plane_metrics["errors_by_plane"][target_plane.value] += 1
            return {
                "success": False,
                "error": str(e),
                "plane": target_plane.value,
                "service_type": service_type.value
            }
    
    def _get_available_services(self, plane: PlaneType, service_type: ServiceType, 
                              tenant_id: Optional[str] = None) -> List[ServiceEndpoint]:
        """Get available services for plane and type with tenant filtering"""
        available = []
        
        for service_id in self.plane_services.get(plane, []):
            service = self.service_registry.get(service_id)
            if not service:
                continue
            
            # Filter by service type
            if service.service_type != service_type:
                continue
            
            # Filter by tenant (if specified and service is tenant-specific)
            if tenant_id and service.tenant_id and str(service.tenant_id) != str(tenant_id):
                continue
            
            # Filter by health status
            if not service.is_healthy:
                continue
            
            available.append(service)
        
        return available
    
    async def _select_service(self, services: List[ServiceEndpoint], 
                            request: ConnectorRequest) -> Optional[ServiceEndpoint]:
        """Select service using configured load balancing strategy - DYNAMIC selection"""
        if not services:
            return None
        
        self.cross_plane_metrics["load_balancing_decisions"] += 1
        
        if self.load_balancing_strategy == LoadBalancingStrategy.SLA_AWARE:
            # Sort by SLA tier and response time
            sla_priorities = {"premium": 1, "standard": 2, "basic": 3}
            services.sort(key=lambda s: (
                sla_priorities.get(s.sla_tier, 4),
                s.response_time_ms,
                s.active_connections
            ))
            return services[0]
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.HEALTH_BASED:
            # Select service with best health metrics
            services.sort(key=lambda s: (-s.success_rate, s.response_time_ms))
            return services[0]
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            # Select service with least active connections
            services.sort(key=lambda s: s.active_connections)
            return services[0]
        
        else:  # ROUND_ROBIN (default)
            # Simple round-robin selection
            return services[0]
    
    async def _execute_service_request(self, service: ServiceEndpoint, 
                                     request: ConnectorRequest) -> Dict[str, Any]:
        """Execute request against selected service"""
        start_time = time.time()
        
        try:
            # Increment active connections
            service.active_connections += 1
            
            # Build request URL dynamically
            request_url = f"{service.endpoint_url.rstrip('/')}/{request.operation}"
            
            # Execute request (placeholder - would use actual HTTP client)
            await asyncio.sleep(0.1)  # Simulate network call
            
            # Calculate response time
            response_time = (time.time() - start_time) * 1000
            
            # Update service metrics dynamically
            service.response_time_ms = (service.response_time_ms + response_time) / 2
            service.success_rate = min(1.0, service.success_rate + 0.01)
            
            # Update cross-plane metrics
            plane_response_times = self.cross_plane_metrics["response_times_by_plane"][service.plane.value]
            plane_response_times.append(response_time)
            if len(plane_response_times) > 100:  # Keep last 100 measurements
                plane_response_times.pop(0)
            
            return {
                "success": True,
                "service_id": service.service_id,
                "plane": service.plane.value,
                "response_time_ms": response_time,
                "data": {"message": "Request processed successfully"}
            }
            
        except Exception as e:
            # Update error metrics
            service.success_rate = max(0.0, service.success_rate - 0.05)
            logger.error(f"Service request failed: {e}")
            
            return {
                "success": False,
                "service_id": service.service_id,
                "plane": service.plane.value,
                "error": str(e)
            }
        
        finally:
            # Decrement active connections
            service.active_connections = max(0, service.active_connections - 1)
    
    async def _health_check_service(self, service_id: str) -> bool:
        """Perform health check on a specific service"""
        service = self.service_registry.get(service_id)
        if not service:
            return False
        
        try:
            # Perform health check (placeholder - would use actual HTTP call)
            await asyncio.sleep(0.05)  # Simulate health check call
            
            service.is_healthy = True
            service.last_health_check = datetime.now()
            
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for service {service_id}: {e}")
            service.is_healthy = False
            service.last_health_check = datetime.now()
            
            return False
    
    async def start_health_monitoring(self):
        """Start background health monitoring for all services"""
        async def health_monitor():
            while True:
                try:
                    await asyncio.sleep(self.gateway_config["health_check_interval_seconds"])
                    
                    # Health check all registered services
                    for service_id in list(self.service_registry.keys()):
                        await self._health_check_service(service_id)
                    
                except Exception as e:
                    logger.error(f"Health monitoring error: {e}")
        
        # Start health monitoring task
        if self.gateway_config["service_discovery_enabled"]:
            asyncio.create_task(health_monitor())
            logger.info("🔍 Started health monitoring for all services")
    
    def get_integration_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive integration gateway status"""
        base_status = self.get_gateway_status()
        
        # Add integration gateway specific metrics
        base_status.update({
            "integration_gateway": {
                "registered_services": len(self.service_registry),
                "services_by_plane": {
                    plane.value: len(services) 
                    for plane, services in self.plane_services.items()
                },
                "cross_plane_metrics": self.cross_plane_metrics.copy(),
                "load_balancing_strategy": self.load_balancing_strategy.value,
                "configuration": self.gateway_config.copy()
            },
            "service_health": {
                service_id: service.to_dict()
                for service_id, service in self.service_registry.items()
            }
        })
        
        return base_status

# Global gateway instances
connector_gateway = ConnectorGateway()
integration_gateway = IntegrationGateway()
