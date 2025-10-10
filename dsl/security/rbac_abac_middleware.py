# RBAC/ABAC Enforcement Middleware for FastAPI
# Tasks 18.1.8, 18.1.9, 18.1.15: RBAC/ABAC enforcement middleware and orchestration binding

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class AccessDecision(Enum):
    """Access control decisions"""
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"

class PolicyType(Enum):
    """ABAC policy types"""
    REGIONAL = "regional"
    TEMPORAL = "temporal"
    SENSITIVITY = "sensitivity"
    INDUSTRY = "industry"
    SLA_TIER = "sla_tier"

@dataclass
class AccessContext:
    """Access control context"""
    user_id: int
    tenant_id: int
    role_ids: List[str]
    permissions: Set[str]
    attributes: Dict[str, Any]
    request_path: str
    request_method: str
    resource_id: Optional[str] = None
    industry_overlay: Optional[str] = None
    region: Optional[str] = None
    sla_tier: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

@dataclass
class AccessResult:
    """Access control result"""
    decision: AccessDecision
    reason: str
    applicable_policies: List[str]
    violations: List[str]
    context: AccessContext
    evidence_id: Optional[str] = None
    override_required: bool = False

class RBACMiddleware:
    """
    RBAC Enforcement Middleware
    Task 18.1.8: Implement RBAC enforcement middleware
    """
    
    def __init__(self):
        self.role_cache: Dict[str, Dict[str, Any]] = {}
        self.permission_cache: Dict[str, Set[str]] = {}
        self.security = HTTPBearer()
    
    async def __call__(self, request: Request, call_next):
        """RBAC middleware execution"""
        try:
            # Extract authorization context
            auth_context = await self._extract_auth_context(request)
            
            if not auth_context:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # Check RBAC permissions
            access_result = await self._check_rbac_permissions(auth_context)
            
            if access_result.decision == AccessDecision.DENY:
                await self._log_access_denial(access_result)
                raise HTTPException(status_code=403, detail=f"Access denied: {access_result.reason}")
            
            # Add context to request state
            request.state.auth_context = auth_context
            request.state.access_result = access_result
            
            # Log successful access
            await self._log_access_success(access_result)
            
            response = await call_next(request)
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ RBAC middleware error: {e}")
            raise HTTPException(status_code=500, detail="Authentication service error")
    
    async def _extract_auth_context(self, request: Request) -> Optional[AccessContext]:
        """Extract authentication context from request"""
        try:
            # Get authorization header
            authorization = request.headers.get("Authorization")
            if not authorization or not authorization.startswith("Bearer "):
                return None
            
            token = authorization.split(" ")[1]
            
            # Decode JWT token (in production, verify signature)
            payload = jwt.decode(token, options={"verify_signature": False})
            
            # Extract user information
            user_id = payload.get("user_id")
            tenant_id = payload.get("tenant_id")
            role_ids = payload.get("roles", [])
            
            if not all([user_id, tenant_id]):
                return None
            
            # Get permissions for roles
            permissions = await self._get_permissions_for_roles(role_ids, tenant_id)
            
            # Extract additional attributes
            attributes = {
                "industry_overlay": payload.get("industry_overlay", "global"),
                "region": payload.get("region", "GLOBAL"),
                "sla_tier": payload.get("sla_tier", "bronze"),
                "clearance_level": payload.get("clearance_level", 1),
                "department": payload.get("department"),
                "cost_center": payload.get("cost_center")
            }
            
            return AccessContext(
                user_id=user_id,
                tenant_id=tenant_id,
                role_ids=role_ids,
                permissions=permissions,
                attributes=attributes,
                request_path=str(request.url.path),
                request_method=request.method,
                resource_id=request.path_params.get("id"),
                industry_overlay=attributes["industry_overlay"],
                region=attributes["region"],
                sla_tier=attributes["sla_tier"]
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to extract auth context: {e}")
            return None
    
    async def _get_permissions_for_roles(self, role_ids: List[str], tenant_id: int) -> Set[str]:
        """Get permissions for user roles"""
        permissions = set()
        
        for role_id in role_ids:
            cache_key = f"{tenant_id}:{role_id}"
            
            if cache_key in self.permission_cache:
                permissions.update(self.permission_cache[cache_key])
                continue
            
            # In production, query database
            role_permissions = await self._query_role_permissions(role_id, tenant_id)
            self.permission_cache[cache_key] = role_permissions
            permissions.update(role_permissions)
        
        return permissions
    
    async def _query_role_permissions(self, role_id: str, tenant_id: int) -> Set[str]:
        """Query role permissions from database"""
        # Mock implementation - in production, query authz_policy_role table
        role_permission_map = {
            "cro": {"workflow_read", "workflow_approve", "workflow_publish", "audit_read", "audit_export"},
            "cfo": {"workflow_read", "workflow_approve", "audit_read", "audit_export"},
            "revops_manager": {"workflow_read", "workflow_create", "workflow_edit", "dashboard_access"},
            "compliance_officer": {"policy_read", "policy_create", "audit_read", "compliance_report"},
            "sales_exec": {"workflow_read", "dashboard_access"},
            "developer": {"workflow_create", "workflow_edit", "workflow_test"}
        }
        
        return set(role_permission_map.get(role_id, []))
    
    async def _check_rbac_permissions(self, context: AccessContext) -> AccessResult:
        """Check RBAC permissions for request"""
        violations = []
        
        # Define required permissions for endpoints
        endpoint_permissions = {
            "/api/workflows": {"GET": "workflow_read", "POST": "workflow_create"},
            "/api/policies": {"GET": "policy_read", "POST": "policy_create"},
            "/api/audit": {"GET": "audit_read"},
            "/api/dashboards": {"GET": "dashboard_access"}
        }
        
        # Check if endpoint requires specific permission
        required_permission = None
        for endpoint, methods in endpoint_permissions.items():
            if context.request_path.startswith(endpoint):
                required_permission = methods.get(context.request_method)
                break
        
        if required_permission and required_permission not in context.permissions:
            violations.append(f"Missing required permission: {required_permission}")
        
        # Check industry overlay access
        if context.industry_overlay != "global":
            industry_permission = f"industry_{context.industry_overlay}_access"
            if industry_permission not in context.permissions:
                violations.append(f"Missing industry overlay permission: {industry_permission}")
        
        decision = AccessDecision.DENY if violations else AccessDecision.ALLOW
        reason = "; ".join(violations) if violations else "RBAC permissions granted"
        
        return AccessResult(
            decision=decision,
            reason=reason,
            applicable_policies=["rbac_enforcement"],
            violations=violations,
            context=context
        )
    
    async def _log_access_success(self, result: AccessResult) -> None:
        """Log successful access"""
        logger.info(f"✅ RBAC access granted: user={result.context.user_id}, "
                   f"tenant={result.context.tenant_id}, path={result.context.request_path}")
    
    async def _log_access_denial(self, result: AccessResult) -> None:
        """Log access denial"""
        logger.warning(f"❌ RBAC access denied: user={result.context.user_id}, "
                      f"tenant={result.context.tenant_id}, path={result.context.request_path}, "
                      f"reason={result.reason}")

class ABACMiddleware:
    """
    ABAC Enforcement Middleware
    Task 18.1.9: Implement ABAC enforcement middleware
    """
    
    def __init__(self):
        self.policy_cache: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_policies()
    
    def _initialize_default_policies(self):
        """Initialize default ABAC policies"""
        # Task 18.1.17: Regional access attributes
        self.policy_cache["regional_access"] = {
            "policy_type": PolicyType.REGIONAL,
            "rules": [
                {
                    "condition": "user.region == resource.region",
                    "action": "allow",
                    "description": "Allow same-region access"
                },
                {
                    "condition": "user.region == 'US' and resource.region == 'EU' and user.gdpr_trained == true",
                    "action": "allow",
                    "description": "Allow US-EU access with GDPR training"
                },
                {
                    "condition": "resource.sensitivity == 'high' and user.clearance_level < 3",
                    "action": "deny",
                    "description": "Deny high sensitivity access for low clearance"
                }
            ]
        }
        
        # Task 18.1.18: SLA tier attributes
        self.policy_cache["sla_tier_access"] = {
            "policy_type": PolicyType.SLA_TIER,
            "rules": [
                {
                    "condition": "user.sla_tier == 'enterprise' and resource.type == 'premium_feature'",
                    "action": "allow",
                    "description": "Allow enterprise users premium features"
                },
                {
                    "condition": "user.sla_tier == 'bronze' and resource.type == 'premium_feature'",
                    "action": "deny",
                    "description": "Deny bronze users premium features"
                }
            ]
        }
        
        # Task 18.1.19: Trust tier attributes
        self.policy_cache["trust_tier_access"] = {
            "policy_type": PolicyType.SENSITIVITY,
            "rules": [
                {
                    "condition": "resource.trust_tier == 'T3' and user.role not in ['admin', 'cro']",
                    "action": "deny",
                    "description": "Only admins can access T3 agents"
                }
            ]
        }
    
    async def evaluate_abac_policies(self, context: AccessContext) -> AccessResult:
        """
        Evaluate ABAC policies for access request
        Task 18.1.9: ABAC enforcement middleware
        """
        applicable_policies = []
        decisions = []
        reasons = []
        violations = []
        
        # Prepare evaluation context
        user_context = {
            "region": context.region,
            "sla_tier": context.sla_tier,
            "clearance_level": context.attributes.get("clearance_level", 1),
            "gdpr_trained": context.attributes.get("gdpr_trained", False),
            "role": context.role_ids[0] if context.role_ids else "user"
        }
        
        resource_context = {
            "region": context.region,  # Would be extracted from resource metadata
            "sensitivity": "medium",   # Would be extracted from resource metadata
            "type": "standard_feature", # Would be extracted from resource metadata
            "trust_tier": "T1"         # Would be extracted from resource metadata
        }
        
        # Evaluate each policy
        for policy_id, policy in self.policy_cache.items():
            try:
                policy_result = await self._evaluate_policy_rules(
                    policy, user_context, resource_context, context
                )
                
                if policy_result["applies"]:
                    applicable_policies.append(policy_id)
                    decisions.append(policy_result["decision"])
                    reasons.append(policy_result["reason"])
                    
                    if policy_result["decision"] == "deny":
                        violations.append(policy_result["reason"])
                        
            except Exception as e:
                logger.error(f"❌ Error evaluating ABAC policy {policy_id}: {e}")
                decisions.append("deny")
                reasons.append(f"Policy evaluation error: {str(e)}")
                violations.append(f"Policy {policy_id} evaluation failed")
        
        # Determine final decision (most restrictive wins)
        if "deny" in decisions:
            final_decision = AccessDecision.DENY
            final_reason = next(reason for decision, reason in zip(decisions, reasons) if decision == "deny")
        elif "require_approval" in decisions:
            final_decision = AccessDecision.REQUIRE_APPROVAL
            final_reason = next(reason for decision, reason in zip(decisions, reasons) if decision == "require_approval")
        elif "allow" in decisions:
            final_decision = AccessDecision.ALLOW
            final_reason = "ABAC policies allow access"
        else:
            final_decision = AccessDecision.DENY
            final_reason = "No applicable ABAC policies found"
        
        return AccessResult(
            decision=final_decision,
            reason=final_reason,
            applicable_policies=applicable_policies,
            violations=violations,
            context=context
        )
    
    async def _evaluate_policy_rules(
        self,
        policy: Dict[str, Any],
        user_context: Dict[str, Any],
        resource_context: Dict[str, Any],
        access_context: AccessContext
    ) -> Dict[str, Any]:
        """Evaluate individual policy rules"""
        
        # Simplified rule evaluation - in production would use proper policy engine
        for rule in policy.get("rules", []):
            condition = rule["condition"]
            
            # Simple condition evaluation (would use proper expression evaluator)
            if self._evaluate_condition(condition, user_context, resource_context):
                return {
                    "applies": True,
                    "decision": rule["action"],
                    "reason": rule["description"]
                }
        
        return {"applies": False, "decision": "allow", "reason": "No matching rules"}
    
    def _evaluate_condition(
        self,
        condition: str,
        user_context: Dict[str, Any],
        resource_context: Dict[str, Any]
    ) -> bool:
        """Evaluate policy condition"""
        try:
            # Replace context variables in condition
            eval_condition = condition
            
            # Replace user context
            for key, value in user_context.items():
                if isinstance(value, str):
                    eval_condition = eval_condition.replace(f"user.{key}", f"'{value}'")
                else:
                    eval_condition = eval_condition.replace(f"user.{key}", str(value))
            
            # Replace resource context
            for key, value in resource_context.items():
                if isinstance(value, str):
                    eval_condition = eval_condition.replace(f"resource.{key}", f"'{value}'")
                else:
                    eval_condition = eval_condition.replace(f"resource.{key}", str(value))
            
            # Evaluate condition (in production, use safe expression evaluator)
            return eval(eval_condition)
            
        except Exception as e:
            logger.error(f"❌ Condition evaluation error: {e}")
            return False

class SecurityMiddlewareManager:
    """
    Combined RBAC/ABAC Middleware Manager
    Task 18.1.15: Bind policies to orchestration
    """
    
    def __init__(self):
        self.rbac_middleware = RBACMiddleware()
        self.abac_middleware = ABACMiddleware()
        self.evidence_service = None  # Would be injected
    
    async def enforce_access_control(self, request: Request) -> AccessResult:
        """
        Comprehensive access control enforcement
        Combines RBAC and ABAC evaluation
        """
        try:
            # Extract authentication context
            auth_context = await self.rbac_middleware._extract_auth_context(request)
            
            if not auth_context:
                return AccessResult(
                    decision=AccessDecision.DENY,
                    reason="Authentication required",
                    applicable_policies=[],
                    violations=["No authentication context"],
                    context=AccessContext(
                        user_id=0, tenant_id=0, role_ids=[], permissions=set(),
                        attributes={}, request_path=str(request.url.path),
                        request_method=request.method
                    )
                )
            
            # Check RBAC permissions
            rbac_result = await self.rbac_middleware._check_rbac_permissions(auth_context)
            
            if rbac_result.decision == AccessDecision.DENY:
                await self._log_evidence(rbac_result, "rbac_denial")
                return rbac_result
            
            # Check ABAC policies
            abac_result = await self.abac_middleware.evaluate_abac_policies(auth_context)
            
            if abac_result.decision == AccessDecision.DENY:
                await self._log_evidence(abac_result, "abac_denial")
                return abac_result
            
            # Combine results
            final_result = AccessResult(
                decision=AccessDecision.ALLOW,
                reason="RBAC and ABAC policies allow access",
                applicable_policies=rbac_result.applicable_policies + abac_result.applicable_policies,
                violations=[],
                context=auth_context,
                evidence_id=str(uuid.uuid4())
            )
            
            await self._log_evidence(final_result, "access_granted")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Access control enforcement error: {e}")
            return AccessResult(
                decision=AccessDecision.DENY,
                reason=f"Access control error: {str(e)}",
                applicable_policies=[],
                violations=[f"System error: {str(e)}"],
                context=auth_context if 'auth_context' in locals() else AccessContext(
                    user_id=0, tenant_id=0, role_ids=[], permissions=set(),
                    attributes={}, request_path=str(request.url.path),
                    request_method=request.method
                )
            )
    
    async def _log_evidence(self, result: AccessResult, event_type: str) -> None:
        """
        Log access control evidence
        Task 18.1.21: Configure evidence logging
        """
        evidence = {
            "evidence_id": result.evidence_id or str(uuid.uuid4()),
            "event_type": event_type,
            "decision": result.decision.value,
            "reason": result.reason,
            "user_id": result.context.user_id,
            "tenant_id": result.context.tenant_id,
            "request_path": result.context.request_path,
            "request_method": result.context.request_method,
            "applicable_policies": result.applicable_policies,
            "violations": result.violations,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # In production, store in evidence service
        logger.info(f"🔐 Access control evidence: {json.dumps(evidence, indent=2)}")

# FastAPI dependency for access control
async def require_access_control(request: Request) -> AccessResult:
    """FastAPI dependency for access control enforcement"""
    middleware_manager = SecurityMiddlewareManager()
    return await middleware_manager.enforce_access_control(request)

# Decorator for endpoint protection
def require_permissions(*permissions: str):
    """Decorator to require specific permissions for endpoints"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                raise HTTPException(status_code=500, detail="Request context not found")
            
            # Check access control
            access_result = await require_access_control(request)
            
            if access_result.decision == AccessDecision.DENY:
                raise HTTPException(status_code=403, detail=access_result.reason)
            
            # Check specific permissions
            missing_permissions = []
            for permission in permissions:
                if permission not in access_result.context.permissions:
                    missing_permissions.append(permission)
            
            if missing_permissions:
                raise HTTPException(
                    status_code=403, 
                    detail=f"Missing required permissions: {', '.join(missing_permissions)}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator
