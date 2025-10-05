"""
Governance Hooks Integration Service
Chapter 8.5 Tasks: Integration with existing RBA components

Integrates governance hooks with:
- DSL Compiler (static governance validation)
- Workflow Runtime (dynamic governance enforcement)
- Policy Engine (policy-driven governance)
- Trust Scoring (trust-based governance decisions)
- Evidence Packs (governance evidence generation)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

from .governance_hooks_middleware import (
    GovernanceHooksMiddleware, GovernanceContext, GovernancePlane,
    GovernanceHookType, GovernanceDecision, GovernanceHookResult
)
from .saas_governance_hooks import SaaSGovernanceHooks, SaaSTier

logger = logging.getLogger(__name__)

class GovernanceHooksIntegrationService:
    """
    Integration service for governance hooks with existing RBA components
    Tasks 8.5.2-8.5.42: Comprehensive governance integration
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize governance middleware
        self.base_middleware = GovernanceHooksMiddleware(pool_manager)
        self.saas_middleware = SaaSGovernanceHooks(pool_manager)
        
        # Integration configurations
        self.integration_config = {
            "dsl_compiler_integration": True,
            "workflow_runtime_integration": True,
            "policy_engine_integration": True,
            "trust_scoring_integration": True,
            "evidence_pack_integration": True,
            "real_time_monitoring": True
        }
        
        # Governance enforcement points
        self.enforcement_points = {
            "workflow_compilation": {
                "plane": GovernancePlane.CONTROL,
                "hooks": [GovernanceHookType.POLICY, GovernanceHookType.SOD]
            },
            "workflow_execution": {
                "plane": GovernancePlane.EXECUTION,
                "hooks": [GovernanceHookType.TRUST, GovernanceHookType.CONSENT, GovernanceHookType.ANOMALY]
            },
            "data_access": {
                "plane": GovernancePlane.DATA,
                "hooks": [GovernanceHookType.CONSENT, GovernanceHookType.RESIDENCY, GovernanceHookType.LINEAGE]
            },
            "evidence_generation": {
                "plane": GovernancePlane.GOVERNANCE,
                "hooks": [GovernanceHookType.POLICY, GovernanceHookType.FINOPS, GovernanceHookType.LINEAGE]
            }
        }
    
    async def enforce_workflow_compilation_governance(self, 
                                                    tenant_id: int,
                                                    user_id: str,
                                                    user_role: str,
                                                    workflow_definition: Dict[str, Any],
                                                    compilation_context: Dict[str, Any]) -> GovernanceHookResult:
        """
        Enforce governance during DSL workflow compilation
        Task 8.5.2: Control Plane governance at orchestration level
        """
        try:
            self.logger.info(f"🔒 Enforcing compilation governance for workflow: {workflow_definition.get('workflow_id')}")
            
            # Create governance context
            context = GovernanceContext(
                tenant_id=tenant_id,
                user_id=user_id,
                user_role=user_role,
                workflow_id=workflow_definition.get("workflow_id", "unknown"),
                execution_id=f"compile_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                plane=GovernancePlane.CONTROL,
                hook_type=GovernanceHookType.POLICY,
                request_data={
                    "workflow_definition": workflow_definition,
                    "compilation_context": compilation_context,
                    "operation_type": "workflow_compilation"
                },
                metadata={"enforcement_point": "workflow_compilation"},
                timestamp=datetime.now()
            )
            
            # Check if SaaS tenant
            saas_tier = compilation_context.get("saas_tier")
            if saas_tier:
                try:
                    tier_enum = SaaSTier(saas_tier)
                    return await self.saas_middleware.enforce_saas_tier_governance(context, tier_enum)
                except ValueError:
                    self.logger.warning(f"Invalid SaaS tier: {saas_tier}, using base governance")
            
            # Use base governance middleware
            return await self.base_middleware.enforce_governance(context)
            
        except Exception as e:
            self.logger.error(f"❌ Compilation governance failed: {e}")
            # Fail closed - deny compilation on error
            return GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=GovernanceHookType.POLICY,
                context=context,
                violations=[f"Compilation governance error: {str(e)}"]
            )
    
    async def enforce_workflow_execution_governance(self,
                                                  tenant_id: int,
                                                  user_id: str,
                                                  user_role: str,
                                                  workflow_id: str,
                                                  execution_id: str,
                                                  execution_context: Dict[str, Any]) -> List[GovernanceHookResult]:
        """
        Enforce governance during workflow execution
        Task 8.5.3: Execution Plane governance with trust engine integration
        """
        results = []
        
        try:
            self.logger.info(f"🔒 Enforcing execution governance for workflow: {workflow_id}")
            
            # Get enforcement hooks for execution plane
            execution_hooks = self.enforcement_points["workflow_execution"]["hooks"]
            
            for hook_type in execution_hooks:
                context = GovernanceContext(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    user_role=user_role,
                    workflow_id=workflow_id,
                    execution_id=execution_id,
                    plane=GovernancePlane.EXECUTION,
                    hook_type=hook_type,
                    request_data={
                        "execution_context": execution_context,
                        "operation_type": "workflow_execution",
                        "trust_score": execution_context.get("trust_score", 1.0),
                        "consent_id": execution_context.get("consent_id"),
                        "api_calls_per_minute": execution_context.get("api_calls_per_minute", 0)
                    },
                    metadata={"enforcement_point": "workflow_execution"},
                    timestamp=datetime.now()
                )
                
                # Check if SaaS tenant
                saas_tier = execution_context.get("saas_tier")
                if saas_tier and hook_type == GovernanceHookType.TRUST:
                    try:
                        tier_enum = SaaSTier(saas_tier)
                        result = await self.saas_middleware.enforce_saas_tier_governance(context, tier_enum)
                    except ValueError:
                        result = await self.base_middleware.enforce_governance(context)
                else:
                    result = await self.base_middleware.enforce_governance(context)
                
                results.append(result)
                
                # Stop execution if any hook denies
                if result.decision == GovernanceDecision.DENY:
                    self.logger.warning(f"🚫 Execution denied by {hook_type.value} hook")
                    break
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Execution governance failed: {e}")
            # Return denial result
            error_context = GovernanceContext(
                tenant_id=tenant_id, user_id=user_id, user_role=user_role,
                workflow_id=workflow_id, execution_id=execution_id,
                plane=GovernancePlane.EXECUTION, hook_type=GovernanceHookType.POLICY,
                request_data={}, metadata={}, timestamp=datetime.now()
            )
            
            return [GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=GovernanceHookType.POLICY,
                context=error_context,
                violations=[f"Execution governance error: {str(e)}"]
            )]
    
    async def enforce_data_access_governance(self,
                                           tenant_id: int,
                                           user_id: str,
                                           user_role: str,
                                           workflow_id: str,
                                           execution_id: str,
                                           data_access_request: Dict[str, Any]) -> GovernanceHookResult:
        """
        Enforce governance for data access operations
        Task 8.5.4: Data Plane governance with ABAC + masking
        """
        try:
            self.logger.info(f"🔒 Enforcing data access governance for workflow: {workflow_id}")
            
            # Create governance context for data access
            context = GovernanceContext(
                tenant_id=tenant_id,
                user_id=user_id,
                user_role=user_role,
                workflow_id=workflow_id,
                execution_id=execution_id,
                plane=GovernancePlane.DATA,
                hook_type=GovernanceHookType.CONSENT,
                request_data={
                    "data_access_request": data_access_request,
                    "operation_type": "data_access",
                    "data_classification": data_access_request.get("data_classification", "internal"),
                    "target_tenant_id": data_access_request.get("target_tenant_id"),
                    "consent_id": data_access_request.get("consent_id"),
                    "consent_scope": data_access_request.get("consent_scope", []),
                    "tenant_region": data_access_request.get("tenant_region", "GLOBAL"),
                    "target_region": data_access_request.get("target_region", "GLOBAL")
                },
                metadata={"enforcement_point": "data_access"},
                timestamp=datetime.now()
            )
            
            # Check for SaaS multi-tenancy isolation
            if data_access_request.get("saas_tenant", False):
                return await self.saas_middleware.enforce_saas_multi_tenancy_isolation(context)
            
            # Use base governance middleware
            return await self.base_middleware.enforce_governance(context)
            
        except Exception as e:
            self.logger.error(f"❌ Data access governance failed: {e}")
            return GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=GovernanceHookType.CONSENT,
                context=context,
                violations=[f"Data access governance error: {str(e)}"]
            )
    
    async def enforce_evidence_generation_governance(self,
                                                   tenant_id: int,
                                                   user_id: str,
                                                   user_role: str,
                                                   workflow_id: str,
                                                   execution_id: str,
                                                   evidence_request: Dict[str, Any]) -> GovernanceHookResult:
        """
        Enforce governance for evidence pack generation
        Task 8.5.5: Governance Plane governance with OPA hooks
        """
        try:
            self.logger.info(f"🔒 Enforcing evidence generation governance for workflow: {workflow_id}")
            
            context = GovernanceContext(
                tenant_id=tenant_id,
                user_id=user_id,
                user_role=user_role,
                workflow_id=workflow_id,
                execution_id=execution_id,
                plane=GovernancePlane.GOVERNANCE,
                hook_type=GovernanceHookType.LINEAGE,
                request_data={
                    "evidence_request": evidence_request,
                    "operation_type": "evidence_generation",
                    "evidence_type": evidence_request.get("evidence_type", "workflow_execution"),
                    "compliance_framework": evidence_request.get("compliance_framework", "SaaS_Governance"),
                    "retention_years": evidence_request.get("retention_years", 7),
                    "cost_center": evidence_request.get("cost_center", "governance"),
                    "estimated_cost_usd": evidence_request.get("estimated_cost_usd", 0.01)
                },
                metadata={"enforcement_point": "evidence_generation"},
                timestamp=datetime.now()
            )
            
            return await self.base_middleware.enforce_governance(context)
            
        except Exception as e:
            self.logger.error(f"❌ Evidence generation governance failed: {e}")
            return GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=GovernanceHookType.LINEAGE,
                context=context,
                violations=[f"Evidence generation governance error: {str(e)}"]
            )
    
    async def integrate_with_policy_engine(self, 
                                         policy_engine_result: Dict[str, Any],
                                         governance_context: GovernanceContext) -> GovernanceHookResult:
        """
        Integrate governance hooks with existing policy engine
        Task 8.5.11: Trust score enforcement integration
        """
        try:
            # Extract policy engine results
            policy_allowed = policy_engine_result.get("allowed", True)
            policy_violations = policy_engine_result.get("violations", [])
            trust_score = policy_engine_result.get("trust_score", 1.0)
            
            # Create governance-enhanced result
            if not policy_allowed:
                decision = GovernanceDecision.DENY
            elif trust_score < 0.7:  # Low trust threshold
                decision = GovernanceDecision.OVERRIDE_REQUIRED
            else:
                decision = GovernanceDecision.ALLOW
            
            result = GovernanceHookResult(
                decision=decision,
                hook_type=governance_context.hook_type,
                context=governance_context,
                violations=policy_violations,
                override_required=(decision == GovernanceDecision.OVERRIDE_REQUIRED)
            )
            
            result.trust_score_impact = trust_score - 1.0  # Impact relative to perfect score
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Policy engine integration failed: {e}")
            return GovernanceHookResult(
                decision=GovernanceDecision.DENY,
                hook_type=governance_context.hook_type,
                context=governance_context,
                violations=[f"Policy engine integration error: {str(e)}"]
            )
    
    async def get_governance_health_status(self, tenant_id: int) -> Dict[str, Any]:
        """
        Get overall governance health status for a tenant
        Task 8.5.39: Resilience dashboards scoped to governance
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            # Get governance metrics from database
            governance_stats = await self._get_governance_statistics(tenant_id, start_time, end_time)
            
            # Calculate governance health score
            total_events = governance_stats.get("total_events", 1)
            violation_events = governance_stats.get("violation_events", 0)
            denied_events = governance_stats.get("denied_events", 0)
            
            # Health score calculation (0-100)
            violation_rate = (violation_events / total_events) * 100
            denial_rate = (denied_events / total_events) * 100
            
            health_score = max(0, 100 - (violation_rate * 2) - (denial_rate * 5))
            
            # Determine health status
            if health_score >= 90:
                health_status = "excellent"
            elif health_score >= 75:
                health_status = "good"
            elif health_score >= 50:
                health_status = "fair"
            else:
                health_status = "poor"
            
            return {
                "tenant_id": tenant_id,
                "health_score": round(health_score, 2),
                "health_status": health_status,
                "governance_statistics": governance_stats,
                "recommendations": await self._get_governance_recommendations(tenant_id, health_score),
                "last_updated": end_time.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Governance health status failed: {e}")
            return {
                "tenant_id": tenant_id,
                "health_score": 0,
                "health_status": "unknown",
                "error": str(e)
            }
    
    async def _get_governance_statistics(self, tenant_id: int, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Get governance statistics from database"""
        try:
            if not self.pool_manager or not hasattr(self.pool_manager, 'postgres_pool'):
                return {"total_events": 0, "violation_events": 0, "denied_events": 0}
            
            async with self.pool_manager.postgres_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(CASE WHEN violations_count > 0 THEN 1 END) as violation_events,
                        COUNT(CASE WHEN decision = 'deny' THEN 1 END) as denied_events,
                        COUNT(CASE WHEN decision = 'override_required' THEN 1 END) as override_events,
                        AVG(execution_time_ms) as avg_execution_time,
                        COUNT(DISTINCT workflow_id) as unique_workflows
                    FROM governance_event_logs 
                    WHERE tenant_id = $1 AND timestamp BETWEEN $2 AND $3
                """, tenant_id, start_time, end_time)
                
                return {
                    "total_events": result["total_events"] or 0,
                    "violation_events": result["violation_events"] or 0,
                    "denied_events": result["denied_events"] or 0,
                    "override_events": result["override_events"] or 0,
                    "avg_execution_time_ms": float(result["avg_execution_time"] or 0),
                    "unique_workflows": result["unique_workflows"] or 0
                }
        except Exception as e:
            self.logger.error(f"Failed to get governance statistics: {e}")
            return {"total_events": 0, "violation_events": 0, "denied_events": 0}
    
    async def _get_governance_recommendations(self, tenant_id: int, health_score: float) -> List[Dict[str, Any]]:
        """Get governance improvement recommendations"""
        recommendations = []
        
        if health_score < 50:
            recommendations.extend([
                {
                    "type": "critical",
                    "title": "Review Governance Policies",
                    "description": "High violation rate indicates governance policies may need adjustment",
                    "action": "Conduct governance policy review and optimization"
                },
                {
                    "type": "critical",
                    "title": "Implement Proactive Monitoring",
                    "description": "Enable real-time governance monitoring and alerting",
                    "action": "Configure governance anomaly detection and alerts"
                }
            ])
        
        elif health_score < 75:
            recommendations.extend([
                {
                    "type": "warning",
                    "title": "Optimize Governance Rules",
                    "description": "Moderate violation rate suggests room for improvement",
                    "action": "Fine-tune governance rules and thresholds"
                },
                {
                    "type": "info",
                    "title": "Training and Documentation",
                    "description": "Provide governance training to reduce violations",
                    "action": "Create governance best practices documentation"
                }
            ])
        
        else:
            recommendations.append({
                "type": "success",
                "title": "Maintain Current Governance",
                "description": "Governance health is good, continue current practices",
                "action": "Regular governance health monitoring"
            })
        
        return recommendations

# Convenience function for integrated governance enforcement
async def enforce_integrated_governance(integration_service: GovernanceHooksIntegrationService,
                                      enforcement_point: str,
                                      tenant_id: int,
                                      user_id: str,
                                      user_role: str,
                                      workflow_id: str,
                                      execution_id: str,
                                      request_data: Dict[str, Any]) -> GovernanceHookResult:
    """Enforce governance at specified enforcement point"""
    
    if enforcement_point == "workflow_compilation":
        return await integration_service.enforce_workflow_compilation_governance(
            tenant_id, user_id, user_role, request_data, {}
        )
    elif enforcement_point == "workflow_execution":
        results = await integration_service.enforce_workflow_execution_governance(
            tenant_id, user_id, user_role, workflow_id, execution_id, request_data
        )
        # Return the most restrictive result
        for result in results:
            if result.decision == GovernanceDecision.DENY:
                return result
        return results[0] if results else None
    elif enforcement_point == "data_access":
        return await integration_service.enforce_data_access_governance(
            tenant_id, user_id, user_role, workflow_id, execution_id, request_data
        )
    elif enforcement_point == "evidence_generation":
        return await integration_service.enforce_evidence_generation_governance(
            tenant_id, user_id, user_role, workflow_id, execution_id, request_data
        )
    else:
        raise ValueError(f"Unknown enforcement point: {enforcement_point}")
