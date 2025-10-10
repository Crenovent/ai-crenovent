#!/usr/bin/env python3
"""
Automation Continuum Mapper - Task 1.2-T2
==========================================
Build continuum map (RBA ↔ RBIA ↔ AALA) - Clear evolution model with promotion/demotion

Features:
- Dynamic automation type classification
- Promotion/demotion rules based on trust scores
- Workflow evolution tracking
- Multi-tenant continuum management
- Industry-specific promotion thresholds
- Evidence-based decision making
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json
import asyncio
from dataclasses import dataclass

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# TASK 1.2-T2: AUTOMATION CONTINUUM MAPPER
# =====================================================

class AutomationType(str, Enum):
    RBA = "RBA"      # Rule-Based Automation
    RBIA = "RBIA"    # Rule-Based Intelligent Automation  
    AALA = "AALA"    # AI Agent-Led Automation

class PromotionDirection(str, Enum):
    PROMOTE = "promote"      # RBA → RBIA → AALA
    DEMOTE = "demote"        # AALA → RBIA → RBA
    MAINTAIN = "maintain"    # Stay at current level

class TrustLevel(str, Enum):
    HIGH = "high"           # >0.95
    MEDIUM = "medium"       # 0.8-0.95
    LOW = "low"            # <0.8

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

@dataclass
class ContinuumRule:
    """Task 1.2-T2: Continuum promotion/demotion rule"""
    rule_id: str
    from_type: AutomationType
    to_type: AutomationType
    direction: PromotionDirection
    trust_threshold: float
    execution_count_min: int
    success_rate_min: float
    industry_specific: bool = False
    conditions: Dict[str, Any] = None

class AutomationContinuumMapper:
    """
    Task 1.2-T2: Automation Continuum Mapping Implementation
    Manages workflow evolution across RBA → RBIA → AALA continuum
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize continuum rules (Task 1.2-T2)
        self.continuum_rules = self._initialize_continuum_rules()
        
        # Trust score thresholds per automation type
        self.trust_thresholds = {
            AutomationType.RBA: 1.0,      # Deterministic = perfect trust
            AutomationType.RBIA: 0.85,    # ML-augmented = high trust required
            AutomationType.AALA: 0.75     # Agent-led = variable trust
        }
        
        # Industry-specific promotion thresholds
        self.industry_thresholds = {
            IndustryCode.BANKING: {
                "rba_to_rbia": 0.98,      # Higher threshold for banking
                "rbia_to_aala": 0.92,
                "min_executions": 1000
            },
            IndustryCode.INSURANCE: {
                "rba_to_rbia": 0.97,
                "rbia_to_aala": 0.90,
                "min_executions": 500
            },
            IndustryCode.SAAS: {
                "rba_to_rbia": 0.90,      # Lower threshold for SaaS
                "rbia_to_aala": 0.85,
                "min_executions": 100
            },
            IndustryCode.HEALTHCARE: {
                "rba_to_rbia": 0.99,      # Highest threshold for healthcare
                "rbia_to_aala": 0.95,
                "min_executions": 2000
            },
            IndustryCode.ECOMMERCE: {
                "rba_to_rbia": 0.92,
                "rbia_to_aala": 0.87,
                "min_executions": 200
            },
            IndustryCode.FINTECH: {
                "rba_to_rbia": 0.95,
                "rbia_to_aala": 0.88,
                "min_executions": 300
            }
        }
    
    def _initialize_continuum_rules(self) -> List[ContinuumRule]:
        """Initialize promotion/demotion rules (Task 1.2-T2)"""
        
        rules = []
        
        # RBA → RBIA PROMOTION RULES
        rules.append(ContinuumRule(
            rule_id="rba_to_rbia_standard",
            from_type=AutomationType.RBA,
            to_type=AutomationType.RBIA,
            direction=PromotionDirection.PROMOTE,
            trust_threshold=0.95,
            execution_count_min=100,
            success_rate_min=0.98,
            conditions={
                "complexity_score": ">0.7",
                "ml_benefit_potential": ">0.6",
                "data_availability": "sufficient"
            }
        ))
        
        # RBIA → AALA PROMOTION RULES
        rules.append(ContinuumRule(
            rule_id="rbia_to_aala_standard",
            from_type=AutomationType.RBIA,
            to_type=AutomationType.AALA,
            direction=PromotionDirection.PROMOTE,
            trust_threshold=0.90,
            execution_count_min=50,
            success_rate_min=0.95,
            conditions={
                "reasoning_complexity": ">0.8",
                "multi_step_required": True,
                "context_awareness_needed": True
            }
        ))
        
        # AALA → RBIA DEMOTION RULES
        rules.append(ContinuumRule(
            rule_id="aala_to_rbia_fallback",
            from_type=AutomationType.AALA,
            to_type=AutomationType.RBIA,
            direction=PromotionDirection.DEMOTE,
            trust_threshold=0.70,  # Below threshold triggers demotion
            execution_count_min=10,
            success_rate_min=0.80,
            conditions={
                "consecutive_failures": ">3",
                "trust_degradation": True,
                "safety_concern": True
            }
        ))
        
        # RBIA → RBA DEMOTION RULES
        rules.append(ContinuumRule(
            rule_id="rbia_to_rba_fallback",
            from_type=AutomationType.RBIA,
            to_type=AutomationType.RBA,
            direction=PromotionDirection.DEMOTE,
            trust_threshold=0.75,
            execution_count_min=20,
            success_rate_min=0.85,
            conditions={
                "ml_model_degradation": True,
                "data_quality_issues": True,
                "deterministic_alternative": True
            }
        ))
        
        # BANKING-SPECIFIC RULES (Higher thresholds)
        rules.append(ContinuumRule(
            rule_id="rba_to_rbia_banking",
            from_type=AutomationType.RBA,
            to_type=AutomationType.RBIA,
            direction=PromotionDirection.PROMOTE,
            trust_threshold=0.98,
            execution_count_min=1000,
            success_rate_min=0.995,
            industry_specific=True,
            conditions={
                "industry": "Banking",
                "regulatory_approval": True,
                "audit_trail_complete": True
            }
        ))
        
        return rules
    
    async def classify_workflow(
        self,
        workflow_id: str,
        tenant_id: int,
        current_type: Optional[AutomationType] = None
    ) -> Dict[str, Any]:
        """
        Classify workflow into appropriate automation type (Task 1.2-T2)
        """
        try:
            # Get workflow details and execution history
            workflow_data = await self._get_workflow_data(workflow_id, tenant_id)
            execution_stats = await self._get_execution_statistics(workflow_id, tenant_id)
            industry_code = await self._get_tenant_industry(tenant_id)
            
            # Analyze workflow characteristics
            characteristics = await self._analyze_workflow_characteristics(workflow_data)
            
            # Determine recommended automation type
            recommended_type = await self._determine_automation_type(
                characteristics, execution_stats, industry_code
            )
            
            # Check if promotion/demotion is recommended
            promotion_analysis = None
            if current_type and current_type != recommended_type:
                promotion_analysis = await self._analyze_promotion_opportunity(
                    workflow_id, tenant_id, current_type, recommended_type, 
                    execution_stats, industry_code
                )
            
            return {
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "current_type": current_type,
                "recommended_type": recommended_type,
                "confidence_score": characteristics.get("classification_confidence", 0.0),
                "characteristics": characteristics,
                "execution_stats": execution_stats,
                "promotion_analysis": promotion_analysis,
                "classification_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying workflow {workflow_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    
    async def evaluate_promotion_eligibility(
        self,
        workflow_id: str,
        tenant_id: int,
        target_type: AutomationType
    ) -> Dict[str, Any]:
        """
        Evaluate if workflow is eligible for promotion (Task 1.2-T2)
        """
        try:
            # Get current workflow type
            current_type = await self._get_current_automation_type(workflow_id, tenant_id)
            
            # Get execution statistics
            execution_stats = await self._get_execution_statistics(workflow_id, tenant_id)
            
            # Get industry-specific thresholds
            industry_code = await self._get_tenant_industry(tenant_id)
            thresholds = self.industry_thresholds.get(industry_code, self.industry_thresholds[IndustryCode.SAAS])
            
            # Find applicable promotion rule
            applicable_rule = self._find_applicable_rule(current_type, target_type, industry_code)
            
            if not applicable_rule:
                return {
                    "eligible": False,
                    "reason": f"No promotion rule found for {current_type} → {target_type}",
                    "workflow_id": workflow_id
                }
            
            # Evaluate eligibility criteria
            eligibility_checks = await self._evaluate_eligibility_criteria(
                applicable_rule, execution_stats, thresholds
            )
            
            # Calculate overall eligibility
            eligible = all(check["passed"] for check in eligibility_checks.values())
            
            return {
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "current_type": current_type,
                "target_type": target_type,
                "eligible": eligible,
                "rule_applied": applicable_rule.rule_id,
                "eligibility_checks": eligibility_checks,
                "industry_code": industry_code,
                "evaluation_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating promotion eligibility: {e}")
            raise HTTPException(status_code=500, detail=f"Promotion evaluation failed: {e}")
    
    async def execute_promotion(
        self,
        workflow_id: str,
        tenant_id: int,
        target_type: AutomationType,
        user_id: int,
        reason: str
    ) -> Dict[str, Any]:
        """
        Execute workflow promotion/demotion (Task 1.2-T2)
        """
        try:
            # Verify eligibility first
            eligibility = await self.evaluate_promotion_eligibility(workflow_id, tenant_id, target_type)
            
            if not eligibility["eligible"]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Workflow not eligible for promotion: {eligibility['eligibility_checks']}"
                )
            
            current_type = eligibility["current_type"]
            
            # Create promotion record
            promotion_id = str(uuid.uuid4())
            
            # Update workflow type
            await self._update_workflow_type(workflow_id, tenant_id, target_type, promotion_id)
            
            # Log promotion in continuum history
            await self._log_continuum_change(
                promotion_id, workflow_id, tenant_id, user_id,
                current_type, target_type, reason, eligibility
            )
            
            # Generate evidence pack for promotion
            evidence_pack_id = await self._generate_promotion_evidence_pack(
                promotion_id, workflow_id, tenant_id, current_type, target_type, eligibility
            )
            
            # Update trust scoring
            await self._update_trust_scoring_for_promotion(workflow_id, tenant_id, target_type)
            
            return {
                "promotion_id": promotion_id,
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "from_type": current_type,
                "to_type": target_type,
                "status": "completed",
                "evidence_pack_id": evidence_pack_id,
                "promoted_by": user_id,
                "promotion_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error executing promotion: {e}")
            raise HTTPException(status_code=500, detail=f"Promotion execution failed: {e}")
    
    async def get_continuum_map(self, tenant_id: int) -> Dict[str, Any]:
        """
        Get complete continuum map for tenant (Task 1.2-T2)
        """
        try:
            # Get all workflows for tenant
            workflows = await self._get_tenant_workflows(tenant_id)
            
            # Classify each workflow
            continuum_map = {
                AutomationType.RBA: [],
                AutomationType.RBIA: [],
                AutomationType.AALA: []
            }
            
            promotion_opportunities = []
            
            for workflow in workflows:
                classification = await self.classify_workflow(
                    workflow["workflow_id"], tenant_id, workflow.get("current_type")
                )
                
                current_type = classification.get("current_type", classification["recommended_type"])
                continuum_map[current_type].append(classification)
                
                # Check for promotion opportunities
                if classification.get("promotion_analysis"):
                    promotion_opportunities.append(classification["promotion_analysis"])
            
            # Get industry-specific statistics
            industry_code = await self._get_tenant_industry(tenant_id)
            
            return {
                "tenant_id": tenant_id,
                "industry_code": industry_code,
                "continuum_map": continuum_map,
                "promotion_opportunities": promotion_opportunities,
                "statistics": {
                    "total_workflows": len(workflows),
                    "rba_count": len(continuum_map[AutomationType.RBA]),
                    "rbia_count": len(continuum_map[AutomationType.RBIA]),
                    "aala_count": len(continuum_map[AutomationType.AALA]),
                    "promotion_ready": len(promotion_opportunities)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating continuum map: {e}")
            raise HTTPException(status_code=500, detail=f"Continuum map generation failed: {e}")
    
    async def _get_workflow_data(self, workflow_id: str, tenant_id: int) -> Dict[str, Any]:
        """Get workflow definition and metadata"""
        query = """
            SELECT workflow_id, workflow_name, workflow_type, workflow_definition, 
                   industry_overlay, created_at, trust_score
            FROM dsl_workflows 
            WHERE workflow_id = $1 AND tenant_id = $2
        """
        
        async with self.pool_manager.get_connection() as conn:
            result = await conn.fetchrow(query, workflow_id, tenant_id)
            
            if not result:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            return dict(result)
    
    async def _get_execution_statistics(self, workflow_id: str, tenant_id: int) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        query = """
            SELECT 
                COUNT(*) as total_executions,
                COUNT(*) FILTER (WHERE status = 'completed') as successful_executions,
                COUNT(*) FILTER (WHERE status = 'failed') as failed_executions,
                AVG(execution_time_ms) as avg_execution_time,
                AVG(trust_score) as avg_trust_score,
                MIN(created_at) as first_execution,
                MAX(created_at) as last_execution
            FROM dsl_execution_traces 
            WHERE workflow_id = $1 AND tenant_id = $2
            AND created_at >= NOW() - INTERVAL '30 days'
        """
        
        async with self.pool_manager.get_connection() as conn:
            result = await conn.fetchrow(query, workflow_id, tenant_id)
            
            stats = dict(result) if result else {}
            
            # Calculate success rate
            total = stats.get("total_executions", 0)
            successful = stats.get("successful_executions", 0)
            stats["success_rate"] = (successful / total) if total > 0 else 0.0
            
            return stats
    
    async def _analyze_workflow_characteristics(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow characteristics for classification"""
        workflow_definition = workflow_data.get("workflow_definition", {})
        
        characteristics = {
            "has_ml_components": False,
            "has_agent_calls": False,
            "complexity_score": 0.0,
            "decision_points": 0,
            "external_integrations": 0,
            "classification_confidence": 0.0
        }
        
        # Analyze workflow steps
        steps = workflow_definition.get("steps", [])
        
        for step in steps:
            step_type = step.get("type", "")
            
            if step_type == "ml_decision":
                characteristics["has_ml_components"] = True
            elif step_type == "agent_call":
                characteristics["has_agent_calls"] = True
            elif step_type == "decision":
                characteristics["decision_points"] += 1
            elif step_type == "query" and step.get("params", {}).get("external", False):
                characteristics["external_integrations"] += 1
        
        # Calculate complexity score
        complexity_factors = [
            len(steps) / 10.0,  # Step count factor
            characteristics["decision_points"] / 5.0,  # Decision complexity
            characteristics["external_integrations"] / 3.0,  # Integration complexity
        ]
        
        characteristics["complexity_score"] = min(sum(complexity_factors) / len(complexity_factors), 1.0)
        
        # Calculate classification confidence
        if characteristics["has_agent_calls"]:
            characteristics["classification_confidence"] = 0.95  # High confidence for AALA
        elif characteristics["has_ml_components"]:
            characteristics["classification_confidence"] = 0.90  # High confidence for RBIA
        else:
            characteristics["classification_confidence"] = 0.85  # Good confidence for RBA
        
        return characteristics
    
    async def _determine_automation_type(
        self, 
        characteristics: Dict[str, Any], 
        execution_stats: Dict[str, Any],
        industry_code: IndustryCode
    ) -> AutomationType:
        """Determine appropriate automation type based on characteristics"""
        
        # Agent-led automation indicators
        if characteristics.get("has_agent_calls", False):
            return AutomationType.AALA
        
        # ML-augmented automation indicators
        if characteristics.get("has_ml_components", False):
            return AutomationType.RBIA
        
        # High complexity might benefit from ML
        if (characteristics.get("complexity_score", 0) > 0.7 and 
            execution_stats.get("total_executions", 0) > 100):
            return AutomationType.RBIA
        
        # Default to rule-based automation
        return AutomationType.RBA
    
    def _find_applicable_rule(
        self, 
        current_type: AutomationType, 
        target_type: AutomationType,
        industry_code: IndustryCode
    ) -> Optional[ContinuumRule]:
        """Find applicable promotion/demotion rule"""
        
        for rule in self.continuum_rules:
            if (rule.from_type == current_type and 
                rule.to_type == target_type):
                
                # Check industry-specific rules first
                if rule.industry_specific:
                    conditions = rule.conditions or {}
                    if conditions.get("industry") == industry_code.value:
                        return rule
                elif not rule.industry_specific:
                    return rule
        
        return None
    
    async def _evaluate_eligibility_criteria(
        self,
        rule: ContinuumRule,
        execution_stats: Dict[str, Any],
        thresholds: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate all eligibility criteria for promotion"""
        
        checks = {}
        
        # Trust threshold check
        avg_trust_score = execution_stats.get("avg_trust_score", 0.0)
        checks["trust_threshold"] = {
            "required": rule.trust_threshold,
            "actual": avg_trust_score,
            "passed": avg_trust_score >= rule.trust_threshold
        }
        
        # Execution count check
        total_executions = execution_stats.get("total_executions", 0)
        checks["execution_count"] = {
            "required": rule.execution_count_min,
            "actual": total_executions,
            "passed": total_executions >= rule.execution_count_min
        }
        
        # Success rate check
        success_rate = execution_stats.get("success_rate", 0.0)
        checks["success_rate"] = {
            "required": rule.success_rate_min,
            "actual": success_rate,
            "passed": success_rate >= rule.success_rate_min
        }
        
        return checks

# Initialize service
automation_continuum_mapper = None

def get_automation_continuum_mapper():
    """Get automation continuum mapper instance"""
    global automation_continuum_mapper
    if automation_continuum_mapper is None:
        pool_manager = get_pool_manager()
        automation_continuum_mapper = AutomationContinuumMapper(pool_manager)
    return automation_continuum_mapper

# =====================================================
# API ENDPOINTS
# =====================================================

@router.get("/continuum/map/{tenant_id}")
async def get_continuum_map_endpoint(
    tenant_id: int,
    mapper: AutomationContinuumMapper = Depends(get_automation_continuum_mapper)
):
    """
    Get complete automation continuum map for tenant (Task 1.2-T2)
    """
    return await mapper.get_continuum_map(tenant_id)

@router.post("/continuum/classify")
async def classify_workflow_endpoint(
    workflow_id: str = Query(..., description="Workflow ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    current_type: Optional[AutomationType] = Query(None, description="Current automation type"),
    mapper: AutomationContinuumMapper = Depends(get_automation_continuum_mapper)
):
    """
    Classify workflow into appropriate automation type (Task 1.2-T2)
    """
    return await mapper.classify_workflow(workflow_id, tenant_id, current_type)

@router.get("/continuum/promotion/eligibility")
async def evaluate_promotion_eligibility_endpoint(
    workflow_id: str = Query(..., description="Workflow ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    target_type: AutomationType = Query(..., description="Target automation type"),
    mapper: AutomationContinuumMapper = Depends(get_automation_continuum_mapper)
):
    """
    Evaluate workflow promotion eligibility (Task 1.2-T2)
    """
    return await mapper.evaluate_promotion_eligibility(workflow_id, tenant_id, target_type)

@router.post("/continuum/promotion/execute")
async def execute_promotion_endpoint(
    workflow_id: str = Query(..., description="Workflow ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    target_type: AutomationType = Query(..., description="Target automation type"),
    user_id: int = Query(..., description="User executing promotion"),
    reason: str = Query(..., description="Reason for promotion"),
    mapper: AutomationContinuumMapper = Depends(get_automation_continuum_mapper)
):
    """
    Execute workflow promotion/demotion (Task 1.2-T2)
    """
    return await mapper.execute_promotion(workflow_id, tenant_id, target_type, user_id, reason)
