#!/usr/bin/env python3
"""
Workflow Classifier - Task 1.3-T1
==================================
Define classification rules for RBA vs RBIA vs AALA - Clear boundaries across automation types

Features:
- Dynamic workflow classification based on characteristics
- Industry-specific classification rules
- Multi-tenant classification with tenant isolation
- Trust-based classification adjustments
- Evidence-based classification decisions
- Real-time classification updates
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
import json
import re
from dataclasses import dataclass

from src.database.connection_pool import get_pool_manager

logger = logging.getLogger(__name__)
router = APIRouter()

# =====================================================
# TASK 1.3-T1: WORKFLOW CLASSIFIER
# =====================================================

class AutomationType(str, Enum):
    RBA = "RBA"      # Rule-Based Automation
    RBIA = "RBIA"    # Rule-Based Intelligent Automation  
    AALA = "AALA"    # AI Agent-Led Automation

class ClassificationConfidence(str, Enum):
    HIGH = "high"         # >90% confidence
    MEDIUM = "medium"     # 70-90% confidence
    LOW = "low"          # <70% confidence

class WorkflowComplexity(str, Enum):
    SIMPLE = "simple"         # Linear, few decisions
    MODERATE = "moderate"     # Some branching, moderate decisions
    COMPLEX = "complex"       # Multiple branches, many decisions
    VERY_COMPLEX = "very_complex"  # Highly branched, complex logic

class IndustryCode(str, Enum):
    SAAS = "SaaS"
    BANKING = "Banking"
    INSURANCE = "Insurance"
    HEALTHCARE = "Healthcare"
    ECOMMERCE = "E-commerce"
    FINTECH = "FinTech"

@dataclass
class ClassificationRule:
    """Task 1.3-T1: Classification rule definition"""
    rule_id: str
    automation_type: AutomationType
    conditions: Dict[str, Any]
    weight: float
    industry_specific: bool = False
    description: str = ""

class WorkflowClassifier:
    """
    Task 1.3-T1: Workflow Classification Implementation
    Classifies workflows into RBA, RBIA, or AALA based on characteristics
    """
    
    def __init__(self, pool_manager):
        self.pool_manager = pool_manager
        self.logger = logging.getLogger(__name__)
        
        # Initialize classification rules (Task 1.3-T1)
        self.classification_rules = self._initialize_classification_rules()
        
        # RBA indicators - deterministic, rule-based patterns
        self.rba_indicators = {
            "step_types": {"query", "decision", "notify", "governance"},
            "keywords": {"if", "then", "else", "when", "condition", "rule"},
            "patterns": {
                "simple_conditions": r"\b(if|when)\s+\w+\s*(==|!=|>|<|>=|<=)\s*\w+",
                "basic_logic": r"\b(and|or|not)\b",
                "field_validation": r"validate|check|verify"
            }
        }
        
        # RBIA indicators - ML-augmented, intelligent patterns
        self.rbia_indicators = {
            "step_types": {"ml_decision", "predict", "classify", "score"},
            "keywords": {"model", "prediction", "probability", "confidence", "score", "classify"},
            "patterns": {
                "ml_operations": r"\b(predict|classify|score|model)\b",
                "probability": r"\b(probability|confidence|likelihood)\b",
                "threshold": r"\bthreshold\s*[><=]\s*\d+\.?\d*"
            }
        }
        
        # AALA indicators - agent-led, reasoning patterns
        self.aala_indicators = {
            "step_types": {"agent_call", "reasoning", "planning", "multi_step"},
            "keywords": {"agent", "reasoning", "planning", "context", "memory", "goal"},
            "patterns": {
                "agent_operations": r"\b(agent|reasoning|planning|goal)\b",
                "context_usage": r"\b(context|memory|history)\b",
                "multi_step": r"\b(multi.step|sequential|chain)\b"
            }
        }
    
    def _initialize_classification_rules(self) -> List[ClassificationRule]:
        """Initialize classification rules (Task 1.3-T1)"""
        
        rules = []
        
        # RBA CLASSIFICATION RULES
        rules.append(ClassificationRule(
            rule_id="rba_deterministic",
            automation_type=AutomationType.RBA,
            conditions={
                "has_ml_components": False,
                "has_agent_calls": False,
                "complexity_score": "<0.5",
                "decision_type": "deterministic"
            },
            weight=0.9,
            description="Pure deterministic workflow with simple rules"
        ))
        
        rules.append(ClassificationRule(
            rule_id="rba_simple_validation",
            automation_type=AutomationType.RBA,
            conditions={
                "primary_purpose": "validation",
                "step_count": "<10",
                "external_apis": "<3",
                "has_loops": False
            },
            weight=0.8,
            description="Simple validation or data processing workflow"
        ))
        
        # RBIA CLASSIFICATION RULES
        rules.append(ClassificationRule(
            rule_id="rbia_ml_augmented",
            automation_type=AutomationType.RBIA,
            conditions={
                "has_ml_components": True,
                "has_agent_calls": False,
                "ml_decision_count": ">0",
                "rule_decision_count": ">0"
            },
            weight=0.95,
            description="ML-augmented workflow with rule-based structure"
        ))
        
        rules.append(ClassificationRule(
            rule_id="rbia_intelligent_routing",
            automation_type=AutomationType.RBIA,
            conditions={
                "has_scoring": True,
                "has_classification": True,
                "complexity_score": "0.5-0.8",
                "data_driven": True
            },
            weight=0.85,
            description="Intelligent routing or scoring workflow"
        ))
        
        # AALA CLASSIFICATION RULES
        rules.append(ClassificationRule(
            rule_id="aala_agent_led",
            automation_type=AutomationType.AALA,
            conditions={
                "has_agent_calls": True,
                "reasoning_required": True,
                "multi_step_planning": True,
                "context_awareness": True
            },
            weight=0.95,
            description="Agent-led workflow with reasoning and planning"
        ))
        
        rules.append(ClassificationRule(
            rule_id="aala_complex_reasoning",
            automation_type=AutomationType.AALA,
            conditions={
                "complexity_score": ">0.8",
                "decision_depth": ">3",
                "requires_context": True,
                "adaptive_behavior": True
            },
            weight=0.9,
            description="Complex workflow requiring reasoning and adaptation"
        ))
        
        # INDUSTRY-SPECIFIC RULES
        
        # Banking - Higher threshold for AALA due to regulations
        rules.append(ClassificationRule(
            rule_id="banking_conservative_rba",
            automation_type=AutomationType.RBA,
            conditions={
                "industry": "Banking",
                "regulatory_sensitive": True,
                "audit_requirements": "high",
                "deterministic_preferred": True
            },
            weight=0.8,
            industry_specific=True,
            description="Banking workflows prefer deterministic RBA for compliance"
        ))
        
        # SaaS - More open to AALA adoption
        rules.append(ClassificationRule(
            rule_id="saas_aala_friendly",
            automation_type=AutomationType.AALA,
            conditions={
                "industry": "SaaS",
                "innovation_tolerance": "high",
                "user_experience_focus": True,
                "rapid_iteration": True
            },
            weight=0.7,
            industry_specific=True,
            description="SaaS workflows can leverage AALA for better UX"
        ))
        
        return rules
    
    async def classify_workflow(
        self,
        workflow_id: str,
        tenant_id: int,
        workflow_definition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify workflow into appropriate automation type (Task 1.3-T1)
        """
        try:
            # Get workflow definition if not provided
            if not workflow_definition:
                workflow_data = await self._get_workflow_data(workflow_id, tenant_id)
                workflow_definition = workflow_data.get("workflow_definition", {})
            
            # Extract workflow characteristics
            characteristics = await self._extract_workflow_characteristics(
                workflow_definition, tenant_id
            )
            
            # Apply classification rules
            classification_scores = await self._apply_classification_rules(
                characteristics, tenant_id
            )
            
            # Determine final classification
            final_classification = self._determine_final_classification(classification_scores)
            
            # Calculate confidence level
            confidence = self._calculate_confidence(classification_scores, characteristics)
            
            # Get industry-specific adjustments
            industry_code = await self._get_tenant_industry(tenant_id)
            industry_adjustments = self._get_industry_adjustments(
                final_classification, industry_code, characteristics
            )
            
            return {
                "workflow_id": workflow_id,
                "tenant_id": tenant_id,
                "classification": final_classification,
                "confidence": confidence,
                "confidence_score": classification_scores[final_classification],
                "characteristics": characteristics,
                "classification_scores": classification_scores,
                "industry_adjustments": industry_adjustments,
                "classification_timestamp": datetime.utcnow().isoformat(),
                "classifier_version": "1.0"
            }
            
        except Exception as e:
            self.logger.error(f"Error classifying workflow {workflow_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Classification failed: {e}")
    
    async def batch_classify_workflows(
        self,
        tenant_id: int,
        workflow_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classify multiple workflows for a tenant (Task 1.3-T1)
        """
        try:
            # Get all workflows for tenant if not specified
            if not workflow_ids:
                workflows = await self._get_tenant_workflows(tenant_id)
                workflow_ids = [w["workflow_id"] for w in workflows]
            
            classifications = {}
            classification_summary = {
                AutomationType.RBA: 0,
                AutomationType.RBIA: 0,
                AutomationType.AALA: 0
            }
            
            # Classify each workflow
            for workflow_id in workflow_ids:
                try:
                    classification = await self.classify_workflow(workflow_id, tenant_id)
                    classifications[workflow_id] = classification
                    
                    # Update summary
                    classified_type = AutomationType(classification["classification"])
                    classification_summary[classified_type] += 1
                    
                except Exception as e:
                    self.logger.error(f"Error classifying workflow {workflow_id}: {e}")
                    classifications[workflow_id] = {"error": str(e)}
            
            return {
                "tenant_id": tenant_id,
                "total_workflows": len(workflow_ids),
                "classifications": classifications,
                "summary": classification_summary,
                "classification_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch classification: {e}")
            raise HTTPException(status_code=500, detail=f"Batch classification failed: {e}")
    
    async def get_classification_rules(
        self,
        automation_type: Optional[AutomationType] = None,
        industry_specific: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Get classification rules (Task 1.3-T1)
        """
        rules = self.classification_rules
        
        # Filter by automation type
        if automation_type:
            rules = [r for r in rules if r.automation_type == automation_type]
        
        # Filter by industry specificity
        if industry_specific is not None:
            rules = [r for r in rules if r.industry_specific == industry_specific]
        
        return {
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "automation_type": rule.automation_type,
                    "conditions": rule.conditions,
                    "weight": rule.weight,
                    "industry_specific": rule.industry_specific,
                    "description": rule.description
                }
                for rule in rules
            ],
            "total_rules": len(rules)
        }
    
    async def _extract_workflow_characteristics(
        self,
        workflow_definition: Dict[str, Any],
        tenant_id: int
    ) -> Dict[str, Any]:
        """Extract characteristics from workflow definition"""
        
        characteristics = {
            "step_count": 0,
            "step_types": set(),
            "has_ml_components": False,
            "has_agent_calls": False,
            "has_loops": False,
            "has_conditions": False,
            "complexity_score": 0.0,
            "decision_count": 0,
            "ml_decision_count": 0,
            "external_api_count": 0,
            "keywords": set(),
            "patterns_found": [],
            "primary_purpose": "unknown"
        }
        
        steps = workflow_definition.get("steps", [])
        characteristics["step_count"] = len(steps)
        
        # Analyze each step
        for step in steps:
            step_type = step.get("type", "")
            characteristics["step_types"].add(step_type)
            
            # Check for specific step types
            if step_type == "ml_decision":
                characteristics["has_ml_components"] = True
                characteristics["ml_decision_count"] += 1
            elif step_type == "agent_call":
                characteristics["has_agent_calls"] = True
            elif step_type == "decision":
                characteristics["has_conditions"] = True
                characteristics["decision_count"] += 1
            elif step_type == "loop":
                characteristics["has_loops"] = True
            
            # Check for external API calls
            if step.get("params", {}).get("external", False):
                characteristics["external_api_count"] += 1
            
            # Extract keywords from step description/params
            step_text = json.dumps(step).lower()
            for keyword_set in [self.rba_indicators["keywords"], 
                              self.rbia_indicators["keywords"], 
                              self.aala_indicators["keywords"]]:
                for keyword in keyword_set:
                    if keyword in step_text:
                        characteristics["keywords"].add(keyword)
            
            # Check for patterns
            for pattern_name, pattern in self.rba_indicators["patterns"].items():
                if re.search(pattern, step_text, re.IGNORECASE):
                    characteristics["patterns_found"].append(f"rba_{pattern_name}")
            
            for pattern_name, pattern in self.rbia_indicators["patterns"].items():
                if re.search(pattern, step_text, re.IGNORECASE):
                    characteristics["patterns_found"].append(f"rbia_{pattern_name}")
            
            for pattern_name, pattern in self.aala_indicators["patterns"].items():
                if re.search(pattern, step_text, re.IGNORECASE):
                    characteristics["patterns_found"].append(f"aala_{pattern_name}")
        
        # Calculate complexity score
        characteristics["complexity_score"] = self._calculate_complexity_score(characteristics)
        
        # Determine primary purpose
        characteristics["primary_purpose"] = self._determine_primary_purpose(characteristics)
        
        # Convert sets to lists for JSON serialization
        characteristics["step_types"] = list(characteristics["step_types"])
        characteristics["keywords"] = list(characteristics["keywords"])
        
        return characteristics
    
    def _calculate_complexity_score(self, characteristics: Dict[str, Any]) -> float:
        """Calculate workflow complexity score (0.0 to 1.0)"""
        
        factors = []
        
        # Step count factor (normalized to 0-1)
        step_count = characteristics.get("step_count", 0)
        factors.append(min(step_count / 20.0, 1.0))
        
        # Decision complexity factor
        decision_count = characteristics.get("decision_count", 0)
        factors.append(min(decision_count / 10.0, 1.0))
        
        # Integration complexity factor
        api_count = characteristics.get("external_api_count", 0)
        factors.append(min(api_count / 5.0, 1.0))
        
        # Loop complexity factor
        if characteristics.get("has_loops", False):
            factors.append(0.3)
        
        # ML complexity factor
        if characteristics.get("has_ml_components", False):
            factors.append(0.4)
        
        # Agent complexity factor
        if characteristics.get("has_agent_calls", False):
            factors.append(0.5)
        
        # Calculate weighted average
        if factors:
            return sum(factors) / len(factors)
        
        return 0.0
    
    def _determine_primary_purpose(self, characteristics: Dict[str, Any]) -> str:
        """Determine primary purpose of workflow"""
        
        keywords = characteristics.get("keywords", [])
        patterns = characteristics.get("patterns_found", [])
        
        # Validation/verification workflows
        if any(kw in keywords for kw in ["validate", "verify", "check"]):
            return "validation"
        
        # Data processing workflows
        if any(kw in keywords for kw in ["process", "transform", "sync"]):
            return "data_processing"
        
        # Decision/routing workflows
        if characteristics.get("decision_count", 0) > 2:
            return "decision_routing"
        
        # ML/scoring workflows
        if any(kw in keywords for kw in ["score", "predict", "classify"]):
            return "ml_scoring"
        
        # Agent/reasoning workflows
        if any(kw in keywords for kw in ["agent", "reasoning", "planning"]):
            return "agent_reasoning"
        
        # Notification workflows
        if any(kw in keywords for kw in ["notify", "alert", "send"]):
            return "notification"
        
        return "general_automation"
    
    async def _apply_classification_rules(
        self,
        characteristics: Dict[str, Any],
        tenant_id: int
    ) -> Dict[AutomationType, float]:
        """Apply classification rules and calculate scores"""
        
        scores = {
            AutomationType.RBA: 0.0,
            AutomationType.RBIA: 0.0,
            AutomationType.AALA: 0.0
        }
        
        industry_code = await self._get_tenant_industry(tenant_id)
        
        for rule in self.classification_rules:
            # Skip industry-specific rules if not applicable
            if rule.industry_specific:
                if rule.conditions.get("industry") != industry_code.value:
                    continue
            
            # Evaluate rule conditions
            rule_score = self._evaluate_rule_conditions(rule.conditions, characteristics)
            
            # Apply weighted score
            scores[rule.automation_type] += rule_score * rule.weight
        
        # Normalize scores
        max_score = max(scores.values()) if max(scores.values()) > 0 else 1.0
        normalized_scores = {k: v / max_score for k, v in scores.items()}
        
        return normalized_scores
    
    def _evaluate_rule_conditions(
        self,
        conditions: Dict[str, Any],
        characteristics: Dict[str, Any]
    ) -> float:
        """Evaluate how well characteristics match rule conditions"""
        
        matches = 0
        total_conditions = len(conditions)
        
        if total_conditions == 0:
            return 0.0
        
        for condition_key, condition_value in conditions.items():
            if condition_key in characteristics:
                char_value = characteristics[condition_key]
                
                if self._condition_matches(condition_value, char_value):
                    matches += 1
        
        return matches / total_conditions
    
    def _condition_matches(self, condition_value: Any, char_value: Any) -> bool:
        """Check if characteristic value matches condition"""
        
        if isinstance(condition_value, bool):
            return condition_value == char_value
        
        if isinstance(condition_value, str):
            if condition_value.startswith("<"):
                threshold = float(condition_value[1:])
                return float(char_value) < threshold
            elif condition_value.startswith(">"):
                threshold = float(condition_value[1:])
                return float(char_value) > threshold
            elif "-" in condition_value:
                # Range condition like "0.5-0.8"
                min_val, max_val = map(float, condition_value.split("-"))
                return min_val <= float(char_value) <= max_val
            else:
                return condition_value == str(char_value)
        
        return condition_value == char_value
    
    def _determine_final_classification(
        self,
        classification_scores: Dict[AutomationType, float]
    ) -> AutomationType:
        """Determine final classification based on scores"""
        
        return max(classification_scores.keys(), key=lambda k: classification_scores[k])
    
    def _calculate_confidence(
        self,
        classification_scores: Dict[AutomationType, float],
        characteristics: Dict[str, Any]
    ) -> ClassificationConfidence:
        """Calculate confidence level of classification"""
        
        max_score = max(classification_scores.values())
        second_max = sorted(classification_scores.values(), reverse=True)[1]
        
        # Calculate confidence based on score separation
        confidence_score = max_score - second_max
        
        if confidence_score > 0.3:
            return ClassificationConfidence.HIGH
        elif confidence_score > 0.15:
            return ClassificationConfidence.MEDIUM
        else:
            return ClassificationConfidence.LOW

# Initialize service
workflow_classifier = None

def get_workflow_classifier():
    """Get workflow classifier instance"""
    global workflow_classifier
    if workflow_classifier is None:
        pool_manager = get_pool_manager()
        workflow_classifier = WorkflowClassifier(pool_manager)
    return workflow_classifier

# =====================================================
# API ENDPOINTS
# =====================================================

@router.post("/classify/workflow")
async def classify_workflow_endpoint(
    workflow_id: str = Query(..., description="Workflow ID"),
    tenant_id: int = Query(..., description="Tenant ID"),
    classifier: WorkflowClassifier = Depends(get_workflow_classifier)
):
    """
    Classify single workflow (Task 1.3-T1)
    """
    return await classifier.classify_workflow(workflow_id, tenant_id)

@router.post("/classify/batch")
async def batch_classify_workflows_endpoint(
    tenant_id: int = Query(..., description="Tenant ID"),
    workflow_ids: Optional[List[str]] = Query(None, description="Workflow IDs (optional)"),
    classifier: WorkflowClassifier = Depends(get_workflow_classifier)
):
    """
    Batch classify workflows for tenant (Task 1.3-T1)
    """
    return await classifier.batch_classify_workflows(tenant_id, workflow_ids)

@router.get("/classify/rules")
async def get_classification_rules_endpoint(
    automation_type: Optional[AutomationType] = Query(None, description="Filter by automation type"),
    industry_specific: Optional[bool] = Query(None, description="Filter by industry specificity"),
    classifier: WorkflowClassifier = Depends(get_workflow_classifier)
):
    """
    Get classification rules (Task 1.3-T1)
    """
    return await classifier.get_classification_rules(automation_type, industry_specific)
