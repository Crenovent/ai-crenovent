#!/usr/bin/env python3
"""
AI-Assisted Builder: Natural Language to DSL Converter
=====================================================
Implements Chapter 12.1 T44-T45: AI-assisted builder mode (NL → DSL draft)

Features:
- Natural language workflow description → DSL conversion
- Context-aware DSL generation with industry overlays
- Multi-step workflow parsing and validation
- Integration with existing DSL compiler and governance
- Support for SaaS, BFSI, Insurance, E-comm, FS, IT workflows
- Policy-aware DSL generation with compliance embedding
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import re

# OpenAI Integration
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# DSL Components
from ..compiler.parser import DSLParser, WorkflowStatus
from ..governance.policy_engine import PolicyEngine
from ..registry.enhanced_capability_registry import EnhancedCapabilityRegistry, IndustryCode
from ..overlays.saas_overlay import SaaSOverlay
from ..overlays.banking_overlay import BankingOverlay
from ..overlays.insurance_overlay import InsuranceOverlay

logger = logging.getLogger(__name__)

@dataclass
class NLToDSLRequest:
    """Request for NL → DSL conversion"""
    natural_language: str
    tenant_id: int
    user_id: int
    industry: str = "SaaS"
    persona: str = "RevOps Manager"
    context: Optional[Dict[str, Any]] = None
    compliance_frameworks: List[str] = None
    
    def __post_init__(self):
        if self.compliance_frameworks is None:
            self.compliance_frameworks = ["SOX", "GDPR"]
        if self.context is None:
            self.context = {}

@dataclass
class DSLGenerationResult:
    """Result of NL → DSL conversion"""
    success: bool
    dsl_workflow: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    reasoning: str = ""
    validation_errors: List[str] = None
    governance_warnings: List[str] = None
    suggested_improvements: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.governance_warnings is None:
            self.governance_warnings = []
        if self.suggested_improvements is None:
            self.suggested_improvements = []

class NLToDSLConverter:
    """
    AI-Assisted Builder for converting natural language to DSL workflows
    
    Capabilities:
    - Parse natural language workflow descriptions
    - Generate DSL-compliant workflow definitions
    - Apply industry-specific overlays and compliance frameworks
    - Validate generated DSL against governance policies
    - Provide suggestions for workflow improvements
    """
    
    def __init__(self, pool_manager=None):
        self.pool_manager = pool_manager
        self.openai_client = None
        self.dsl_parser = None
        self.policy_engine = None
        self.capability_registry = None
        self.industry_overlays = {}
        self.initialized = False
        
        # Industry-specific DSL templates
        self.industry_templates = {
            "SaaS": self._get_saas_templates(),
            "Banking": self._get_banking_templates(),
            "Insurance": self._get_insurance_templates(),
            "E-commerce": self._get_ecommerce_templates(),
            "FinancialServices": self._get_fs_templates(),
            "ITServices": self._get_it_templates()
        }
        
    async def initialize(self):
        """Initialize the NL → DSL converter"""
        if self.initialized:
            return
            
        try:
            # Initialize OpenAI client
            if OPENAI_AVAILABLE:
                import os
                api_key = os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY')
                if api_key:
                    if os.getenv('AZURE_OPENAI_ENDPOINT'):
                        # Azure OpenAI
                        self.openai_client = AsyncOpenAI(
                            api_key=api_key,
                            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                            api_version="2024-02-15-preview"
                        )
                    else:
                        # Standard OpenAI
                        self.openai_client = AsyncOpenAI(api_key=api_key)
                        
            # Initialize DSL components
            self.dsl_parser = DSLParser()
            self.policy_engine = PolicyEngine(self.pool_manager)
            await self.policy_engine.initialize()
            
            self.capability_registry = EnhancedCapabilityRegistry(self.pool_manager)
            await self.capability_registry.initialize()
            
            # Initialize industry overlays
            self.industry_overlays = {
                "SaaS": SaaSOverlay(),
                "Banking": BankingOverlay(),
                "Insurance": InsuranceOverlay()
            }
            
            self.initialized = True
            logger.info("✅ NL → DSL Converter initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize NL → DSL Converter: {e}")
            raise
    
    async def convert_nl_to_dsl(self, request: NLToDSLRequest) -> DSLGenerationResult:
        """
        Convert natural language description to DSL workflow
        
        Args:
            request: NL → DSL conversion request
            
        Returns:
            DSLGenerationResult with generated workflow and validation
        """
        try:
            logger.info(f"🤖 Converting NL to DSL: '{request.natural_language[:100]}...'")
            
            # Step 1: Parse natural language intent
            intent_analysis = await self._analyze_nl_intent(request)
            
            # Step 2: Generate DSL workflow structure
            dsl_workflow = await self._generate_dsl_workflow(request, intent_analysis)
            
            # Step 3: Apply industry overlays
            dsl_workflow = await self._apply_industry_overlays(dsl_workflow, request)
            
            # Step 4: Embed governance and compliance
            dsl_workflow = await self._embed_governance(dsl_workflow, request)
            
            # Step 5: Validate generated DSL
            validation_result = await self._validate_dsl(dsl_workflow, request)
            
            # Step 6: Generate suggestions
            suggestions = await self._generate_suggestions(dsl_workflow, request, validation_result)
            
            confidence_score = self._calculate_confidence_score(intent_analysis, validation_result)
            
            return DSLGenerationResult(
                success=validation_result["valid"],
                dsl_workflow=dsl_workflow,
                confidence_score=confidence_score,
                reasoning=intent_analysis.get("reasoning", ""),
                validation_errors=validation_result.get("errors", []),
                governance_warnings=validation_result.get("warnings", []),
                suggested_improvements=suggestions
            )
            
        except Exception as e:
            logger.error(f"❌ NL → DSL conversion failed: {e}")
            return DSLGenerationResult(
                success=False,
                reasoning=f"Conversion failed: {str(e)}"
            )
    
    async def _analyze_nl_intent(self, request: NLToDSLRequest) -> Dict[str, Any]:
        """Analyze natural language to extract workflow intent"""
        
        if not self.openai_client:
            # Fallback to rule-based intent parsing
            return self._fallback_intent_analysis(request.natural_language)
        
        system_prompt = f"""
        You are an expert RevOps workflow analyst. Analyze the natural language description and extract:
        
        1. Workflow Type: (data_sync, approval, notification, calculation, compliance_check, etc.)
        2. Industry Context: {request.industry}
        3. Key Entities: (opportunities, accounts, leads, etc.)
        4. Actions Required: (query, decision, notify, transform, etc.)
        5. Conditions/Rules: (if-then logic, thresholds, etc.)
        6. Compliance Requirements: {', '.join(request.compliance_frameworks)}
        7. Data Sources: (Salesforce, database, API, etc.)
        8. Outputs/Notifications: (email, dashboard, API call, etc.)
        
        Respond in JSON format with detailed analysis.
        """
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4'),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.natural_language}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            logger.warning(f"⚠️ OpenAI intent analysis failed, using fallback: {e}")
            return self._fallback_intent_analysis(request.natural_language)
    
    def _fallback_intent_analysis(self, nl_text: str) -> Dict[str, Any]:
        """Rule-based fallback for intent analysis"""
        
        # Simple keyword-based analysis
        workflow_types = {
            "pipeline": ["pipeline", "opportunity", "deal", "forecast"],
            "approval": ["approve", "approval", "review", "sign-off"],
            "notification": ["notify", "alert", "email", "message"],
            "sync": ["sync", "synchronize", "update", "import"],
            "compliance": ["compliance", "audit", "policy", "governance"]
        }
        
        detected_type = "general"
        for wf_type, keywords in workflow_types.items():
            if any(keyword in nl_text.lower() for keyword in keywords):
                detected_type = wf_type
                break
        
        return {
            "workflow_type": detected_type,
            "entities": self._extract_entities(nl_text),
            "actions": self._extract_actions(nl_text),
            "conditions": self._extract_conditions(nl_text),
            "reasoning": f"Rule-based analysis detected {detected_type} workflow"
        }
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract business entities from text"""
        entities = []
        entity_patterns = {
            "opportunity": r"\b(opportunity|opportunities|deal|deals)\b",
            "account": r"\b(account|accounts|customer|customers)\b",
            "lead": r"\b(lead|leads|prospect|prospects)\b",
            "contact": r"\b(contact|contacts|person|people)\b"
        }
        
        for entity, pattern in entity_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                entities.append(entity)
        
        return entities
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract workflow actions from text"""
        actions = []
        action_patterns = {
            "query": r"\b(find|search|get|retrieve|fetch)\b",
            "decision": r"\b(if|when|check|validate|verify)\b",
            "notify": r"\b(notify|alert|email|send|message)\b",
            "transform": r"\b(calculate|compute|transform|convert)\b"
        }
        
        for action, pattern in action_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                actions.append(action)
        
        return actions
    
    def _extract_conditions(self, text: str) -> List[str]:
        """Extract conditional logic from text"""
        conditions = []
        
        # Look for conditional patterns
        if_patterns = [
            r"if\s+(.+?)\s+then",
            r"when\s+(.+?)\s+do",
            r"where\s+(.+?)(?:\s|$)"
        ]
        
        for pattern in if_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend(matches)
        
        return conditions
    
    async def _generate_dsl_workflow(self, request: NLToDSLRequest, intent: Dict[str, Any]) -> Dict[str, Any]:
        """Generate DSL workflow structure from intent analysis"""
        
        # Get industry template
        industry_template = self.industry_templates.get(request.industry, self.industry_templates["SaaS"])
        
        # Base workflow structure
        workflow = {
            "workflow_id": f"nl_generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": f"Generated Workflow - {intent.get('workflow_type', 'General')}",
            "description": request.natural_language[:200] + "..." if len(request.natural_language) > 200 else request.natural_language,
            "industry": request.industry,
            "tenant_id": request.tenant_id,
            "created_by": request.user_id,
            "status": WorkflowStatus.DRAFT.value,
            "metadata": {
                "generated_from_nl": True,
                "original_request": request.natural_language,
                "intent_analysis": intent,
                "generation_timestamp": datetime.now().isoformat()
            },
            "steps": []
        }
        
        # Generate workflow steps based on intent
        steps = await self._generate_workflow_steps(intent, industry_template)
        workflow["steps"] = steps
        
        return workflow
    
    async def _generate_workflow_steps(self, intent: Dict[str, Any], template: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate workflow steps from intent analysis"""
        
        steps = []
        step_counter = 1
        
        # Add governance step (always first)
        steps.append({
            "id": f"step_{step_counter}",
            "type": "governance",
            "name": "Governance Check",
            "params": {
                "policy_pack": "default",
                "evidence_capture": True,
                "override_allowed": True
            }
        })
        step_counter += 1
        
        # Add data query steps for detected entities
        entities = intent.get("entities", [])
        for entity in entities:
            steps.append({
                "id": f"step_{step_counter}",
                "type": "query",
                "name": f"Query {entity.title()}",
                "params": {
                    "data_source": "salesforce",
                    "entity": entity,
                    "filters": {},
                    "fields": ["*"]
                }
            })
            step_counter += 1
        
        # Add decision steps for detected conditions
        conditions = intent.get("conditions", [])
        for condition in conditions:
            steps.append({
                "id": f"step_{step_counter}",
                "type": "decision",
                "name": f"Check: {condition}",
                "params": {
                    "condition": condition,
                    "true_path": f"step_{step_counter + 1}",
                    "false_path": f"step_{step_counter + 2}"
                }
            })
            step_counter += 1
        
        # Add notification step if detected
        actions = intent.get("actions", [])
        if "notify" in actions:
            steps.append({
                "id": f"step_{step_counter}",
                "type": "notify",
                "name": "Send Notification",
                "params": {
                    "channel": "email",
                    "recipients": ["${user.email}"],
                    "template": "workflow_completion",
                    "data": {}
                }
            })
            step_counter += 1
        
        return steps
    
    async def _apply_industry_overlays(self, workflow: Dict[str, Any], request: NLToDSLRequest) -> Dict[str, Any]:
        """Apply industry-specific overlays to the workflow"""
        
        industry_overlay = self.industry_overlays.get(request.industry)
        if not industry_overlay:
            logger.warning(f"⚠️ No overlay found for industry: {request.industry}")
            return workflow
        
        try:
            # Apply industry-specific enhancements
            enhanced_workflow = await industry_overlay.enhance_workflow(workflow)
            return enhanced_workflow
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply {request.industry} overlay: {e}")
            return workflow
    
    async def _embed_governance(self, workflow: Dict[str, Any], request: NLToDSLRequest) -> Dict[str, Any]:
        """Embed governance and compliance into the workflow"""
        
        try:
            # Add compliance framework metadata
            if "metadata" not in workflow:
                workflow["metadata"] = {}
            
            workflow["metadata"]["compliance_frameworks"] = request.compliance_frameworks
            workflow["metadata"]["governance_embedded"] = True
            
            # Add policy pack references to governance steps
            for step in workflow.get("steps", []):
                if step.get("type") == "governance":
                    step["params"]["compliance_frameworks"] = request.compliance_frameworks
                    step["params"]["tenant_id"] = request.tenant_id
            
            return workflow
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to embed governance: {e}")
            return workflow
    
    async def _validate_dsl(self, workflow: Dict[str, Any], request: NLToDSLRequest) -> Dict[str, Any]:
        """Validate the generated DSL workflow"""
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            # Basic structure validation
            required_fields = ["workflow_id", "name", "steps"]
            for field in required_fields:
                if field not in workflow:
                    validation_result["errors"].append(f"Missing required field: {field}")
                    validation_result["valid"] = False
            
            # Step validation
            steps = workflow.get("steps", [])
            if not steps:
                validation_result["errors"].append("Workflow must have at least one step")
                validation_result["valid"] = False
            
            # Validate step structure
            for i, step in enumerate(steps):
                if "id" not in step:
                    validation_result["errors"].append(f"Step {i+1} missing 'id' field")
                    validation_result["valid"] = False
                
                if "type" not in step:
                    validation_result["errors"].append(f"Step {i+1} missing 'type' field")
                    validation_result["valid"] = False
            
            # Policy validation
            if self.policy_engine:
                policy_result = await self.policy_engine.validate_workflow(workflow)
                if not policy_result.get("valid", True):
                    validation_result["warnings"].extend(policy_result.get("violations", []))
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {str(e)}")
            validation_result["valid"] = False
        
        return validation_result
    
    async def _generate_suggestions(self, workflow: Dict[str, Any], request: NLToDSLRequest, validation: Dict[str, Any]) -> List[str]:
        """Generate suggestions for workflow improvement"""
        
        suggestions = []
        
        # Suggest error handling if missing
        has_error_handling = any(step.get("type") == "error_handler" for step in workflow.get("steps", []))
        if not has_error_handling:
            suggestions.append("Consider adding error handling steps for better resilience")
        
        # Suggest evidence capture if missing
        has_evidence = any("evidence_capture" in step.get("params", {}) for step in workflow.get("steps", []))
        if not has_evidence:
            suggestions.append("Consider enabling evidence capture for audit compliance")
        
        # Industry-specific suggestions
        if request.industry == "Banking":
            suggestions.append("Consider adding KYC/AML compliance checks for banking workflows")
        elif request.industry == "Insurance":
            suggestions.append("Consider adding solvency checks for insurance workflows")
        
        # Validation-based suggestions
        if validation.get("warnings"):
            suggestions.append("Address governance warnings to improve compliance")
        
        return suggestions
    
    def _calculate_confidence_score(self, intent: Dict[str, Any], validation: Dict[str, Any]) -> float:
        """Calculate confidence score for the generated DSL"""
        
        base_score = 0.7  # Base confidence
        
        # Boost for successful intent analysis
        if intent.get("workflow_type") != "general":
            base_score += 0.1
        
        # Boost for detected entities
        if intent.get("entities"):
            base_score += 0.1
        
        # Penalty for validation errors
        if validation.get("errors"):
            base_score -= 0.2 * len(validation["errors"])
        
        # Penalty for warnings
        if validation.get("warnings"):
            base_score -= 0.05 * len(validation["warnings"])
        
        return max(0.0, min(1.0, base_score))
    
    def _get_saas_templates(self) -> Dict[str, Any]:
        """Get SaaS industry templates"""
        return {
            "pipeline_hygiene": {
                "entities": ["opportunity", "account"],
                "common_actions": ["query", "decision", "notify"],
                "compliance": ["SOX", "GDPR"]
            }
        }
    
    def _get_banking_templates(self) -> Dict[str, Any]:
        """Get Banking industry templates"""
        return {
            "loan_approval": {
                "entities": ["loan", "customer", "credit_score"],
                "common_actions": ["query", "decision", "governance"],
                "compliance": ["RBI", "KYC", "AML"]
            }
        }
    
    def _get_insurance_templates(self) -> Dict[str, Any]:
        """Get Insurance industry templates"""
        return {
            "claims_processing": {
                "entities": ["claim", "policy", "customer"],
                "common_actions": ["query", "decision", "notify"],
                "compliance": ["IRDAI", "Solvency"]
            }
        }
    
    def _get_ecommerce_templates(self) -> Dict[str, Any]:
        """Get E-commerce industry templates"""
        return {
            "order_validation": {
                "entities": ["order", "customer", "payment"],
                "common_actions": ["query", "decision", "notify"],
                "compliance": ["PCI_DSS", "GDPR"]
            }
        }
    
    def _get_fs_templates(self) -> Dict[str, Any]:
        """Get Financial Services industry templates"""
        return {
            "portfolio_monitoring": {
                "entities": ["portfolio", "client", "risk"],
                "common_actions": ["query", "decision", "notify"],
                "compliance": ["SOX", "FINRA"]
            }
        }
    
    def _get_it_templates(self) -> Dict[str, Any]:
        """Get IT Services industry templates"""
        return {
            "access_review": {
                "entities": ["user", "role", "permission"],
                "common_actions": ["query", "decision", "governance"],
                "compliance": ["SOC2", "ISO27001"]
            }
        }

# Factory function
async def get_nl_to_dsl_converter(pool_manager=None) -> NLToDSLConverter:
    """Get initialized NL → DSL converter instance"""
    converter = NLToDSLConverter(pool_manager)
    await converter.initialize()
    return converter
