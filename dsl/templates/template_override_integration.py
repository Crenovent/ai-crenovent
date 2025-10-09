"""
Template Override Integration System
Task 4.2.15: Add override ledger integration to all templates

This system integrates the override ledger and override service into all industry templates,
ensuring governance-first adoption with mandatory justifications for all ML decisions.
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from operators.override_service import OverrideService
from operators.override_ledger import OverrideLedger

logger = logging.getLogger(__name__)

@dataclass
class TemplateOverrideConfig:
    """Override configuration for templates"""
    template_id: str
    override_enabled: bool = True
    require_justification: bool = True
    require_approval: bool = True
    approval_roles: List[str] = field(default_factory=lambda: ["CRO", "Compliance", "Risk_Manager"])
    auto_approve_threshold: Optional[float] = None  # Auto-approve if confidence > threshold
    override_expiry_hours: int = 24
    audit_level: str = "full"  # 'full', 'summary', 'minimal'
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])

@dataclass
class TemplateOverrideHook:
    """Override hook configuration for specific ML node in template"""
    node_id: str
    node_name: str
    override_type: str  # 'prediction', 'classification', 'scoring'
    can_override: bool = True
    requires_reason_codes: List[str] = field(default_factory=list)
    risk_level: str = "medium"  # 'low', 'medium', 'high', 'critical'
    escalation_required: bool = False

class TemplateOverrideIntegration:
    """
    Integrates override ledger functionality into all industry templates
    
    Features:
    - Override hook embedding in all ML nodes
    - Mandatory justification enforcement
    - Approval workflow integration
    - Immutable audit trail via override ledger
    - Template-specific override policies
    - Role-based override permissions
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.override_service = OverrideService()
        self.override_ledger = OverrideLedger()
        self.template_configs: Dict[str, TemplateOverrideConfig] = {}
        self.override_hooks: Dict[str, List[TemplateOverrideHook]] = {}
        self._initialize_template_overrides()
    
    def _initialize_template_overrides(self):
        """Initialize override configurations for all templates"""
        
        # SaaS Templates
        self._register_saas_overrides()
        
        # Banking Templates
        self._register_banking_overrides()
        
        # Insurance Templates
        self._register_insurance_overrides()
        
        # E-commerce Templates
        self._register_ecommerce_overrides()
        
        # Financial Services Templates
        self._register_financial_services_overrides()
        
        self.logger.info(f"Initialized override configs for {len(self.template_configs)} templates")
    
    def _register_saas_overrides(self):
        """Register override configurations for SaaS templates"""
        
        # SaaS Churn Risk Alert
        self.template_configs["saas_churn_risk_alert"] = TemplateOverrideConfig(
            template_id="saas_churn_risk_alert",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["CRO", "CSM_Manager"],
            auto_approve_threshold=0.95,
            override_expiry_hours=48,
            audit_level="full"
        )
        
        self.override_hooks["saas_churn_risk_alert"] = [
            TemplateOverrideHook(
                node_id="churn_prediction_node",
                node_name="Churn Risk Prediction",
                override_type="prediction",
                can_override=True,
                requires_reason_codes=["customer_feedback", "contract_renewal", "special_circumstances"],
                risk_level="medium",
                escalation_required=False
            ),
            TemplateOverrideHook(
                node_id="engagement_scoring_node",
                node_name="Customer Engagement Score",
                override_type="scoring",
                can_override=True,
                requires_reason_codes=["manual_assessment", "external_data"],
                risk_level="low",
                escalation_required=False
            )
        ]
        
        # SaaS Forecast Variance Detector
        self.template_configs["saas_forecast_variance_detector"] = TemplateOverrideConfig(
            template_id="saas_forecast_variance_detector",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["CFO", "Finance_Manager"],
            auto_approve_threshold=0.90,
            override_expiry_hours=24,
            audit_level="full"
        )
        
        self.override_hooks["saas_forecast_variance_detector"] = [
            TemplateOverrideHook(
                node_id="variance_detection_node",
                node_name="Forecast Variance Detection",
                override_type="prediction",
                can_override=True,
                requires_reason_codes=["market_conditions", "seasonal_adjustment", "data_quality"],
                risk_level="high",
                escalation_required=True
            )
        ]
    
    def _register_banking_overrides(self):
        """Register override configurations for Banking templates"""
        
        # Banking Credit Scoring Check
        self.template_configs["banking_credit_scoring_check"] = TemplateOverrideConfig(
            template_id="banking_credit_scoring_check",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["Risk_Manager", "Compliance_Officer", "Credit_Committee"],
            auto_approve_threshold=None,  # Never auto-approve
            override_expiry_hours=12,
            audit_level="full"
        )
        
        self.override_hooks["banking_credit_scoring_check"] = [
            TemplateOverrideHook(
                node_id="credit_score_node",
                node_name="Credit Score Calculation",
                override_type="scoring",
                can_override=True,
                requires_reason_codes=["relationship_banking", "collateral_adjustment", "regulatory_exception"],
                risk_level="critical",
                escalation_required=True
            ),
            TemplateOverrideHook(
                node_id="fraud_detection_node",
                node_name="Fraud Risk Assessment",
                override_type="classification",
                can_override=True,
                requires_reason_codes=["verified_identity", "known_customer", "false_positive"],
                risk_level="critical",
                escalation_required=True
            )
        ]
        
        # Banking Fraudulent Disbursal Detector
        self.template_configs["banking_fraudulent_disbursal_detector"] = TemplateOverrideConfig(
            template_id="banking_fraudulent_disbursal_detector",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["Fraud_Manager", "Operations_Head"],
            auto_approve_threshold=None,
            override_expiry_hours=6,
            audit_level="full"
        )
        
        self.override_hooks["banking_fraudulent_disbursal_detector"] = [
            TemplateOverrideHook(
                node_id="fraud_anomaly_node",
                node_name="Fraud Anomaly Detection",
                override_type="classification",
                can_override=True,
                requires_reason_codes=["verified_transaction", "customer_confirmed", "system_error"],
                risk_level="critical",
                escalation_required=True
            )
        ]
    
    def _register_insurance_overrides(self):
        """Register override configurations for Insurance templates"""
        
        # Insurance Claim Fraud Anomaly
        self.template_configs["insurance_claim_fraud_anomaly"] = TemplateOverrideConfig(
            template_id="insurance_claim_fraud_anomaly",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["Claims_Manager", "Fraud_Investigator", "Compliance"],
            auto_approve_threshold=None,
            override_expiry_hours=24,
            audit_level="full"
        )
        
        self.override_hooks["insurance_claim_fraud_anomaly"] = [
            TemplateOverrideHook(
                node_id="claim_fraud_node",
                node_name="Claim Fraud Detection",
                override_type="classification",
                can_override=True,
                requires_reason_codes=["investigation_complete", "documentation_verified", "expert_review"],
                risk_level="high",
                escalation_required=True
            )
        ]
        
        # Insurance Policy Lapse Predictor
        self.template_configs["insurance_policy_lapse_predictor"] = TemplateOverrideConfig(
            template_id="insurance_policy_lapse_predictor",
            override_enabled=True,
            require_justification=True,
            require_approval=False,
            approval_roles=["Retention_Manager"],
            auto_approve_threshold=0.85,
            override_expiry_hours=48,
            audit_level="summary"
        )
        
        self.override_hooks["insurance_policy_lapse_predictor"] = [
            TemplateOverrideHook(
                node_id="lapse_prediction_node",
                node_name="Policy Lapse Prediction",
                override_type="prediction",
                can_override=True,
                requires_reason_codes=["customer_contact", "payment_confirmed", "retention_offer"],
                risk_level="medium",
                escalation_required=False
            )
        ]
    
    def _register_ecommerce_overrides(self):
        """Register override configurations for E-commerce templates"""
        
        # E-commerce Fraud Scoring at Checkout
        self.template_configs["ecommerce_fraud_scoring_checkout"] = TemplateOverrideConfig(
            template_id="ecommerce_fraud_scoring_checkout",
            override_enabled=True,
            require_justification=True,
            require_approval=False,
            approval_roles=["Fraud_Analyst"],
            auto_approve_threshold=0.90,
            override_expiry_hours=2,
            audit_level="full"
        )
        
        self.override_hooks["ecommerce_fraud_scoring_checkout"] = [
            TemplateOverrideHook(
                node_id="checkout_fraud_node",
                node_name="Checkout Fraud Detection",
                override_type="scoring",
                can_override=True,
                requires_reason_codes=["customer_verified", "returning_customer", "false_alarm"],
                risk_level="high",
                escalation_required=False
            )
        ]
        
        # E-commerce Refund Delay Predictor
        self.template_configs["ecommerce_refund_delay_predictor"] = TemplateOverrideConfig(
            template_id="ecommerce_refund_delay_predictor",
            override_enabled=True,
            require_justification=True,
            require_approval=False,
            approval_roles=["CS_Manager"],
            auto_approve_threshold=0.80,
            override_expiry_hours=24,
            audit_level="summary"
        )
        
        self.override_hooks["ecommerce_refund_delay_predictor"] = [
            TemplateOverrideHook(
                node_id="refund_delay_node",
                node_name="Refund Delay Prediction",
                override_type="prediction",
                can_override=True,
                requires_reason_codes=["priority_customer", "service_recovery", "error_correction"],
                risk_level="low",
                escalation_required=False
            )
        ]
    
    def _register_financial_services_overrides(self):
        """Register override configurations for Financial Services templates"""
        
        # FS Liquidity Risk Early Warning
        self.template_configs["fs_liquidity_risk_early_warning"] = TemplateOverrideConfig(
            template_id="fs_liquidity_risk_early_warning",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["CRO", "CFO", "Risk_Committee"],
            auto_approve_threshold=None,
            override_expiry_hours=4,
            audit_level="full"
        )
        
        self.override_hooks["fs_liquidity_risk_early_warning"] = [
            TemplateOverrideHook(
                node_id="liquidity_risk_node",
                node_name="Liquidity Risk Assessment",
                override_type="scoring",
                can_override=True,
                requires_reason_codes=["capital_injection", "regulatory_waiver", "market_adjustment"],
                risk_level="critical",
                escalation_required=True
            )
        ]
        
        # FS MiFID/Reg Reporting with Anomaly Detection
        self.template_configs["fs_mifid_reg_reporting_anomaly"] = TemplateOverrideConfig(
            template_id="fs_mifid_reg_reporting_anomaly",
            override_enabled=True,
            require_justification=True,
            require_approval=True,
            approval_roles=["Compliance_Officer", "Regulatory_Affairs"],
            auto_approve_threshold=None,
            override_expiry_hours=12,
            audit_level="full"
        )
        
        self.override_hooks["fs_mifid_reg_reporting_anomaly"] = [
            TemplateOverrideHook(
                node_id="regulatory_anomaly_node",
                node_name="Regulatory Reporting Anomaly Detection",
                override_type="classification",
                can_override=True,
                requires_reason_codes=["data_correction", "regulatory_guidance", "reporting_adjustment"],
                risk_level="critical",
                escalation_required=True
            )
        ]
    
    async def create_template_override(
        self,
        template_id: str,
        workflow_id: str,
        step_id: str,
        node_id: str,
        original_prediction: Any,
        override_prediction: Any,
        justification: str,
        requested_by: int,
        tenant_id: str,
        reason_codes: Optional[List[str]] = None
    ) -> str:
        """
        Create an override request for a template ML node
        
        Args:
            template_id: Template identifier
            workflow_id: Workflow execution ID
            step_id: Workflow step ID
            node_id: ML node ID
            original_prediction: Original ML prediction/score
            override_prediction: Override value
            justification: Detailed justification
            requested_by: User ID requesting override
            tenant_id: Tenant identifier
            reason_codes: Predefined reason codes
        
        Returns:
            Override request ID
        """
        try:
            # Validate template configuration
            if template_id not in self.template_configs:
                raise ValueError(f"Template {template_id} not found in override registry")
            
            config = self.template_configs[template_id]
            
            if not config.override_enabled:
                raise ValueError(f"Overrides are disabled for template {template_id}")
            
            # Validate justification
            if config.require_justification and not justification:
                raise ValueError("Justification is mandatory for this template")
            
            # Validate reason codes
            hooks = self.override_hooks.get(template_id, [])
            node_hook = next((h for h in hooks if h.node_id == node_id), None)
            
            if node_hook and node_hook.requires_reason_codes:
                if not reason_codes:
                    raise ValueError(f"Reason codes required: {node_hook.requires_reason_codes}")
                
                # Validate reason codes
                valid_codes = set(node_hook.requires_reason_codes)
                provided_codes = set(reason_codes)
                if not provided_codes.intersection(valid_codes):
                    raise ValueError(f"At least one valid reason code required: {valid_codes}")
            
            # Create override request
            override_id = await self.override_service.create_override_request(
                workflow_id=workflow_id,
                step_id=step_id,
                model_id=template_id,
                node_id=node_id,
                original_prediction=original_prediction,
                override_prediction=override_prediction,
                justification=justification,
                override_type="manual",
                requested_by=requested_by,
                tenant_id=tenant_id
            )
            
            # Add to override ledger with template context
            await self.override_ledger.add_entry(
                override_id=override_id,
                workflow_id=workflow_id,
                step_id=step_id,
                model_id=template_id,
                node_id=node_id,
                original_prediction=original_prediction,
                override_prediction=override_prediction,
                justification=justification,
                override_type="template_override",
                requested_by=requested_by,
                approved_by=None,
                approval_status="pending",
                approval_reason="",
                tenant_id=tenant_id
            )
            
            self.logger.info(
                f"Created override request {override_id} for template {template_id}, "
                f"node {node_id}, tenant {tenant_id}"
            )
            
            return override_id
            
        except Exception as e:
            self.logger.error(f"Failed to create template override: {e}")
            raise
    
    async def approve_template_override(
        self,
        override_id: str,
        approver_id: int,
        approval_status: str,
        approval_reason: str,
        tenant_id: str
    ) -> bool:
        """
        Approve or reject an override request
        
        Args:
            override_id: Override request ID
            approver_id: User ID approving the request
            approval_status: 'approved' or 'rejected'
            approval_reason: Reason for approval/rejection
            tenant_id: Tenant identifier
        
        Returns:
            True if successful
        """
        try:
            # Approve in override service
            await self.override_service.approve_override(
                override_id=override_id,
                approver_id=approver_id,
                approval_status=approval_status,
                approval_reason=approval_reason,
                tenant_id=tenant_id
            )
            
            # Update ledger
            await self.override_ledger.update_approval(
                override_id=override_id,
                approver_id=approver_id,
                approval_status=approval_status,
                approval_reason=approval_reason
            )
            
            self.logger.info(
                f"Override {override_id} {approval_status} by user {approver_id}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to approve template override: {e}")
            raise
    
    def get_template_override_config(self, template_id: str) -> Optional[TemplateOverrideConfig]:
        """Get override configuration for a template"""
        return self.template_configs.get(template_id)
    
    def get_template_override_hooks(self, template_id: str) -> List[TemplateOverrideHook]:
        """Get override hooks for a template"""
        return self.override_hooks.get(template_id, [])
    
    async def get_template_override_history(
        self,
        template_id: str,
        tenant_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get override history for a template"""
        try:
            entries = await self.override_ledger.query_by_model(
                model_id=template_id,
                tenant_id=tenant_id
            )
            
            return entries[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get template override history: {e}")
            return []
    
    async def get_override_analytics(
        self,
        template_id: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """
        Get analytics on override patterns for a template
        
        Returns:
            Analytics including:
            - Total overrides
            - Approval rate
            - Most common reason codes
            - Override frequency by node
            - Average response time
        """
        try:
            history = await self.get_template_override_history(template_id, tenant_id, limit=1000)
            
            if not history:
                return {
                    "template_id": template_id,
                    "total_overrides": 0,
                    "analytics": {}
                }
            
            total = len(history)
            approved = sum(1 for h in history if h.get("approval_status") == "approved")
            rejected = sum(1 for h in history if h.get("approval_status") == "rejected")
            pending = sum(1 for h in history if h.get("approval_status") == "pending")
            
            # Node frequency
            node_freq = {}
            for h in history:
                node_id = h.get("node_id", "unknown")
                node_freq[node_id] = node_freq.get(node_id, 0) + 1
            
            return {
                "template_id": template_id,
                "tenant_id": tenant_id,
                "total_overrides": total,
                "approved": approved,
                "rejected": rejected,
                "pending": pending,
                "approval_rate": approved / total if total > 0 else 0,
                "rejection_rate": rejected / total if total > 0 else 0,
                "node_frequency": node_freq,
                "most_overridden_node": max(node_freq.items(), key=lambda x: x[1])[0] if node_freq else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get override analytics: {e}")
            return {"error": str(e)}
    
    def export_template_override_config(self, template_id: str) -> Dict[str, Any]:
        """Export override configuration for a template"""
        config = self.template_configs.get(template_id)
        hooks = self.override_hooks.get(template_id, [])
        
        if not config:
            return {}
        
        return {
            "template_id": template_id,
            "config": {
                "override_enabled": config.override_enabled,
                "require_justification": config.require_justification,
                "require_approval": config.require_approval,
                "approval_roles": config.approval_roles,
                "auto_approve_threshold": config.auto_approve_threshold,
                "override_expiry_hours": config.override_expiry_hours,
                "audit_level": config.audit_level,
                "notification_channels": config.notification_channels
            },
            "hooks": [
                {
                    "node_id": h.node_id,
                    "node_name": h.node_name,
                    "override_type": h.override_type,
                    "can_override": h.can_override,
                    "requires_reason_codes": h.requires_reason_codes,
                    "risk_level": h.risk_level,
                    "escalation_required": h.escalation_required
                }
                for h in hooks
            ]
        }


# Example usage
async def main():
    """Demonstrate template override integration"""
    
    integration = TemplateOverrideIntegration()
    
    print("=" * 80)
    print("TEMPLATE OVERRIDE INTEGRATION SYSTEM")
    print("=" * 80)
    print()
    
    # Example 1: SaaS Churn Risk Alert Override
    print("Example 1: SaaS Churn Risk Alert - Override Request")
    print("-" * 80)
    
    override_id = await integration.create_template_override(
        template_id="saas_churn_risk_alert",
        workflow_id="wf_12345",
        step_id="step_churn_check",
        node_id="churn_prediction_node",
        original_prediction={"churn_probability": 0.85, "confidence": 0.92},
        override_prediction={"churn_probability": 0.25, "confidence": 1.0},
        justification="Customer signed 2-year renewal contract yesterday. Manual override justified.",
        requested_by=12345,
        tenant_id="tenant_acme",
        reason_codes=["contract_renewal"]
    )
    
    print(f"✅ Override request created: {override_id}")
    print()
    
    # Approve override
    await integration.approve_template_override(
        override_id=override_id,
        approver_id=67890,
        approval_status="approved",
        approval_reason="Contract renewal verified by CSM team",
        tenant_id="tenant_acme"
    )
    
    print(f"✅ Override approved")
    print()
    
    # Example 2: Banking Credit Scoring Override
    print("Example 2: Banking Credit Scoring - High-Risk Override")
    print("-" * 80)
    
    override_id_2 = await integration.create_template_override(
        template_id="banking_credit_scoring_check",
        workflow_id="wf_67890",
        step_id="step_credit_check",
        node_id="credit_score_node",
        original_prediction={"credit_score": 620, "risk_category": "high"},
        override_prediction={"credit_score": 720, "risk_category": "medium"},
        justification="Long-standing relationship banking customer with 15+ years history. Collateral value increased.",
        requested_by=54321,
        tenant_id="tenant_megabank",
        reason_codes=["relationship_banking", "collateral_adjustment"]
    )
    
    print(f"✅ Override request created: {override_id_2}")
    print()
    
    # Example 3: Get Template Override Analytics
    print("Example 3: Override Analytics")
    print("-" * 80)
    
    analytics = await integration.get_override_analytics(
        template_id="saas_churn_risk_alert",
        tenant_id="tenant_acme"
    )
    
    print(f"Template: {analytics.get('template_id')}")
    print(f"Total Overrides: {analytics.get('total_overrides')}")
    print(f"Approved: {analytics.get('approved')}")
    print(f"Rejected: {analytics.get('rejected')}")
    print(f"Approval Rate: {analytics.get('approval_rate', 0):.2%}")
    print()
    
    # Example 4: Export Template Override Configuration
    print("Example 4: Export Override Configuration")
    print("-" * 80)
    
    config = integration.export_template_override_config("banking_credit_scoring_check")
    print(json.dumps(config, indent=2))
    print()
    
    print("=" * 80)
    print("All templates now have override ledger integration!")
    print(f"✅ {len(integration.template_configs)} templates configured")
    print(f"✅ {sum(len(hooks) for hooks in integration.override_hooks.values())} override hooks registered")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())

