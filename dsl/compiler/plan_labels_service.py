"""
Plan Labels - Task 6.2.63
==========================

Plan labels for reporting (industry, risk tier, PII exposure)
- IR labels for overlays
- Categorization and tagging system
- Metadata for reporting and filtering
"""

from typing import Dict, List, Any, Optional, Set
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class IndustryType(Enum):
    FINANCIAL_SERVICES = "financial_services"
    HEALTHCARE = "healthcare"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TECHNOLOGY = "technology"
    GOVERNMENT = "government"
    EDUCATION = "education"
    GENERIC = "generic"

class RiskTier(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PIIExposureLevel(Enum):
    NONE = "none"
    LOW = "low"      # Names, emails
    MEDIUM = "medium"  # Phone numbers, addresses
    HIGH = "high"    # SSN, financial data, health records

class ComplianceFramework(Enum):
    GDPR = "gdpr"
    HIPAA = "hipaa"
    SOX = "sox"
    PCI_DSS = "pci_dss"
    CCPA = "ccpa"

@dataclass
class PlanLabel:
    label_key: str
    label_value: str
    label_category: str
    confidence: float = 1.0  # Confidence in auto-detected labels

class PlanLabelsService:
    """Service for automatically labeling plans"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.industry_keywords = self._initialize_industry_keywords()
        self.pii_patterns = self._initialize_pii_patterns()
    
    def _initialize_industry_keywords(self) -> Dict[IndustryType, List[str]]:
        """Initialize industry detection keywords"""
        return {
            IndustryType.FINANCIAL_SERVICES: [
                'loan', 'credit', 'payment', 'transaction', 'banking', 'investment',
                'portfolio', 'risk_assessment', 'compliance', 'aml', 'kyc'
            ],
            IndustryType.HEALTHCARE: [
                'patient', 'medical', 'diagnosis', 'treatment', 'hipaa',
                'clinical', 'healthcare', 'hospital', 'pharmacy'
            ],
            IndustryType.RETAIL: [
                'customer', 'order', 'inventory', 'product', 'sales',
                'ecommerce', 'retail', 'shopping', 'merchandise'
            ],
            IndustryType.MANUFACTURING: [
                'production', 'manufacturing', 'supply_chain', 'quality',
                'assembly', 'factory', 'equipment', 'maintenance'
            ],
            IndustryType.TECHNOLOGY: [
                'software', 'api', 'system', 'application', 'platform',
                'infrastructure', 'cloud', 'deployment'
            ]
        }
    
    def _initialize_pii_patterns(self) -> Dict[PIIExposureLevel, List[str]]:
        """Initialize PII detection patterns"""
        return {
            PIIExposureLevel.LOW: [
                'name', 'email', 'username', 'first_name', 'last_name'
            ],
            PIIExposureLevel.MEDIUM: [
                'phone', 'address', 'zip_code', 'date_of_birth', 'age'
            ],
            PIIExposureLevel.HIGH: [
                'ssn', 'social_security', 'credit_card', 'bank_account',
                'medical_record', 'health_data', 'financial_data'
            ]
        }
    
    def analyze_and_label_plan(self, ir_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze plan and generate labels"""
        labels = []
        
        # Detect industry
        industry = self._detect_industry(ir_data)
        labels.append(PlanLabel(
            label_key="industry",
            label_value=industry.value,
            label_category="classification"
        ))
        
        # Assess risk tier
        risk_tier = self._assess_risk_tier(ir_data)
        labels.append(PlanLabel(
            label_key="risk_tier",
            label_value=risk_tier.value,
            label_category="risk"
        ))
        
        # Detect PII exposure
        pii_level = self._detect_pii_exposure(ir_data)
        labels.append(PlanLabel(
            label_key="pii_exposure",
            label_value=pii_level.value,
            label_category="privacy"
        ))
        
        # Detect compliance frameworks
        compliance_frameworks = self._detect_compliance_frameworks(ir_data)
        for framework in compliance_frameworks:
            labels.append(PlanLabel(
                label_key="compliance_framework",
                label_value=framework.value,
                label_category="compliance"
            ))
        
        # Detect additional characteristics
        additional_labels = self._detect_additional_characteristics(ir_data)
        labels.extend(additional_labels)
        
        # Add labels to IR metadata
        if 'metadata' not in ir_data:
            ir_data['metadata'] = {}
        
        ir_data['metadata']['plan_labels'] = [
            {
                'key': label.label_key,
                'value': label.label_value,
                'category': label.label_category,
                'confidence': label.confidence
            }
            for label in labels
        ]
        
        return ir_data
    
    def _detect_industry(self, ir_data: Dict[str, Any]) -> IndustryType:
        """Detect industry based on plan content"""
        plan_text = self._extract_plan_text(ir_data)
        plan_text_lower = plan_text.lower()
        
        industry_scores = {}
        
        for industry, keywords in self.industry_keywords.items():
            score = sum(1 for keyword in keywords if keyword in plan_text_lower)
            if score > 0:
                industry_scores[industry] = score
        
        if industry_scores:
            return max(industry_scores.keys(), key=lambda k: industry_scores[k])
        
        return IndustryType.GENERIC
    
    def _assess_risk_tier(self, ir_data: Dict[str, Any]) -> RiskTier:
        """Assess risk tier based on plan characteristics"""
        risk_factors = 0
        nodes = ir_data.get('nodes', [])
        
        # Count risk factors
        ml_nodes = len([n for n in nodes if 'ml_' in n.get('type', '')])
        external_nodes = len([n for n in nodes if 'external' in n.get('type', '')])
        high_risk_policies = len([
            n for n in nodes 
            if n.get('policies', {}).get('risk_level') == 'high'
        ])
        
        # Risk scoring
        if ml_nodes > 5:
            risk_factors += 2
        elif ml_nodes > 2:
            risk_factors += 1
        
        if external_nodes > 3:
            risk_factors += 2
        elif external_nodes > 1:
            risk_factors += 1
        
        risk_factors += high_risk_policies
        
        # Check for sensitive operations
        plan_text = self._extract_plan_text(ir_data).lower()
        sensitive_operations = ['payment', 'financial', 'medical', 'personal_data']
        if any(op in plan_text for op in sensitive_operations):
            risk_factors += 1
        
        # Determine risk tier
        if risk_factors >= 5:
            return RiskTier.CRITICAL
        elif risk_factors >= 3:
            return RiskTier.HIGH
        elif risk_factors >= 1:
            return RiskTier.MEDIUM
        else:
            return RiskTier.LOW
    
    def _detect_pii_exposure(self, ir_data: Dict[str, Any]) -> PIIExposureLevel:
        """Detect PII exposure level"""
        plan_text = self._extract_plan_text(ir_data).lower()
        
        # Check for high-sensitivity PII first
        for pattern in self.pii_patterns[PIIExposureLevel.HIGH]:
            if pattern in plan_text:
                return PIIExposureLevel.HIGH
        
        # Check for medium-sensitivity PII
        for pattern in self.pii_patterns[PIIExposureLevel.MEDIUM]:
            if pattern in plan_text:
                return PIIExposureLevel.MEDIUM
        
        # Check for low-sensitivity PII
        for pattern in self.pii_patterns[PIIExposureLevel.LOW]:
            if pattern in plan_text:
                return PIIExposureLevel.LOW
        
        return PIIExposureLevel.NONE
    
    def _detect_compliance_frameworks(self, ir_data: Dict[str, Any]) -> List[ComplianceFramework]:
        """Detect applicable compliance frameworks"""
        frameworks = []
        plan_text = self._extract_plan_text(ir_data).lower()
        policies = ir_data.get('policies', {})
        
        # Check explicit policy declarations
        if policies.get('gdpr_compliance'):
            frameworks.append(ComplianceFramework.GDPR)
        if policies.get('hipaa_compliance'):
            frameworks.append(ComplianceFramework.HIPAA)
        if policies.get('sox_compliance'):
            frameworks.append(ComplianceFramework.SOX)
        if policies.get('pci_compliance'):
            frameworks.append(ComplianceFramework.PCI_DSS)
        
        # Infer from content
        if any(term in plan_text for term in ['medical', 'patient', 'health']):
            if ComplianceFramework.HIPAA not in frameworks:
                frameworks.append(ComplianceFramework.HIPAA)
        
        if any(term in plan_text for term in ['payment', 'credit_card', 'financial']):
            if ComplianceFramework.PCI_DSS not in frameworks:
                frameworks.append(ComplianceFramework.PCI_DSS)
        
        if any(term in plan_text for term in ['personal_data', 'privacy', 'gdpr']):
            if ComplianceFramework.GDPR not in frameworks:
                frameworks.append(ComplianceFramework.GDPR)
        
        return frameworks
    
    def _detect_additional_characteristics(self, ir_data: Dict[str, Any]) -> List[PlanLabel]:
        """Detect additional plan characteristics"""
        labels = []
        nodes = ir_data.get('nodes', [])
        
        # Workflow complexity
        total_nodes = len(nodes)
        if total_nodes > 20:
            labels.append(PlanLabel("complexity", "high", "characteristics"))
        elif total_nodes > 10:
            labels.append(PlanLabel("complexity", "medium", "characteristics"))
        else:
            labels.append(PlanLabel("complexity", "low", "characteristics"))
        
        # Automation level
        ml_nodes = len([n for n in nodes if 'ml_' in n.get('type', '')])
        human_nodes = len([n for n in nodes if 'human' in n.get('type', '') or 'manual' in n.get('type', '')])
        
        if ml_nodes > human_nodes * 2:
            labels.append(PlanLabel("automation_level", "high", "characteristics"))
        elif ml_nodes > human_nodes:
            labels.append(PlanLabel("automation_level", "medium", "characteristics"))
        else:
            labels.append(PlanLabel("automation_level", "low", "characteristics"))
        
        # Data processing intensity
        data_nodes = len([n for n in nodes if 'data' in n.get('type', '') or 'transform' in n.get('type', '')])
        if data_nodes > total_nodes * 0.5:
            labels.append(PlanLabel("data_intensity", "high", "characteristics"))
        elif data_nodes > total_nodes * 0.2:
            labels.append(PlanLabel("data_intensity", "medium", "characteristics"))
        else:
            labels.append(PlanLabel("data_intensity", "low", "characteristics"))
        
        return labels
    
    def _extract_plan_text(self, ir_data: Dict[str, Any]) -> str:
        """Extract text content from plan for analysis"""
        text_parts = []
        
        # Extract from plan metadata
        metadata = ir_data.get('metadata', {})
        text_parts.append(metadata.get('description', ''))
        text_parts.append(metadata.get('purpose', ''))
        
        # Extract from nodes
        nodes = ir_data.get('nodes', [])
        for node in nodes:
            text_parts.append(node.get('name', ''))
            text_parts.append(node.get('description', ''))
            text_parts.append(node.get('type', ''))
            
            # Extract from node configuration
            config = node.get('config', {})
            for value in config.values():
                if isinstance(value, str):
                    text_parts.append(value)
        
        return ' '.join(filter(None, text_parts))
    
    def get_labels_by_category(self, ir_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Get labels grouped by category"""
        labels = ir_data.get('metadata', {}).get('plan_labels', [])
        
        categories = {}
        for label in labels:
            category = label.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(label)
        
        return categories
    
    def filter_plans_by_labels(self, plans: List[Dict[str, Any]], 
                              label_filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter plans based on label criteria"""
        filtered_plans = []
        
        for plan in plans:
            labels = plan.get('metadata', {}).get('plan_labels', [])
            label_dict = {label['key']: label['value'] for label in labels}
            
            # Check if plan matches all filters
            matches = True
            for filter_key, filter_value in label_filters.items():
                if label_dict.get(filter_key) != filter_value:
                    matches = False
                    break
            
            if matches:
                filtered_plans.append(plan)
        
        return filtered_plans

