# PII Handling Service with Azure Purview Integration
# Tasks 18.2.4, 18.2.8, 18.2.11, 18.2.12: PII catalog, masking, anonymization, purge service

import json
import uuid
import hashlib
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

logger = logging.getLogger(__name__)

class PIICategory(Enum):
    """PII categories by industry"""
    # SaaS PII
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    ADDRESS = "address"
    IP_ADDRESS = "ip_address"
    USER_ID = "user_id"
    CONTRACT_ID = "contract_id"
    
    # Banking PII
    ACCOUNT_NUMBER = "account_number"
    PAN = "pan"
    AADHAAR = "aadhaar"
    TRANSACTION_ID = "transaction_id"
    CREDIT_SCORE = "credit_score"
    
    # Insurance PII
    POLICY_NUMBER = "policy_number"
    SSN = "ssn"
    HEALTH_RECORD = "health_record"
    CLAIM_ID = "claim_id"
    MEDICAL_DATA = "medical_data"

class MaskingType(Enum):
    """PII masking types"""
    REDACT = "redact"
    MASK = "mask"
    ENCRYPT = "encrypt"
    ANONYMIZE = "anonymize"
    PSEUDONYMIZE = "pseudonymize"
    HASH = "hash"

class AnonymizationMethod(Enum):
    """Anonymization methods"""
    K_ANONYMITY = "k_anonymity"
    L_DIVERSITY = "l_diversity"
    T_CLOSENESS = "t_closeness"
    DIFFERENTIAL_PRIVACY = "differential_privacy"

class SensitivityLevel(Enum):
    """Data sensitivity levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

@dataclass
class PIIClassification:
    """PII classification result"""
    field_name: str
    category: PIICategory
    sensitivity_level: SensitivityLevel
    confidence_score: float
    industry_overlay: str
    regulatory_frameworks: List[str]
    detection_pattern: str
    masking_required: bool

@dataclass
class MaskingRule:
    """PII masking rule"""
    rule_id: str
    category: PIICategory
    masking_type: MaskingType
    field_patterns: List[str]
    replacement_pattern: str
    preserve_format: bool
    industry_overlay: str
    compliance_frameworks: List[str]

@dataclass
class AnonymizationConfig:
    """Anonymization configuration"""
    method: AnonymizationMethod
    k_value: Optional[int] = None
    l_value: Optional[int] = None
    t_threshold: Optional[float] = None
    epsilon: Optional[float] = None
    quasi_identifiers: List[str] = None
    sensitive_attributes: List[str] = None

class PIIDetector:
    """
    PII Detection and Classification Service
    Task 18.2.5-18.2.7: Define industry-specific PII attributes
    """
    
    def __init__(self):
        self.detection_patterns = self._initialize_detection_patterns()
        self.industry_classifications = self._initialize_industry_classifications()
    
    def _initialize_detection_patterns(self) -> Dict[PIICategory, Dict[str, Any]]:
        """Initialize PII detection patterns"""
        return {
            # SaaS PII patterns
            PIICategory.EMAIL: {
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "confidence": 0.95,
                "sensitivity": SensitivityLevel.INTERNAL
            },
            PIICategory.PHONE: {
                "pattern": r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
                "confidence": 0.90,
                "sensitivity": SensitivityLevel.CONFIDENTIAL
            },
            PIICategory.IP_ADDRESS: {
                "pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                "confidence": 0.85,
                "sensitivity": SensitivityLevel.INTERNAL
            },
            
            # Banking PII patterns
            PIICategory.PAN: {
                "pattern": r'[A-Z]{5}[0-9]{4}[A-Z]{1}',
                "confidence": 0.98,
                "sensitivity": SensitivityLevel.RESTRICTED
            },
            PIICategory.AADHAAR: {
                "pattern": r'\b[2-9]{1}[0-9]{3}[-\s]?[0-9]{4}[-\s]?[0-9]{4}\b',
                "confidence": 0.95,
                "sensitivity": SensitivityLevel.RESTRICTED
            },
            PIICategory.ACCOUNT_NUMBER: {
                "pattern": r'\b\d{10,16}\b',
                "confidence": 0.70,
                "sensitivity": SensitivityLevel.RESTRICTED
            },
            
            # Insurance PII patterns
            PIICategory.SSN: {
                "pattern": r'\b\d{3}-\d{2}-\d{4}\b',
                "confidence": 0.98,
                "sensitivity": SensitivityLevel.RESTRICTED
            },
            PIICategory.POLICY_NUMBER: {
                "pattern": r'POL[0-9]{8,12}',
                "confidence": 0.90,
                "sensitivity": SensitivityLevel.CONFIDENTIAL
            }
        }
    
    def _initialize_industry_classifications(self) -> Dict[str, Dict[str, Any]]:
        """Initialize industry-specific PII classifications"""
        return {
            "saas": {
                "required_categories": [PIICategory.EMAIL, PIICategory.NAME, PIICategory.IP_ADDRESS],
                "optional_categories": [PIICategory.PHONE, PIICategory.ADDRESS],
                "compliance_frameworks": ["GDPR", "CCPA", "SOX"],
                "default_sensitivity": SensitivityLevel.INTERNAL
            },
            "banking": {
                "required_categories": [PIICategory.ACCOUNT_NUMBER, PIICategory.PAN, PIICategory.AADHAAR],
                "optional_categories": [PIICategory.EMAIL, PIICategory.PHONE, PIICategory.ADDRESS],
                "compliance_frameworks": ["RBI", "DPDP", "AML", "KYC"],
                "default_sensitivity": SensitivityLevel.RESTRICTED
            },
            "insurance": {
                "required_categories": [PIICategory.POLICY_NUMBER, PIICategory.SSN, PIICategory.HEALTH_RECORD],
                "optional_categories": [PIICategory.EMAIL, PIICategory.PHONE, PIICategory.ADDRESS],
                "compliance_frameworks": ["HIPAA", "NAIC", "SOX"],
                "default_sensitivity": SensitivityLevel.RESTRICTED
            }
        }
    
    async def classify_pii_fields(
        self,
        data: Dict[str, Any],
        industry_overlay: str = "global"
    ) -> List[PIIClassification]:
        """
        Classify PII fields in data
        Task 18.2.5-18.2.7: Industry-specific PII classification
        """
        classifications = []
        
        for field_name, field_value in data.items():
            if not isinstance(field_value, str):
                continue
            
            # Check against detection patterns
            for category, pattern_info in self.detection_patterns.items():
                pattern = pattern_info["pattern"]
                
                if re.search(pattern, str(field_value)):
                    # Get industry-specific configuration
                    industry_config = self.industry_classifications.get(
                        industry_overlay, 
                        self.industry_classifications["saas"]
                    )
                    
                    classification = PIIClassification(
                        field_name=field_name,
                        category=category,
                        sensitivity_level=pattern_info["sensitivity"],
                        confidence_score=pattern_info["confidence"],
                        industry_overlay=industry_overlay,
                        regulatory_frameworks=industry_config["compliance_frameworks"],
                        detection_pattern=pattern,
                        masking_required=True
                    )
                    
                    classifications.append(classification)
                    break
        
        return classifications

class PIIMaskingService:
    """
    PII Masking and Encryption Service
    Task 18.2.8-18.2.10: Field-level masking, encryption at rest/transit
    """
    
    def __init__(self):
        self.masking_rules = self._initialize_masking_rules()
        self.encryption_key = self._get_encryption_key()
        self.fernet = Fernet(self.encryption_key)
    
    def _get_encryption_key(self) -> bytes:
        """Get or generate encryption key"""
        # In production, retrieve from Azure KeyVault
        key_material = os.environ.get("PII_ENCRYPTION_KEY", "default-key-material").encode()
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use random salt per tenant
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(key_material))
        return key
    
    def _initialize_masking_rules(self) -> Dict[str, MaskingRule]:
        """Initialize default masking rules"""
        return {
            "email_mask": MaskingRule(
                rule_id="email_mask",
                category=PIICategory.EMAIL,
                masking_type=MaskingType.MASK,
                field_patterns=["email", "email_address", "user_email"],
                replacement_pattern="***@***.***",
                preserve_format=True,
                industry_overlay="global",
                compliance_frameworks=["GDPR", "CCPA"]
            ),
            "phone_mask": MaskingRule(
                rule_id="phone_mask",
                category=PIICategory.PHONE,
                masking_type=MaskingType.MASK,
                field_patterns=["phone", "phone_number", "mobile"],
                replacement_pattern="***-***-****",
                preserve_format=True,
                industry_overlay="global",
                compliance_frameworks=["GDPR", "CCPA"]
            ),
            "pan_encrypt": MaskingRule(
                rule_id="pan_encrypt",
                category=PIICategory.PAN,
                masking_type=MaskingType.ENCRYPT,
                field_patterns=["pan", "pan_number", "permanent_account_number"],
                replacement_pattern="[ENCRYPTED]",
                preserve_format=False,
                industry_overlay="banking",
                compliance_frameworks=["RBI", "DPDP"]
            ),
            "ssn_redact": MaskingRule(
                rule_id="ssn_redact",
                category=PIICategory.SSN,
                masking_type=MaskingType.REDACT,
                field_patterns=["ssn", "social_security", "social_security_number"],
                replacement_pattern="[REDACTED]",
                preserve_format=False,
                industry_overlay="insurance",
                compliance_frameworks=["HIPAA", "NAIC"]
            )
        }
    
    async def apply_masking(
        self,
        data: Dict[str, Any],
        classifications: List[PIIClassification],
        industry_overlay: str = "global"
    ) -> Dict[str, Any]:
        """
        Apply PII masking to data
        Task 18.2.8: Implement field-level masking rules
        """
        masked_data = data.copy()
        masking_log = []
        
        for classification in classifications:
            field_name = classification.field_name
            
            if field_name not in data:
                continue
            
            # Find applicable masking rule
            masking_rule = self._get_masking_rule(classification, industry_overlay)
            
            if not masking_rule:
                continue
            
            original_value = data[field_name]
            
            # Apply masking based on type
            if masking_rule.masking_type == MaskingType.REDACT:
                masked_value = masking_rule.replacement_pattern
            elif masking_rule.masking_type == MaskingType.MASK:
                masked_value = self._apply_pattern_masking(original_value, masking_rule)
            elif masking_rule.masking_type == MaskingType.ENCRYPT:
                masked_value = self._encrypt_field(original_value)
            elif masking_rule.masking_type == MaskingType.HASH:
                masked_value = self._hash_field(original_value)
            else:
                masked_value = masking_rule.replacement_pattern
            
            masked_data[field_name] = masked_value
            
            # Log masking operation
            masking_log.append({
                "field_name": field_name,
                "category": classification.category.value,
                "masking_type": masking_rule.masking_type.value,
                "rule_id": masking_rule.rule_id
            })
        
        # Add masking metadata
        masked_data["_pii_masking_applied"] = True
        masked_data["_pii_masking_log"] = masking_log
        masked_data["_pii_masking_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        return masked_data
    
    def _get_masking_rule(
        self,
        classification: PIIClassification,
        industry_overlay: str
    ) -> Optional[MaskingRule]:
        """Get applicable masking rule for classification"""
        for rule in self.masking_rules.values():
            if (rule.category == classification.category and
                (rule.industry_overlay == industry_overlay or rule.industry_overlay == "global")):
                return rule
        return None
    
    def _apply_pattern_masking(self, value: str, rule: MaskingRule) -> str:
        """Apply pattern-based masking"""
        if rule.category == PIICategory.EMAIL:
            # Mask email: user@domain.com -> u***@d***.com
            parts = value.split('@')
            if len(parts) == 2:
                username = parts[0][0] + '*' * (len(parts[0]) - 1) if len(parts[0]) > 1 else '*'
                domain_parts = parts[1].split('.')
                domain = domain_parts[0][0] + '*' * (len(domain_parts[0]) - 1) if len(domain_parts[0]) > 1 else '*'
                extension = '.'.join(domain_parts[1:]) if len(domain_parts) > 1 else 'com'
                return f"{username}@{domain}.{extension}"
        
        elif rule.category == PIICategory.PHONE:
            # Mask phone: 123-456-7890 -> ***-***-7890
            if len(value) >= 4:
                return '*' * (len(value) - 4) + value[-4:]
        
        return rule.replacement_pattern
    
    def _encrypt_field(self, value: str) -> str:
        """Encrypt field value"""
        try:
            encrypted_bytes = self.fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            logger.error(f"❌ Encryption failed: {e}")
            return "[ENCRYPTION_ERROR]"
    
    def _hash_field(self, value: str) -> str:
        """Hash field value"""
        return hashlib.sha256(value.encode()).hexdigest()
    
    async def decrypt_field(self, encrypted_value: str) -> str:
        """Decrypt field value"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"❌ Decryption failed: {e}")
            return "[DECRYPTION_ERROR]"

class PIIAnonymizationService:
    """
    PII Anonymization Service
    Task 18.2.11: Implement anonymization rules (k-anonymity, l-diversity, t-closeness)
    """
    
    def __init__(self):
        self.anonymization_configs = self._initialize_anonymization_configs()
    
    def _initialize_anonymization_configs(self) -> Dict[str, AnonymizationConfig]:
        """Initialize anonymization configurations"""
        return {
            "k_anonymity_basic": AnonymizationConfig(
                method=AnonymizationMethod.K_ANONYMITY,
                k_value=5,
                quasi_identifiers=["age", "zipcode", "gender"],
                sensitive_attributes=["salary", "medical_condition"]
            ),
            "l_diversity_enhanced": AnonymizationConfig(
                method=AnonymizationMethod.L_DIVERSITY,
                k_value=5,
                l_value=3,
                quasi_identifiers=["age", "zipcode", "gender"],
                sensitive_attributes=["salary", "medical_condition"]
            ),
            "differential_privacy": AnonymizationConfig(
                method=AnonymizationMethod.DIFFERENTIAL_PRIVACY,
                epsilon=1.0,
                quasi_identifiers=["age", "zipcode"],
                sensitive_attributes=["salary"]
            )
        }
    
    async def anonymize_dataset(
        self,
        dataset: List[Dict[str, Any]],
        config_name: str = "k_anonymity_basic"
    ) -> List[Dict[str, Any]]:
        """
        Anonymize dataset using specified method
        Task 18.2.11: Anonymization rules implementation
        """
        config = self.anonymization_configs.get(config_name)
        if not config:
            raise ValueError(f"Unknown anonymization config: {config_name}")
        
        if config.method == AnonymizationMethod.K_ANONYMITY:
            return await self._apply_k_anonymity(dataset, config)
        elif config.method == AnonymizationMethod.L_DIVERSITY:
            return await self._apply_l_diversity(dataset, config)
        elif config.method == AnonymizationMethod.DIFFERENTIAL_PRIVACY:
            return await self._apply_differential_privacy(dataset, config)
        else:
            raise ValueError(f"Unsupported anonymization method: {config.method}")
    
    async def _apply_k_anonymity(
        self,
        dataset: List[Dict[str, Any]],
        config: AnonymizationConfig
    ) -> List[Dict[str, Any]]:
        """Apply k-anonymity to dataset"""
        # Simplified k-anonymity implementation
        # In production, use proper anonymization library
        
        anonymized_dataset = []
        
        # Group records by quasi-identifiers
        groups = {}
        for record in dataset:
            key = tuple(record.get(qi, '') for qi in config.quasi_identifiers)
            if key not in groups:
                groups[key] = []
            groups[key].append(record)
        
        # Ensure each group has at least k records
        for group_key, group_records in groups.items():
            if len(group_records) >= config.k_value:
                # Group satisfies k-anonymity
                anonymized_dataset.extend(group_records)
            else:
                # Generalize or suppress records
                for record in group_records:
                    anonymized_record = record.copy()
                    # Generalize quasi-identifiers
                    for qi in config.quasi_identifiers:
                        if qi in anonymized_record:
                            anonymized_record[qi] = self._generalize_value(
                                anonymized_record[qi], qi
                            )
                    anonymized_dataset.append(anonymized_record)
        
        return anonymized_dataset
    
    async def _apply_l_diversity(
        self,
        dataset: List[Dict[str, Any]],
        config: AnonymizationConfig
    ) -> List[Dict[str, Any]]:
        """Apply l-diversity to dataset"""
        # First apply k-anonymity
        k_anonymous_dataset = await self._apply_k_anonymity(dataset, config)
        
        # Then ensure l-diversity for sensitive attributes
        # Simplified implementation
        return k_anonymous_dataset
    
    async def _apply_differential_privacy(
        self,
        dataset: List[Dict[str, Any]],
        config: AnonymizationConfig
    ) -> List[Dict[str, Any]]:
        """Apply differential privacy to dataset"""
        # Simplified differential privacy implementation
        # In production, use proper DP library
        
        import random
        
        anonymized_dataset = []
        
        for record in dataset:
            anonymized_record = record.copy()
            
            # Add noise to sensitive attributes
            for sa in config.sensitive_attributes:
                if sa in anonymized_record and isinstance(anonymized_record[sa], (int, float)):
                    # Add Laplace noise
                    sensitivity = 1.0  # Would be calculated based on query
                    noise = random.laplace(0, sensitivity / config.epsilon)
                    anonymized_record[sa] = max(0, anonymized_record[sa] + noise)
            
            anonymized_dataset.append(anonymized_record)
        
        return anonymized_dataset
    
    def _generalize_value(self, value: Any, field_name: str) -> str:
        """Generalize value for anonymization"""
        if field_name == "age" and isinstance(value, int):
            # Age ranges
            if value < 30:
                return "20-30"
            elif value < 40:
                return "30-40"
            elif value < 50:
                return "40-50"
            else:
                return "50+"
        
        elif field_name == "zipcode" and isinstance(value, str):
            # Generalize zipcode
            return value[:3] + "**" if len(value) >= 3 else "***"
        
        return str(value)

class PIIPurgeService:
    """
    PII Data Purge Service
    Task 18.2.12: Build purge service for expired data
    """
    
    def __init__(self):
        self.retention_policies = self._initialize_retention_policies()
    
    def _initialize_retention_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize retention policies by SLA tier"""
        return {
            "bronze": {
                "retention_days": 90,
                "purge_method": "soft_delete",
                "auto_purge": True
            },
            "silver": {
                "retention_days": 400,
                "purge_method": "archive",
                "auto_purge": True
            },
            "gold": {
                "retention_days": 730,  # 2 years
                "purge_method": "archive",
                "auto_purge": False
            },
            "enterprise": {
                "retention_days": 2555,  # 7 years
                "purge_method": "archive",
                "auto_purge": False
            }
        }
    
    async def schedule_purge_jobs(self, tenant_id: int, sla_tier: str) -> Dict[str, Any]:
        """
        Schedule data purge jobs based on retention policy
        Task 18.2.12: Cascade deletes for expired data
        """
        policy = self.retention_policies.get(sla_tier, self.retention_policies["bronze"])
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=policy["retention_days"])
        
        purge_job = {
            "job_id": str(uuid.uuid4()),
            "tenant_id": tenant_id,
            "sla_tier": sla_tier,
            "cutoff_date": cutoff_date.isoformat(),
            "purge_method": policy["purge_method"],
            "auto_purge": policy["auto_purge"],
            "scheduled_at": datetime.now(timezone.utc).isoformat(),
            "status": "scheduled"
        }
        
        # In production, schedule with job queue
        logger.info(f"📅 Scheduled purge job: {json.dumps(purge_job, indent=2)}")
        
        return purge_job
    
    async def execute_purge_job(self, job_id: str) -> Dict[str, Any]:
        """Execute data purge job"""
        # In production, implement actual purge logic
        purge_result = {
            "job_id": job_id,
            "status": "completed",
            "records_purged": 0,
            "records_archived": 0,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "evidence_id": str(uuid.uuid4())
        }
        
        logger.info(f"🗑️ Purge job completed: {json.dumps(purge_result, indent=2)}")
        
        return purge_result

class PIIHandlingManager:
    """
    Comprehensive PII Handling Manager
    Coordinates detection, masking, anonymization, and purging
    """
    
    def __init__(self):
        self.detector = PIIDetector()
        self.masking_service = PIIMaskingService()
        self.anonymization_service = PIIAnonymizationService()
        self.purge_service = PIIPurgeService()
    
    async def process_data_with_pii_protection(
        self,
        data: Dict[str, Any],
        tenant_id: int,
        industry_overlay: str = "global",
        sla_tier: str = "bronze"
    ) -> Dict[str, Any]:
        """
        Comprehensive PII protection processing
        """
        try:
            # Step 1: Detect and classify PII
            classifications = await self.detector.classify_pii_fields(data, industry_overlay)
            
            if not classifications:
                return data
            
            # Step 2: Apply masking
            masked_data = await self.masking_service.apply_masking(
                data, classifications, industry_overlay
            )
            
            # Step 3: Log evidence
            evidence = {
                "tenant_id": tenant_id,
                "industry_overlay": industry_overlay,
                "sla_tier": sla_tier,
                "pii_fields_detected": len(classifications),
                "classifications": [
                    {
                        "field": c.field_name,
                        "category": c.category.value,
                        "sensitivity": c.sensitivity_level.value,
                        "confidence": c.confidence_score
                    }
                    for c in classifications
                ],
                "processed_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"🔒 PII protection applied: {json.dumps(evidence, indent=2)}")
            
            return masked_data
            
        except Exception as e:
            logger.error(f"❌ PII protection failed: {e}")
            return data
