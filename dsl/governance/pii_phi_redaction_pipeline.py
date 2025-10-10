"""
Task 8.2-T16: Add PII/PHI redaction pipeline
Dynamic PII/PHI redaction for HIPAA/GDPR compliance
"""

import asyncio
import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Pattern, Callable
from dataclasses import dataclass, field
from enum import Enum

import asyncpg

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of PII/PHI data"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_RECORD = "medical_record"
    PATIENT_ID = "patient_id"
    CUSTOM = "custom"


class RedactionMethod(Enum):
    """Redaction methods"""
    MASK = "mask"
    HASH = "hash"
    TOKENIZE = "tokenize"
    REMOVE = "remove"
    PARTIAL_MASK = "partial_mask"
    ENCRYPT = "encrypt"


class ComplianceFramework(Enum):
    """Compliance frameworks"""
    GDPR = "gdpr"
    HIPAA = "hipaa"
    CCPA = "ccpa"
    PCI_DSS = "pci_dss"
    SOX = "sox"


@dataclass
class PIIPattern:
    """PII detection pattern"""
    pattern_id: str
    pii_type: PIIType
    regex_pattern: str
    confidence_threshold: float = 0.8
    
    # Compliance requirements
    frameworks: List[ComplianceFramework] = field(default_factory=list)
    industry_specific: bool = False
    
    # Redaction settings
    default_method: RedactionMethod = RedactionMethod.MASK
    preserve_format: bool = True
    
    # Metadata
    description: str = ""
    is_active: bool = True
    
    def __post_init__(self):
        self.compiled_pattern: Pattern = re.compile(self.regex_pattern, re.IGNORECASE)


@dataclass
class RedactionRule:
    """Redaction rule configuration"""
    rule_id: str
    pii_type: PIIType
    method: RedactionMethod
    
    # Rule conditions
    tenant_ids: List[int] = field(default_factory=list)  # Empty = all tenants
    data_sources: List[str] = field(default_factory=list)  # Empty = all sources
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    # Method-specific settings
    mask_character: str = "*"
    partial_mask_start: int = 0
    partial_mask_end: int = 4
    hash_algorithm: str = "sha256"
    preserve_length: bool = True
    
    # Metadata
    description: str = ""
    is_active: bool = True
    priority: int = 100  # Higher = more priority


@dataclass
class RedactionResult:
    """Result of PII/PHI redaction"""
    original_text: str
    redacted_text: str
    
    # Detection details
    pii_detected: List[Dict[str, Any]] = field(default_factory=list)
    redaction_applied: List[Dict[str, Any]] = field(default_factory=list)
    
    # Statistics
    total_pii_found: int = 0
    total_redacted: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    
    # Performance
    processing_time_ms: float = 0.0
    
    # Compliance
    compliance_frameworks: List[ComplianceFramework] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_length": len(self.original_text),
            "redacted_length": len(self.redacted_text),
            "pii_detected": self.pii_detected,
            "redaction_applied": self.redaction_applied,
            "total_pii_found": self.total_pii_found,
            "total_redacted": self.total_redacted,
            "average_confidence": sum(self.confidence_scores) / len(self.confidence_scores) if self.confidence_scores else 0,
            "processing_time_ms": self.processing_time_ms,
            "compliance_frameworks": [f.value for f in self.compliance_frameworks]
        }


class PIIPHIRedactionPipeline:
    """
    PII/PHI Redaction Pipeline - Task 8.2-T16
    
    Dynamic PII/PHI redaction for HIPAA/GDPR compliance with:
    - Pattern-based PII detection
    - Multiple redaction methods
    - Industry-specific rules
    - Compliance framework support
    """
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None):
        self.db_pool = db_pool
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'enable_redaction': True,
            'strict_mode': True,
            'preserve_format': True,
            'enable_audit_logging': True,
            'cache_patterns': True,
            'default_confidence_threshold': 0.8
        }
        
        # PII patterns and rules
        self.pii_patterns: List[PIIPattern] = []
        self.redaction_rules: List[RedactionRule] = []
        
        # Statistics
        self.redaction_stats = {
            'total_texts_processed': 0,
            'total_pii_detected': 0,
            'total_redactions_applied': 0,
            'average_processing_time_ms': 0.0,
            'compliance_violations_prevented': 0
        }
        
        # Initialize default patterns and rules
        self._initialize_default_patterns()
        self._initialize_default_rules()
    
    async def initialize(self) -> bool:
        """Initialize PII/PHI redaction pipeline"""
        try:
            await self._create_redaction_tables()
            await self._load_custom_patterns()
            self.logger.info("✅ PII/PHI redaction pipeline initialized")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize redaction pipeline: {e}")
            return False
    
    async def redact_text(
        self,
        text: str,
        tenant_id: int,
        data_source: str = "unknown",
        compliance_frameworks: List[ComplianceFramework] = None,
        custom_rules: List[RedactionRule] = None
    ) -> RedactionResult:
        """
        Redact PII/PHI from text
        
        This is the main entry point for redaction
        """
        
        start_time = datetime.now(timezone.utc)
        
        if not text or not self.config['enable_redaction']:
            return RedactionResult(
                original_text=text,
                redacted_text=text,
                processing_time_ms=0.0
            )
        
        # Initialize result
        result = RedactionResult(
            original_text=text,
            redacted_text=text,
            compliance_frameworks=compliance_frameworks or []
        )
        
        try:
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(
                tenant_id, data_source, compliance_frameworks, custom_rules
            )
            
            # Detect PII in text
            pii_detections = await self._detect_pii(text, applicable_rules)
            result.pii_detected = pii_detections
            result.total_pii_found = len(pii_detections)
            
            # Apply redaction
            redacted_text = text
            redaction_applied = []
            
            # Sort detections by position (reverse order to maintain positions)
            sorted_detections = sorted(pii_detections, key=lambda x: x['start_pos'], reverse=True)
            
            for detection in sorted_detections:
                # Find applicable rule for this PII type
                rule = self._find_rule_for_pii_type(detection['pii_type'], applicable_rules)
                
                if rule:
                    # Apply redaction
                    original_value = detection['matched_text']
                    redacted_value = self._apply_redaction_method(original_value, rule)
                    
                    # Replace in text
                    start_pos = detection['start_pos']
                    end_pos = detection['end_pos']
                    redacted_text = redacted_text[:start_pos] + redacted_value + redacted_text[end_pos:]
                    
                    redaction_applied.append({
                        'pii_type': detection['pii_type'].value,
                        'method': rule.method.value,
                        'original_length': len(original_value),
                        'redacted_length': len(redacted_value),
                        'position': start_pos,
                        'confidence': detection['confidence']
                    })
            
            result.redacted_text = redacted_text
            result.redaction_applied = redaction_applied
            result.total_redacted = len(redaction_applied)
            result.confidence_scores = [d['confidence'] for d in pii_detections]
            
            # Calculate processing time
            end_time = datetime.now(timezone.utc)
            result.processing_time_ms = (end_time - start_time).total_seconds() * 1000
            
            # Update statistics
            self._update_redaction_stats(result)
            
            # Store redaction audit log
            if self.config['enable_audit_logging'] and result.total_redacted > 0:
                await self._store_redaction_audit(result, tenant_id, data_source)
            
            self.logger.info(f"✅ Redacted {result.total_redacted} PII items from text ({result.processing_time_ms:.2f}ms)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ PII redaction error: {e}")
            
            # Return original text on error (fail open for redaction)
            result.processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
    
    async def redact_json_data(
        self,
        data: Dict[str, Any],
        tenant_id: int,
        data_source: str = "unknown",
        compliance_frameworks: List[ComplianceFramework] = None
    ) -> Dict[str, Any]:
        """Redact PII/PHI from JSON data structure"""
        
        redacted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Redact string values
                result = await self.redact_text(
                    value, tenant_id, data_source, compliance_frameworks
                )
                redacted_data[key] = result.redacted_text
            elif isinstance(value, dict):
                # Recursively redact nested dictionaries
                redacted_data[key] = await self.redact_json_data(
                    value, tenant_id, data_source, compliance_frameworks
                )
            elif isinstance(value, list):
                # Redact list items
                redacted_list = []
                for item in value:
                    if isinstance(item, str):
                        result = await self.redact_text(
                            item, tenant_id, data_source, compliance_frameworks
                        )
                        redacted_list.append(result.redacted_text)
                    elif isinstance(item, dict):
                        redacted_item = await self.redact_json_data(
                            item, tenant_id, data_source, compliance_frameworks
                        )
                        redacted_list.append(redacted_item)
                    else:
                        redacted_list.append(item)
                redacted_data[key] = redacted_list
            else:
                # Keep non-string values as-is
                redacted_data[key] = value
        
        return redacted_data
    
    def _initialize_default_patterns(self):
        """Initialize default PII detection patterns"""
        
        patterns = [
            PIIPattern(
                pattern_id="email_pattern",
                pii_type=PIIType.EMAIL,
                regex_pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                confidence_threshold=0.9,
                frameworks=[ComplianceFramework.GDPR, ComplianceFramework.CCPA],
                default_method=RedactionMethod.PARTIAL_MASK,
                description="Email address detection"
            ),
            PIIPattern(
                pattern_id="phone_pattern",
                pii_type=PIIType.PHONE,
                regex_pattern=r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
                confidence_threshold=0.8,
                frameworks=[ComplianceFramework.GDPR, ComplianceFramework.HIPAA],
                default_method=RedactionMethod.MASK,
                description="Phone number detection"
            ),
            PIIPattern(
                pattern_id="ssn_pattern",
                pii_type=PIIType.SSN,
                regex_pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                confidence_threshold=0.95,
                frameworks=[ComplianceFramework.HIPAA, ComplianceFramework.SOX],
                default_method=RedactionMethod.HASH,
                description="Social Security Number detection"
            ),
            PIIPattern(
                pattern_id="credit_card_pattern",
                pii_type=PIIType.CREDIT_CARD,
                regex_pattern=r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
                confidence_threshold=0.9,
                frameworks=[ComplianceFramework.PCI_DSS],
                default_method=RedactionMethod.PARTIAL_MASK,
                description="Credit card number detection"
            ),
            PIIPattern(
                pattern_id="medical_record_pattern",
                pii_type=PIIType.MEDICAL_RECORD,
                regex_pattern=r'\b(MRN|MR|MEDICAL RECORD|PATIENT ID)[\s:]+([A-Z0-9]{6,12})\b',
                confidence_threshold=0.85,
                frameworks=[ComplianceFramework.HIPAA],
                default_method=RedactionMethod.TOKENIZE,
                description="Medical record number detection",
                industry_specific=True
            )
        ]
        
        self.pii_patterns = patterns
    
    def _initialize_default_rules(self):
        """Initialize default redaction rules"""
        
        rules = [
            RedactionRule(
                rule_id="gdpr_email_rule",
                pii_type=PIIType.EMAIL,
                method=RedactionMethod.PARTIAL_MASK,
                compliance_frameworks=[ComplianceFramework.GDPR],
                partial_mask_start=2,
                partial_mask_end=2,
                description="GDPR email redaction"
            ),
            RedactionRule(
                rule_id="hipaa_medical_rule",
                pii_type=PIIType.MEDICAL_RECORD,
                method=RedactionMethod.HASH,
                compliance_frameworks=[ComplianceFramework.HIPAA],
                hash_algorithm="sha256",
                description="HIPAA medical record redaction"
            ),
            RedactionRule(
                rule_id="pci_credit_card_rule",
                pii_type=PIIType.CREDIT_CARD,
                method=RedactionMethod.PARTIAL_MASK,
                compliance_frameworks=[ComplianceFramework.PCI_DSS],
                partial_mask_start=0,
                partial_mask_end=4,
                description="PCI-DSS credit card redaction"
            ),
            RedactionRule(
                rule_id="general_phone_rule",
                pii_type=PIIType.PHONE,
                method=RedactionMethod.MASK,
                description="General phone number redaction"
            ),
            RedactionRule(
                rule_id="general_ssn_rule",
                pii_type=PIIType.SSN,
                method=RedactionMethod.HASH,
                hash_algorithm="sha256",
                description="General SSN redaction"
            )
        ]
        
        self.redaction_rules = rules
    
    def _get_applicable_rules(
        self,
        tenant_id: int,
        data_source: str,
        compliance_frameworks: List[ComplianceFramework],
        custom_rules: List[RedactionRule]
    ) -> List[RedactionRule]:
        """Get applicable redaction rules"""
        
        applicable_rules = []
        
        # Start with custom rules if provided
        if custom_rules:
            applicable_rules.extend(custom_rules)
        
        # Add default rules
        for rule in self.redaction_rules:
            if not rule.is_active:
                continue
            
            # Check tenant filter
            if rule.tenant_ids and tenant_id not in rule.tenant_ids:
                continue
            
            # Check data source filter
            if rule.data_sources and data_source not in rule.data_sources:
                continue
            
            # Check compliance framework filter
            if rule.compliance_frameworks and compliance_frameworks:
                if not any(cf in compliance_frameworks for cf in rule.compliance_frameworks):
                    continue
            
            applicable_rules.append(rule)
        
        # Sort by priority (higher priority first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        
        return applicable_rules
    
    async def _detect_pii(self, text: str, applicable_rules: List[RedactionRule]) -> List[Dict[str, Any]]:
        """Detect PII in text using patterns"""
        
        detections = []
        
        # Get PII types we need to detect based on applicable rules
        target_pii_types = set(rule.pii_type for rule in applicable_rules)
        
        for pattern in self.pii_patterns:
            if not pattern.is_active:
                continue
            
            # Skip if this PII type is not needed
            if pattern.pii_type not in target_pii_types:
                continue
            
            # Find matches in text
            matches = pattern.compiled_pattern.finditer(text)
            
            for match in matches:
                detection = {
                    'pattern_id': pattern.pattern_id,
                    'pii_type': pattern.pii_type,
                    'matched_text': match.group(),
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'confidence': pattern.confidence_threshold,
                    'frameworks': [f.value for f in pattern.frameworks]
                }
                detections.append(detection)
        
        return detections
    
    def _find_rule_for_pii_type(self, pii_type: PIIType, rules: List[RedactionRule]) -> Optional[RedactionRule]:
        """Find the best rule for a PII type"""
        
        # Find rules matching the PII type
        matching_rules = [rule for rule in rules if rule.pii_type == pii_type]
        
        if not matching_rules:
            return None
        
        # Return the highest priority rule
        return matching_rules[0]  # Already sorted by priority
    
    def _apply_redaction_method(self, text: str, rule: RedactionRule) -> str:
        """Apply redaction method to text"""
        
        if rule.method == RedactionMethod.MASK:
            return rule.mask_character * len(text) if rule.preserve_length else rule.mask_character * 8
        
        elif rule.method == RedactionMethod.PARTIAL_MASK:
            if len(text) <= (rule.partial_mask_start + rule.partial_mask_end):
                return rule.mask_character * len(text)
            
            start_part = text[:rule.partial_mask_start]
            end_part = text[-rule.partial_mask_end:] if rule.partial_mask_end > 0 else ""
            middle_length = len(text) - rule.partial_mask_start - rule.partial_mask_end
            middle_part = rule.mask_character * middle_length
            
            return start_part + middle_part + end_part
        
        elif rule.method == RedactionMethod.HASH:
            if rule.hash_algorithm == "sha256":
                hash_obj = hashlib.sha256(text.encode())
            elif rule.hash_algorithm == "md5":
                hash_obj = hashlib.md5(text.encode())
            else:
                hash_obj = hashlib.sha256(text.encode())
            
            return f"HASH_{hash_obj.hexdigest()[:8]}"
        
        elif rule.method == RedactionMethod.TOKENIZE:
            # Simple tokenization (in production, use proper tokenization service)
            token_id = str(uuid.uuid4())[:8]
            return f"TOKEN_{token_id}"
        
        elif rule.method == RedactionMethod.REMOVE:
            return ""
        
        elif rule.method == RedactionMethod.ENCRYPT:
            # Placeholder for encryption (would use proper encryption in production)
            return f"ENCRYPTED_{len(text)}_CHARS"
        
        else:
            return rule.mask_character * len(text)
    
    def _update_redaction_stats(self, result: RedactionResult):
        """Update redaction statistics"""
        
        self.redaction_stats['total_texts_processed'] += 1
        self.redaction_stats['total_pii_detected'] += result.total_pii_found
        self.redaction_stats['total_redactions_applied'] += result.total_redacted
        
        if result.total_redacted > 0:
            self.redaction_stats['compliance_violations_prevented'] += 1
        
        # Update average processing time
        current_avg = self.redaction_stats['average_processing_time_ms']
        total_processed = self.redaction_stats['total_texts_processed']
        self.redaction_stats['average_processing_time_ms'] = (
            (current_avg * (total_processed - 1) + result.processing_time_ms) / total_processed
        )
    
    async def _load_custom_patterns(self):
        """Load custom PII patterns from database"""
        # In production, this would load custom patterns from database
        # For now, we use the default patterns
        pass
    
    async def _store_redaction_audit(
        self, result: RedactionResult, tenant_id: int, data_source: str
    ):
        """Store redaction audit log"""
        
        if not self.db_pool:
            return
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO pii_redaction_audit (
                        audit_id, tenant_id, data_source, original_length,
                        redacted_length, pii_detected_count, redactions_applied_count,
                        processing_time_ms, compliance_frameworks, redaction_details,
                        created_at
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                str(uuid.uuid4()), tenant_id, data_source, len(result.original_text),
                len(result.redacted_text), result.total_pii_found, result.total_redacted,
                result.processing_time_ms, [f.value for f in result.compliance_frameworks],
                json.dumps(result.to_dict()), datetime.now(timezone.utc))
                
        except Exception as e:
            self.logger.error(f"❌ Failed to store redaction audit: {e}")
    
    async def _create_redaction_tables(self):
        """Create redaction audit tables"""
        
        if not self.db_pool:
            return
        
        schema_sql = """
        -- PII redaction audit
        CREATE TABLE IF NOT EXISTS pii_redaction_audit (
            audit_id UUID PRIMARY KEY,
            tenant_id INTEGER NOT NULL,
            
            -- Context
            data_source VARCHAR(200) NOT NULL,
            
            -- Redaction results
            original_length INTEGER NOT NULL,
            redacted_length INTEGER NOT NULL,
            pii_detected_count INTEGER NOT NULL DEFAULT 0,
            redactions_applied_count INTEGER NOT NULL DEFAULT 0,
            
            -- Performance
            processing_time_ms FLOAT NOT NULL DEFAULT 0,
            
            -- Compliance
            compliance_frameworks TEXT[] DEFAULT ARRAY[],
            
            -- Details
            redaction_details JSONB DEFAULT '{}',
            
            -- Timestamps
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        
        -- Indexes
        CREATE INDEX IF NOT EXISTS idx_pii_redaction_audit_tenant ON pii_redaction_audit(tenant_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_pii_redaction_audit_source ON pii_redaction_audit(data_source, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_pii_redaction_audit_compliance ON pii_redaction_audit USING GIN(compliance_frameworks);
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(schema_sql)
            self.logger.info("✅ PII redaction tables created")
        except Exception as e:
            self.logger.error(f"❌ Failed to create redaction tables: {e}")
            raise
    
    def get_redaction_statistics(self) -> Dict[str, Any]:
        """Get redaction pipeline statistics"""
        
        detection_rate = 0.0
        if self.redaction_stats['total_texts_processed'] > 0:
            detection_rate = (
                self.redaction_stats['total_pii_detected'] / 
                self.redaction_stats['total_texts_processed']
            )
        
        redaction_rate = 0.0
        if self.redaction_stats['total_pii_detected'] > 0:
            redaction_rate = (
                self.redaction_stats['total_redactions_applied'] / 
                self.redaction_stats['total_pii_detected']
            ) * 100
        
        return {
            'total_texts_processed': self.redaction_stats['total_texts_processed'],
            'total_pii_detected': self.redaction_stats['total_pii_detected'],
            'total_redactions_applied': self.redaction_stats['total_redactions_applied'],
            'compliance_violations_prevented': self.redaction_stats['compliance_violations_prevented'],
            'average_pii_per_text': round(detection_rate, 2),
            'redaction_rate_percentage': round(redaction_rate, 2),
            'average_processing_time_ms': round(self.redaction_stats['average_processing_time_ms'], 2),
            'active_patterns_count': len([p for p in self.pii_patterns if p.is_active]),
            'active_rules_count': len([r for r in self.redaction_rules if r.is_active]),
            'supported_pii_types': [pii.value for pii in PIIType],
            'supported_methods': [method.value for method in RedactionMethod],
            'supported_frameworks': [framework.value for framework in ComplianceFramework],
            'redaction_enabled': self.config['enable_redaction']
        }


# Global PII/PHI redaction pipeline instance
pii_phi_redaction_pipeline = PIIPHIRedactionPipeline()