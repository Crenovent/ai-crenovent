"""
Lint Error Taxonomy - Task 6.5.78
=================================

Telemetry: lint error taxonomy, publish failure reasons
- Categorize and track lint errors for continuous improvement
- Anonymous and tenant-safe metrics collection
- Error pattern analysis for tooling improvements
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging
import json
import hashlib

logger = logging.getLogger(__name__)

class ErrorCategory(Enum):
    SYNTAX = "syntax"
    SEMANTIC = "semantic"
    GOVERNANCE = "governance"
    POLICY = "policy"
    TYPE = "type"
    SECURITY = "security"
    PERFORMANCE = "performance"

class ErrorSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

@dataclass
class LintError:
    """Represents a lint error with taxonomy"""
    error_code: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    file_path: str
    line_number: int
    column_number: int
    rule_name: str
    tenant_safe: bool = True

@dataclass
class ErrorMetrics:
    """Metrics for error tracking"""
    error_code: str
    category: str
    severity: str
    count: int
    first_seen: datetime
    last_seen: datetime
    affected_tenants: int  # Anonymous count
    common_patterns: List[str]

class LintErrorTaxonomy:
    """Taxonomy and telemetry for lint errors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_registry = self._initialize_error_registry()
        self.metrics_store: Dict[str, ErrorMetrics] = {}
    
    def _initialize_error_registry(self) -> Dict[str, Dict[str, Any]]:
        """Initialize error code registry with taxonomy"""
        return {
            # Syntax Errors
            "DSL001": {
                "category": ErrorCategory.SYNTAX,
                "severity": ErrorSeverity.ERROR,
                "description": "Invalid YAML syntax",
                "common_causes": ["Missing quotes", "Incorrect indentation", "Invalid characters"]
            },
            "DSL002": {
                "category": ErrorCategory.SYNTAX,
                "severity": ErrorSeverity.ERROR,
                "description": "Missing required field",
                "common_causes": ["Incomplete workflow definition", "Missing mandatory governance fields"]
            },
            
            # Semantic Errors
            "DSL101": {
                "category": ErrorCategory.SEMANTIC,
                "severity": ErrorSeverity.ERROR,
                "description": "Undefined step reference",
                "common_causes": ["Typo in step ID", "Missing step definition", "Circular dependencies"]
            },
            "DSL102": {
                "category": ErrorCategory.SEMANTIC,
                "severity": ErrorSeverity.WARNING,
                "description": "Unreachable step detected",
                "common_causes": ["Dead code", "Incorrect workflow logic", "Missing conditions"]
            },
            
            # Governance Errors
            "DSL201": {
                "category": ErrorCategory.GOVERNANCE,
                "severity": ErrorSeverity.ERROR,
                "description": "Missing governance metadata",
                "common_causes": ["No tenant_id", "No region_id", "No policy_pack"]
            },
            "DSL202": {
                "category": ErrorCategory.GOVERNANCE,
                "severity": ErrorSeverity.ERROR,
                "description": "Invalid policy reference",
                "common_causes": ["Policy not found", "Policy version mismatch", "Inactive policy"]
            },
            
            # Policy Errors
            "DSL301": {
                "category": ErrorCategory.POLICY,
                "severity": ErrorSeverity.ERROR,
                "description": "Policy violation detected",
                "common_causes": ["SOX compliance failure", "GDPR violation", "Residency constraint"]
            },
            "DSL302": {
                "category": ErrorCategory.POLICY,
                "severity": ErrorSeverity.WARNING,
                "description": "Missing fallback configuration",
                "common_causes": ["ML node without fallback", "No degradation path", "High risk without backup"]
            },
            
            # Type Errors
            "DSL401": {
                "category": ErrorCategory.TYPE,
                "severity": ErrorSeverity.ERROR,
                "description": "Type mismatch",
                "common_causes": ["Wrong parameter type", "Invalid model input", "Incompatible data types"]
            },
            
            # Security Errors
            "DSL501": {
                "category": ErrorCategory.SECURITY,
                "severity": ErrorSeverity.ERROR,
                "description": "Security violation",
                "common_causes": ["Hardcoded secrets", "Insecure references", "PII exposure"]
            },
            
            # Performance Errors
            "DSL601": {
                "category": ErrorCategory.PERFORMANCE,
                "severity": ErrorSeverity.WARNING,
                "description": "Performance concern",
                "common_causes": ["Large workflow", "Complex dependencies", "Inefficient patterns"]
            }
        }
    
    def categorize_error(self, error_code: str, message: str, context: Dict[str, Any]) -> LintError:
        """Categorize a lint error"""
        error_def = self.error_registry.get(error_code, {})
        
        return LintError(
            error_code=error_code,
            category=error_def.get("category", ErrorCategory.SYNTAX),
            severity=error_def.get("severity", ErrorSeverity.ERROR),
            message=message,
            file_path=context.get("file_path", ""),
            line_number=context.get("line_number", 0),
            column_number=context.get("column_number", 0),
            rule_name=context.get("rule_name", ""),
            tenant_safe=self._is_tenant_safe(message)
        )
    
    def record_error(self, error: LintError, tenant_id: Optional[str] = None) -> None:
        """Record error for telemetry (tenant-safe)"""
        # Create anonymous tenant hash for counting
        tenant_hash = None
        if tenant_id:
            tenant_hash = hashlib.sha256(f"tenant_{tenant_id}".encode()).hexdigest()[:8]
        
        # Update metrics
        if error.error_code not in self.metrics_store:
            self.metrics_store[error.error_code] = ErrorMetrics(
                error_code=error.error_code,
                category=error.category.value,
                severity=error.severity.value,
                count=0,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow(),
                affected_tenants=0,
                common_patterns=[]
            )
        
        metrics = self.metrics_store[error.error_code]
        metrics.count += 1
        metrics.last_seen = datetime.utcnow()
        
        # Track unique tenants (anonymously)
        if tenant_hash and tenant_hash not in getattr(metrics, '_tenant_hashes', set()):
            if not hasattr(metrics, '_tenant_hashes'):
                metrics._tenant_hashes = set()
            metrics._tenant_hashes.add(tenant_hash)
            metrics.affected_tenants = len(metrics._tenant_hashes)
        
        # Extract patterns from error message (anonymized)
        pattern = self._extract_error_pattern(error.message)
        if pattern and pattern not in metrics.common_patterns:
            metrics.common_patterns.append(pattern)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get aggregated error statistics"""
        stats = {
            "total_errors": sum(m.count for m in self.metrics_store.values()),
            "by_category": {},
            "by_severity": {},
            "top_errors": [],
            "trends": {}
        }
        
        # Aggregate by category
        for metrics in self.metrics_store.values():
            category = metrics.category
            if category not in stats["by_category"]:
                stats["by_category"][category] = 0
            stats["by_category"][category] += metrics.count
        
        # Aggregate by severity
        for metrics in self.metrics_store.values():
            severity = metrics.severity
            if severity not in stats["by_severity"]:
                stats["by_severity"][severity] = 0
            stats["by_severity"][severity] += metrics.count
        
        # Top errors
        sorted_metrics = sorted(self.metrics_store.values(), key=lambda m: m.count, reverse=True)
        stats["top_errors"] = [
            {
                "error_code": m.error_code,
                "count": m.count,
                "category": m.category,
                "affected_tenants": m.affected_tenants
            }
            for m in sorted_metrics[:10]
        ]
        
        return stats
    
    def _is_tenant_safe(self, message: str) -> bool:
        """Check if error message is safe for telemetry"""
        sensitive_patterns = [
            "tenant_id", "user_id", "account_id", "email", "phone", "ssn",
            "password", "token", "key", "secret", "api_key"
        ]
        
        message_lower = message.lower()
        return not any(pattern in message_lower for pattern in sensitive_patterns)
    
    def _extract_error_pattern(self, message: str) -> Optional[str]:
        """Extract anonymized pattern from error message"""
        # Replace specific values with placeholders
        import re
        
        # Replace numbers with [NUMBER]
        pattern = re.sub(r'\d+', '[NUMBER]', message)
        
        # Replace quoted strings with [STRING]
        pattern = re.sub(r'"[^"]*"', '[STRING]', pattern)
        pattern = re.sub(r"'[^']*'", '[STRING]', pattern)
        
        # Replace file paths with [PATH]
        pattern = re.sub(r'/[^\s]+', '[PATH]', pattern)
        pattern = re.sub(r'[A-Za-z]:\\[^\s]+', '[PATH]', pattern)
        
        return pattern if pattern != message else None
    
    def export_telemetry(self) -> Dict[str, Any]:
        """Export telemetry data for analysis"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": self.get_error_statistics(),
            "error_registry_version": "1.0.0",
            "anonymized": True
        }
