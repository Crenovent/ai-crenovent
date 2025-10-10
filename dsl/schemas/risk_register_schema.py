# Risk Register Schema - Chapter 7.6
# Tasks 7.6-T01 to T50: Risk register, risk scoring, mitigation tracking

import json
import uuid
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import asyncio
import statistics

logger = logging.getLogger(__name__)

class RiskCategory(Enum):
    """Risk categories"""
    OPERATIONAL = "operational"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    FINANCIAL = "financial"
    REPUTATIONAL = "reputational"
    TECHNICAL = "technical"
    BUSINESS = "business"
    REGULATORY = "regulatory"

class RiskSeverity(Enum):
    """Risk severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RiskStatus(Enum):
    """Risk status"""
    IDENTIFIED = "identified"
    ASSESSED = "assessed"
    MITIGATING = "mitigating"
    MONITORING = "monitoring"
    CLOSED = "closed"
    ACCEPTED = "accepted"
    ESCALATED = "escalated"

class RiskLikelihood(Enum):
    """Risk likelihood"""
    VERY_LOW = "very_low"      # 0-10%
    LOW = "low"                # 10-30%
    MEDIUM = "medium"          # 30-60%
    HIGH = "high"              # 60-80%
    VERY_HIGH = "very_high"    # 80-100%

class RiskImpact(Enum):
    """Risk impact levels"""
    NEGLIGIBLE = "negligible"
    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CATASTROPHIC = "catastrophic"

class MitigationStatus(Enum):
    """Mitigation action status"""
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"

@dataclass
class RiskSource:
    """Risk source identification"""
    source_id: str
    source_type: str  # workflow, system, process, external
    source_name: str
    
    # Context
    workflow_id: Optional[str] = None
    execution_id: Optional[str] = None
    tenant_id: Optional[int] = None
    system_component: Optional[str] = None
    
    # Detection
    detected_by: str = "system"
    detection_method: str = "automated"
    detection_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class RiskAssessment:
    """Risk assessment details"""
    assessment_id: str
    assessed_by: str
    assessed_at: str
    
    # Likelihood and impact
    likelihood: RiskLikelihood
    impact: RiskImpact
    
    # Scoring
    likelihood_score: float  # 0.0-1.0
    impact_score: float      # 0.0-1.0
    risk_score: float        # likelihood * impact
    
    # Qualitative assessment
    assessment_notes: str
    assumptions: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Confidence
    confidence_level: float = 0.8  # 0.0-1.0
    assessment_method: str = "expert_judgment"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['likelihood'] = self.likelihood.value
        data['impact'] = self.impact.value
        return data

@dataclass
class MitigationAction:
    """Risk mitigation action"""
    action_id: str
    action_name: str
    description: str
    
    # Ownership
    owner: str
    assigned_to: str
    created_by: str
    created_at: str
    
    # Timeline
    planned_start_date: str
    planned_completion_date: str
    actual_start_date: Optional[str] = None
    actual_completion_date: Optional[str] = None
    
    # Status and progress
    status: MitigationStatus = MitigationStatus.PLANNED
    progress_percentage: int = 0
    
    # Resources
    estimated_effort_hours: int = 0
    actual_effort_hours: int = 0
    budget_allocated: float = 0.0
    budget_spent: float = 0.0
    
    # Effectiveness
    expected_risk_reduction: float = 0.0  # 0.0-1.0
    actual_risk_reduction: Optional[float] = None
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    
    # Updates
    status_updates: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['status'] = self.status.value
        return data

@dataclass
class RiskMonitoring:
    """Risk monitoring configuration"""
    monitoring_id: str
    risk_id: str
    
    # Monitoring schedule
    frequency_days: int = 7
    next_review_date: str = field(default_factory=lambda: (datetime.now(timezone.utc) + timedelta(days=7)).isoformat())
    last_review_date: Optional[str] = None
    
    # Monitoring criteria
    key_indicators: List[str] = field(default_factory=list)
    thresholds: Dict[str, float] = field(default_factory=dict)
    alert_conditions: List[str] = field(default_factory=list)
    
    # Monitoring results
    reviews_conducted: int = 0
    alerts_triggered: int = 0
    escalations_made: int = 0
    
    # Trend analysis
    risk_trend: str = "stable"  # improving, stable, degrading
    trend_confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class RiskRegisterEntry:
    """Complete risk register entry"""
    # Core identification
    risk_id: str
    risk_title: str
    risk_description: str
    
    # Classification
    category: RiskCategory
    severity: RiskSeverity
    
    # Source and context
    source: RiskSource
    
    # Assessment
    current_assessment: RiskAssessment
    assessment_history: List[RiskAssessment] = field(default_factory=list)
    
    # Mitigation
    mitigation_actions: List[MitigationAction] = field(default_factory=list)
    
    # Monitoring
    monitoring: Optional[RiskMonitoring] = None
    
    # Status and lifecycle
    status: RiskStatus = RiskStatus.IDENTIFIED
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Ownership
    risk_owner: str = ""
    business_owner: str = ""
    
    # Compliance and regulatory
    regulatory_requirements: List[str] = field(default_factory=list)
    compliance_frameworks: List[str] = field(default_factory=list)
    
    # Links to other systems
    related_risks: List[str] = field(default_factory=list)
    related_incidents: List[str] = field(default_factory=list)
    evidence_pack_ids: List[str] = field(default_factory=list)
    
    # Audit trail
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    
    # Cryptographic integrity
    entry_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['source'] = self.source.to_dict()
        data['current_assessment'] = self.current_assessment.to_dict()
        data['assessment_history'] = [a.to_dict() for a in self.assessment_history]
        data['mitigation_actions'] = [ma.to_dict() for ma in self.mitigation_actions]
        if self.monitoring:
            data['monitoring'] = self.monitoring.to_dict()
        return data
    
    def calculate_entry_hash(self) -> str:
        """Calculate hash for integrity verification"""
        
        hash_content = {
            'risk_id': self.risk_id,
            'risk_title': self.risk_title,
            'risk_description': self.risk_description,
            'category': self.category.value,
            'severity': self.severity.value,
            'source': self.source.to_dict(),
            'current_assessment': self.current_assessment.to_dict(),
            'status': self.status.value,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
        
        content_json = json.dumps(hash_content, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content_json.encode()).hexdigest()
    
    def add_audit_entry(self, action: str, actor: str, details: str) -> None:
        """Add audit trail entry"""
        
        audit_entry = {
            'audit_id': str(uuid.uuid4()),
            'action': action,
            'actor': actor,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': details
        }
        
        self.audit_trail.append(audit_entry)
        self.updated_at = audit_entry['timestamp']
    
    def get_current_risk_score(self) -> float:
        """Get current risk score"""
        return self.current_assessment.risk_score
    
    def get_residual_risk_score(self) -> float:
        """Calculate residual risk score after mitigation"""
        
        base_score = self.current_assessment.risk_score
        
        # Calculate total risk reduction from completed mitigations
        total_reduction = 0.0
        for action in self.mitigation_actions:
            if action.status == MitigationStatus.COMPLETED and action.actual_risk_reduction:
                total_reduction += action.actual_risk_reduction
            elif action.status == MitigationStatus.IN_PROGRESS and action.expected_risk_reduction:
                # Apply partial reduction based on progress
                partial_reduction = action.expected_risk_reduction * (action.progress_percentage / 100.0)
                total_reduction += partial_reduction
        
        # Cap total reduction at 95% (never completely eliminate risk)
        total_reduction = min(total_reduction, 0.95)
        
        return base_score * (1.0 - total_reduction)

class RiskRegisterManager:
    """
    Risk Register Manager - Tasks 7.6-T01 to T30
    Manages risk identification, assessment, mitigation, and monitoring
    """
    
    def __init__(self):
        self.risk_entries: Dict[str, RiskRegisterEntry] = {}
        self.risk_scoring_matrix: Dict[Tuple[RiskLikelihood, RiskImpact], float] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize risk scoring matrix
        self._initialize_risk_scoring_matrix()
    
    def _initialize_risk_scoring_matrix(self) -> None:
        """Initialize risk scoring matrix"""
        
        # Likelihood scores
        likelihood_scores = {
            RiskLikelihood.VERY_LOW: 0.05,
            RiskLikelihood.LOW: 0.2,
            RiskLikelihood.MEDIUM: 0.45,
            RiskLikelihood.HIGH: 0.7,
            RiskLikelihood.VERY_HIGH: 0.9
        }
        
        # Impact scores
        impact_scores = {
            RiskImpact.NEGLIGIBLE: 0.1,
            RiskImpact.MINOR: 0.3,
            RiskImpact.MODERATE: 0.5,
            RiskImpact.MAJOR: 0.7,
            RiskImpact.CATASTROPHIC: 0.9
        }
        
        # Create scoring matrix
        for likelihood in RiskLikelihood:
            for impact in RiskImpact:
                score = likelihood_scores[likelihood] * impact_scores[impact]
                self.risk_scoring_matrix[(likelihood, impact)] = score
        
        self.logger.info(f"✅ Initialized risk scoring matrix with {len(self.risk_scoring_matrix)} combinations")
    
    async def identify_risk(
        self,
        risk_title: str,
        risk_description: str,
        category: RiskCategory,
        source_type: str,
        source_name: str,
        detected_by: str,
        workflow_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        tenant_id: Optional[int] = None,
        regulatory_requirements: List[str] = None,
        compliance_frameworks: List[str] = None
    ) -> RiskRegisterEntry:
        """Identify new risk"""
        
        risk_id = str(uuid.uuid4())
        
        # Create risk source
        source = RiskSource(
            source_id=str(uuid.uuid4()),
            source_type=source_type,
            source_name=source_name,
            workflow_id=workflow_id,
            execution_id=execution_id,
            tenant_id=tenant_id,
            detected_by=detected_by
        )
        
        # Create initial assessment (placeholder)
        initial_assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessed_by="system",
            assessed_at=datetime.now(timezone.utc).isoformat(),
            likelihood=RiskLikelihood.MEDIUM,
            impact=RiskImpact.MODERATE,
            likelihood_score=0.45,
            impact_score=0.5,
            risk_score=0.225,
            assessment_notes="Initial system assessment - requires expert review"
        )
        
        # Determine initial severity based on risk score
        severity = self._calculate_severity(initial_assessment.risk_score)
        
        # Create risk register entry
        risk_entry = RiskRegisterEntry(
            risk_id=risk_id,
            risk_title=risk_title,
            risk_description=risk_description,
            category=category,
            severity=severity,
            source=source,
            current_assessment=initial_assessment,
            regulatory_requirements=regulatory_requirements or [],
            compliance_frameworks=compliance_frameworks or []
        )
        
        # Add initial audit entry
        risk_entry.add_audit_entry(
            action="risk_identified",
            actor=detected_by,
            details=f"Risk identified: {risk_title} ({category.value})"
        )
        
        # Calculate initial hash
        risk_entry.entry_hash = risk_entry.calculate_entry_hash()
        
        # Store in register
        self.risk_entries[risk_id] = risk_entry
        
        self.logger.info(f"✅ Identified risk: {risk_id} ({risk_title})")
        return risk_entry
    
    def _calculate_severity(self, risk_score: float) -> RiskSeverity:
        """Calculate severity based on risk score"""
        
        if risk_score >= 0.7:
            return RiskSeverity.CRITICAL
        elif risk_score >= 0.5:
            return RiskSeverity.HIGH
        elif risk_score >= 0.3:
            return RiskSeverity.MEDIUM
        else:
            return RiskSeverity.LOW
    
    async def assess_risk(
        self,
        risk_id: str,
        assessed_by: str,
        likelihood: RiskLikelihood,
        impact: RiskImpact,
        assessment_notes: str,
        assumptions: List[str] = None,
        dependencies: List[str] = None,
        confidence_level: float = 0.8,
        assessment_method: str = "expert_judgment"
    ) -> bool:
        """Perform detailed risk assessment"""
        
        risk_entry = self.risk_entries.get(risk_id)
        if not risk_entry:
            self.logger.error(f"❌ Risk {risk_id} not found")
            return False
        
        # Get scores from matrix
        likelihood_score = next(score for (l, i), score in self.risk_scoring_matrix.items() if l == likelihood and i == RiskImpact.MODERATE)
        impact_score = next(score for (l, i), score in self.risk_scoring_matrix.items() if l == RiskLikelihood.MEDIUM and i == impact)
        risk_score = self.risk_scoring_matrix.get((likelihood, impact), 0.5)
        
        # Create new assessment
        assessment = RiskAssessment(
            assessment_id=str(uuid.uuid4()),
            assessed_by=assessed_by,
            assessed_at=datetime.now(timezone.utc).isoformat(),
            likelihood=likelihood,
            impact=impact,
            likelihood_score=likelihood_score,
            impact_score=impact_score,
            risk_score=risk_score,
            assessment_notes=assessment_notes,
            assumptions=assumptions or [],
            dependencies=dependencies or [],
            confidence_level=confidence_level,
            assessment_method=assessment_method
        )
        
        # Move current assessment to history
        risk_entry.assessment_history.append(risk_entry.current_assessment)
        
        # Update current assessment
        risk_entry.current_assessment = assessment
        
        # Update severity
        risk_entry.severity = self._calculate_severity(risk_score)
        
        # Update status
        risk_entry.status = RiskStatus.ASSESSED
        
        # Add audit entry
        risk_entry.add_audit_entry(
            action="risk_assessed",
            actor=assessed_by,
            details=f"Risk assessed: {likelihood.value} likelihood, {impact.value} impact, score: {risk_score:.3f}"
        )
        
        # Update hash
        risk_entry.entry_hash = risk_entry.calculate_entry_hash()
        
        self.logger.info(f"✅ Assessed risk: {risk_id} (score: {risk_score:.3f})")
        return True
    
    async def add_mitigation_action(
        self,
        risk_id: str,
        action_name: str,
        description: str,
        owner: str,
        assigned_to: str,
        created_by: str,
        planned_start_date: str,
        planned_completion_date: str,
        estimated_effort_hours: int = 0,
        budget_allocated: float = 0.0,
        expected_risk_reduction: float = 0.0,
        dependencies: List[str] = None
    ) -> bool:
        """Add mitigation action to risk"""
        
        risk_entry = self.risk_entries.get(risk_id)
        if not risk_entry:
            self.logger.error(f"❌ Risk {risk_id} not found")
            return False
        
        action_id = str(uuid.uuid4())
        
        mitigation_action = MitigationAction(
            action_id=action_id,
            action_name=action_name,
            description=description,
            owner=owner,
            assigned_to=assigned_to,
            created_by=created_by,
            created_at=datetime.now(timezone.utc).isoformat(),
            planned_start_date=planned_start_date,
            planned_completion_date=planned_completion_date,
            estimated_effort_hours=estimated_effort_hours,
            budget_allocated=budget_allocated,
            expected_risk_reduction=expected_risk_reduction,
            dependencies=dependencies or []
        )
        
        risk_entry.mitigation_actions.append(mitigation_action)
        
        # Update risk status
        if risk_entry.status == RiskStatus.ASSESSED:
            risk_entry.status = RiskStatus.MITIGATING
        
        # Add audit entry
        risk_entry.add_audit_entry(
            action="mitigation_added",
            actor=created_by,
            details=f"Mitigation action added: {action_name} (owner: {owner})"
        )
        
        # Update hash
        risk_entry.entry_hash = risk_entry.calculate_entry_hash()
        
        self.logger.info(f"✅ Added mitigation action: {action_id} to risk {risk_id}")
        return True
    
    async def update_mitigation_progress(
        self,
        risk_id: str,
        action_id: str,
        status: MitigationStatus,
        progress_percentage: int,
        status_update: str,
        updated_by: str,
        actual_effort_hours: Optional[int] = None,
        budget_spent: Optional[float] = None,
        actual_risk_reduction: Optional[float] = None
    ) -> bool:
        """Update mitigation action progress"""
        
        risk_entry = self.risk_entries.get(risk_id)
        if not risk_entry:
            self.logger.error(f"❌ Risk {risk_id} not found")
            return False
        
        # Find mitigation action
        mitigation_action = None
        for action in risk_entry.mitigation_actions:
            if action.action_id == action_id:
                mitigation_action = action
                break
        
        if not mitigation_action:
            self.logger.error(f"❌ Mitigation action {action_id} not found")
            return False
        
        # Update action
        mitigation_action.status = status
        mitigation_action.progress_percentage = progress_percentage
        
        if actual_effort_hours is not None:
            mitigation_action.actual_effort_hours = actual_effort_hours
        
        if budget_spent is not None:
            mitigation_action.budget_spent = budget_spent
        
        if actual_risk_reduction is not None:
            mitigation_action.actual_risk_reduction = actual_risk_reduction
        
        # Add status update
        status_update_entry = {
            'update_id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'updated_by': updated_by,
            'status': status.value,
            'progress_percentage': progress_percentage,
            'notes': status_update
        }
        
        mitigation_action.status_updates.append(status_update_entry)
        
        # Update completion dates
        if status == MitigationStatus.IN_PROGRESS and not mitigation_action.actual_start_date:
            mitigation_action.actual_start_date = datetime.now(timezone.utc).isoformat()
        elif status == MitigationStatus.COMPLETED and not mitigation_action.actual_completion_date:
            mitigation_action.actual_completion_date = datetime.now(timezone.utc).isoformat()
        
        # Add audit entry
        risk_entry.add_audit_entry(
            action="mitigation_updated",
            actor=updated_by,
            details=f"Mitigation {action_id} updated: {status.value} ({progress_percentage}%)"
        )
        
        # Update hash
        risk_entry.entry_hash = risk_entry.calculate_entry_hash()
        
        self.logger.info(f"✅ Updated mitigation action: {action_id} ({status.value})")
        return True
    
    async def setup_risk_monitoring(
        self,
        risk_id: str,
        frequency_days: int = 7,
        key_indicators: List[str] = None,
        thresholds: Dict[str, float] = None,
        alert_conditions: List[str] = None
    ) -> bool:
        """Set up risk monitoring"""
        
        risk_entry = self.risk_entries.get(risk_id)
        if not risk_entry:
            self.logger.error(f"❌ Risk {risk_id} not found")
            return False
        
        monitoring_id = str(uuid.uuid4())
        
        monitoring = RiskMonitoring(
            monitoring_id=monitoring_id,
            risk_id=risk_id,
            frequency_days=frequency_days,
            key_indicators=key_indicators or [],
            thresholds=thresholds or {},
            alert_conditions=alert_conditions or []
        )
        
        risk_entry.monitoring = monitoring
        risk_entry.status = RiskStatus.MONITORING
        
        # Add audit entry
        risk_entry.add_audit_entry(
            action="monitoring_setup",
            actor="system",
            details=f"Risk monitoring setup with {frequency_days}-day frequency"
        )
        
        # Update hash
        risk_entry.entry_hash = risk_entry.calculate_entry_hash()
        
        self.logger.info(f"✅ Setup monitoring for risk: {risk_id}")
        return True
    
    def get_risk_entry(self, risk_id: str) -> Optional[RiskRegisterEntry]:
        """Get risk register entry"""
        return self.risk_entries.get(risk_id)
    
    def list_risks(
        self,
        category: Optional[RiskCategory] = None,
        severity: Optional[RiskSeverity] = None,
        status: Optional[RiskStatus] = None,
        tenant_id: Optional[int] = None,
        risk_owner: Optional[str] = None,
        limit: int = 100
    ) -> List[RiskRegisterEntry]:
        """List risks with filtering"""
        
        risks = list(self.risk_entries.values())
        
        if category:
            risks = [r for r in risks if r.category == category]
        
        if severity:
            risks = [r for r in risks if r.severity == severity]
        
        if status:
            risks = [r for r in risks if r.status == status]
        
        if tenant_id:
            risks = [r for r in risks if r.source.tenant_id == tenant_id]
        
        if risk_owner:
            risks = [r for r in risks if r.risk_owner == risk_owner]
        
        # Sort by risk score (highest first)
        risks.sort(key=lambda r: r.get_current_risk_score(), reverse=True)
        
        return risks[:limit]
    
    def get_high_priority_risks(self, limit: int = 20) -> List[RiskRegisterEntry]:
        """Get high priority risks"""
        
        high_priority = [
            risk for risk in self.risk_entries.values()
            if risk.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]
            and risk.status not in [RiskStatus.CLOSED, RiskStatus.ACCEPTED]
        ]
        
        # Sort by risk score
        high_priority.sort(key=lambda r: r.get_current_risk_score(), reverse=True)
        
        return high_priority[:limit]
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get risk dashboard data"""
        
        total_risks = len(self.risk_entries)
        
        risks_by_category = {}
        risks_by_severity = {}
        risks_by_status = {}
        
        risk_scores = []
        residual_risk_scores = []
        
        for risk in self.risk_entries.values():
            # By category
            category = risk.category.value
            risks_by_category[category] = risks_by_category.get(category, 0) + 1
            
            # By severity
            severity = risk.severity.value
            risks_by_severity[severity] = risks_by_severity.get(severity, 0) + 1
            
            # By status
            status = risk.status.value
            risks_by_status[status] = risks_by_status.get(status, 0) + 1
            
            # Risk scores
            risk_scores.append(risk.get_current_risk_score())
            residual_risk_scores.append(risk.get_residual_risk_score())
        
        # Calculate statistics
        avg_risk_score = statistics.mean(risk_scores) if risk_scores else 0.0
        avg_residual_score = statistics.mean(residual_risk_scores) if residual_risk_scores else 0.0
        risk_reduction = ((avg_risk_score - avg_residual_score) / avg_risk_score * 100) if avg_risk_score > 0 else 0.0
        
        return {
            'total_risks': total_risks,
            'risks_by_category': risks_by_category,
            'risks_by_severity': risks_by_severity,
            'risks_by_status': risks_by_status,
            'average_risk_score': round(avg_risk_score, 3),
            'average_residual_score': round(avg_residual_score, 3),
            'risk_reduction_percentage': round(risk_reduction, 1),
            'high_priority_count': len([r for r in self.risk_entries.values() if r.severity in [RiskSeverity.HIGH, RiskSeverity.CRITICAL]]),
            'active_mitigations': sum(len([a for a in r.mitigation_actions if a.status == MitigationStatus.IN_PROGRESS]) for r in self.risk_entries.values())
        }
    
    def verify_risk_integrity(self, risk_id: str) -> Dict[str, Any]:
        """Verify risk register entry integrity"""
        
        risk_entry = self.risk_entries.get(risk_id)
        if not risk_entry:
            return {'valid': False, 'error': 'Risk entry not found'}
        
        try:
            # Recalculate hash
            calculated_hash = risk_entry.calculate_entry_hash()
            stored_hash = risk_entry.entry_hash
            
            hash_valid = calculated_hash == stored_hash
            
            # Verify assessment consistency
            assessment_valid = True
            current_score = risk_entry.current_assessment.risk_score
            expected_score = self.risk_scoring_matrix.get(
                (risk_entry.current_assessment.likelihood, risk_entry.current_assessment.impact),
                0.5
            )
            
            if abs(current_score - expected_score) > 0.01:  # Allow small floating point differences
                assessment_valid = False
            
            return {
                'valid': hash_valid and assessment_valid,
                'hash_valid': hash_valid,
                'assessment_valid': assessment_valid,
                'calculated_hash': calculated_hash,
                'stored_hash': stored_hash,
                'audit_trail_length': len(risk_entry.audit_trail),
                'mitigation_count': len(risk_entry.mitigation_actions)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}

# Global instance
risk_register_manager = RiskRegisterManager()
