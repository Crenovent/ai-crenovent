"""
Task 7.3-T28: Add Template Trust Score (evidence, incidents, usage)
Quantified trust with governance metrics surfaced in UI
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import math

import asyncpg


class TrustFactor(Enum):
    """Trust scoring factors"""
    EXECUTION_SUCCESS_RATE = "execution_success_rate"
    GOVERNANCE_COMPLIANCE = "governance_compliance"
    SECURITY_POSTURE = "security_posture"
    USAGE_ADOPTION = "usage_adoption"
    INCIDENT_HISTORY = "incident_history"
    EVIDENCE_QUALITY = "evidence_quality"
    PEER_REVIEW_SCORE = "peer_review_score"
    MAINTENANCE_HEALTH = "maintenance_health"
    PERFORMANCE_METRICS = "performance_metrics"
    DOCUMENTATION_COMPLETENESS = "documentation_completeness"


class TrustLevel(Enum):
    """Trust level classifications"""
    UNTRUSTED = "untrusted"      # 0.0 - 0.3
    LOW = "low"                  # 0.3 - 0.5
    MEDIUM = "medium"            # 0.5 - 0.7
    HIGH = "high"                # 0.7 - 0.9
    EXCELLENT = "excellent"      # 0.9 - 1.0


class IncidentSeverity(Enum):
    """Incident severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TrustMetric:
    """Individual trust metric"""
    factor: TrustFactor
    score: float  # 0.0 - 1.0
    weight: float  # Importance weight
    evidence_count: int
    last_updated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def weighted_score(self) -> float:
        return self.score * self.weight
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "factor": self.factor.value,
            "score": self.score,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "evidence_count": self.evidence_count,
            "last_updated": self.last_updated.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class TrustIncident:
    """Trust-affecting incident"""
    incident_id: str
    workflow_id: str
    version_id: Optional[str]
    severity: IncidentSeverity
    incident_type: str
    description: str
    occurred_at: datetime
    resolved_at: Optional[datetime] = None
    impact_score: float = 0.0  # Negative impact on trust
    resolution_notes: Optional[str] = None
    
    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None
    
    @property
    def duration_hours(self) -> float:
        if self.resolved_at:
            return (self.resolved_at - self.occurred_at).total_seconds() / 3600
        return (datetime.now(timezone.utc) - self.occurred_at).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "severity": self.severity.value,
            "incident_type": self.incident_type,
            "description": self.description,
            "occurred_at": self.occurred_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "impact_score": self.impact_score,
            "duration_hours": self.duration_hours,
            "is_resolved": self.is_resolved,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class TrustScore:
    """Complete trust score for a workflow"""
    workflow_id: str
    version_id: Optional[str]
    overall_score: float  # 0.0 - 1.0
    trust_level: TrustLevel
    metrics: Dict[TrustFactor, TrustMetric]
    calculated_at: datetime
    expires_at: datetime
    confidence: float = 1.0  # Confidence in the score
    
    @property
    def is_expired(self) -> bool:
        return datetime.now(timezone.utc) > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "version_id": self.version_id,
            "overall_score": self.overall_score,
            "trust_level": self.trust_level.value,
            "confidence": self.confidence,
            "calculated_at": self.calculated_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_expired": self.is_expired,
            "metrics": {factor.value: metric.to_dict() for factor, metric in self.metrics.items()}
        }


class TrustScoringEngine:
    """Trust scoring engine for workflow templates"""
    
    def __init__(self, db_pool: Optional[asyncpg.Pool] = None, config: Optional[Dict[str, Any]] = None):
        self.db_pool = db_pool
        self.config = config or self._get_default_config()
        self.trust_cache: Dict[str, TrustScore] = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default trust scoring configuration"""
        return {
            "cache_ttl_hours": 24,
            "min_executions_for_reliability": 10,
            "incident_decay_days": 90,
            "weights": {
                TrustFactor.EXECUTION_SUCCESS_RATE: 0.25,
                TrustFactor.GOVERNANCE_COMPLIANCE: 0.20,
                TrustFactor.SECURITY_POSTURE: 0.15,
                TrustFactor.USAGE_ADOPTION: 0.10,
                TrustFactor.INCIDENT_HISTORY: 0.10,
                TrustFactor.EVIDENCE_QUALITY: 0.08,
                TrustFactor.PEER_REVIEW_SCORE: 0.05,
                TrustFactor.MAINTENANCE_HEALTH: 0.04,
                TrustFactor.PERFORMANCE_METRICS: 0.02,
                TrustFactor.DOCUMENTATION_COMPLETENESS: 0.01
            },
            "thresholds": {
                TrustLevel.UNTRUSTED: 0.0,
                TrustLevel.LOW: 0.3,
                TrustLevel.MEDIUM: 0.5,
                TrustLevel.HIGH: 0.7,
                TrustLevel.EXCELLENT: 0.9
            },
            "incident_impact": {
                IncidentSeverity.CRITICAL: -0.3,
                IncidentSeverity.HIGH: -0.2,
                IncidentSeverity.MEDIUM: -0.1,
                IncidentSeverity.LOW: -0.05,
                IncidentSeverity.INFO: -0.01
            }
        }
    
    async def calculate_trust_score(
        self,
        workflow_id: str,
        version_id: Optional[str] = None,
        force_recalculate: bool = False
    ) -> TrustScore:
        """Calculate comprehensive trust score for a workflow"""
        
        cache_key = f"{workflow_id}:{version_id or 'latest'}"
        
        # Check cache first
        if not force_recalculate and cache_key in self.trust_cache:
            cached_score = self.trust_cache[cache_key]
            if not cached_score.is_expired:
                return cached_score
        
        # Calculate individual metrics
        metrics = {}
        
        # Execution success rate
        metrics[TrustFactor.EXECUTION_SUCCESS_RATE] = await self._calculate_execution_success_rate(workflow_id, version_id)
        
        # Governance compliance
        metrics[TrustFactor.GOVERNANCE_COMPLIANCE] = await self._calculate_governance_compliance(workflow_id, version_id)
        
        # Security posture
        metrics[TrustFactor.SECURITY_POSTURE] = await self._calculate_security_posture(workflow_id, version_id)
        
        # Usage adoption
        metrics[TrustFactor.USAGE_ADOPTION] = await self._calculate_usage_adoption(workflow_id, version_id)
        
        # Incident history
        metrics[TrustFactor.INCIDENT_HISTORY] = await self._calculate_incident_impact(workflow_id, version_id)
        
        # Evidence quality
        metrics[TrustFactor.EVIDENCE_QUALITY] = await self._calculate_evidence_quality(workflow_id, version_id)
        
        # Peer review score
        metrics[TrustFactor.PEER_REVIEW_SCORE] = await self._calculate_peer_review_score(workflow_id, version_id)
        
        # Maintenance health
        metrics[TrustFactor.MAINTENANCE_HEALTH] = await self._calculate_maintenance_health(workflow_id, version_id)
        
        # Performance metrics
        metrics[TrustFactor.PERFORMANCE_METRICS] = await self._calculate_performance_metrics(workflow_id, version_id)
        
        # Documentation completeness
        metrics[TrustFactor.DOCUMENTATION_COMPLETENESS] = await self._calculate_documentation_completeness(workflow_id, version_id)
        
        # Calculate overall score
        overall_score = sum(metric.weighted_score for metric in metrics.values())
        overall_score = max(0.0, min(1.0, overall_score))  # Clamp to [0, 1]
        
        # Determine trust level
        trust_level = self._determine_trust_level(overall_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(metrics)
        
        # Create trust score
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self.config["cache_ttl_hours"])
        
        trust_score = TrustScore(
            workflow_id=workflow_id,
            version_id=version_id,
            overall_score=overall_score,
            trust_level=trust_level,
            metrics=metrics,
            calculated_at=now,
            expires_at=expires_at,
            confidence=confidence
        )
        
        # Cache the result
        self.trust_cache[cache_key] = trust_score
        
        # Store in database if available
        if self.db_pool:
            await self._store_trust_score(trust_score)
        
        return trust_score
    
    async def _calculate_execution_success_rate(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate execution success rate"""
        
        if not self.db_pool:
            # Default score when no data available
            return TrustMetric(
                factor=TrustFactor.EXECUTION_SUCCESS_RATE,
                score=0.5,
                weight=self.config["weights"][TrustFactor.EXECUTION_SUCCESS_RATE],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
        
        # Query execution traces
        query = """
            SELECT 
                COUNT(*) as total_executions,
                COUNT(*) FILTER (WHERE status = 'success') as successful_executions,
                AVG(CASE WHEN status = 'success' THEN 1.0 ELSE 0.0 END) as success_rate
            FROM dsl_execution_traces 
            WHERE workflow_id = $1
            AND ($2 IS NULL OR version_id = $2)
            AND created_at >= NOW() - INTERVAL '30 days'
        """
        
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(query, workflow_id, version_id)
                
                total_executions = row["total_executions"] or 0
                success_rate = row["success_rate"] or 0.0
                
                # Adjust score based on execution volume
                min_executions = self.config["min_executions_for_reliability"]
                volume_factor = min(1.0, total_executions / min_executions) if min_executions > 0 else 1.0
                adjusted_score = success_rate * volume_factor
                
                return TrustMetric(
                    factor=TrustFactor.EXECUTION_SUCCESS_RATE,
                    score=adjusted_score,
                    weight=self.config["weights"][TrustFactor.EXECUTION_SUCCESS_RATE],
                    evidence_count=total_executions,
                    last_updated=datetime.now(timezone.utc),
                    metadata={
                        "raw_success_rate": success_rate,
                        "total_executions": total_executions,
                        "volume_factor": volume_factor
                    }
                )
        except Exception:
            return TrustMetric(
                factor=TrustFactor.EXECUTION_SUCCESS_RATE,
                score=0.5,
                weight=self.config["weights"][TrustFactor.EXECUTION_SUCCESS_RATE],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_governance_compliance(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate governance compliance score"""
        
        if not self.db_pool:
            return TrustMetric(
                factor=TrustFactor.GOVERNANCE_COMPLIANCE,
                score=0.7,  # Default assumption of good compliance
                weight=self.config["weights"][TrustFactor.GOVERNANCE_COMPLIANCE],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
        
        try:
            # Check policy compliance
            policy_query = """
                SELECT 
                    COUNT(*) as total_policy_checks,
                    COUNT(*) FILTER (WHERE compliance_status = 'compliant') as compliant_checks,
                    AVG(CASE WHEN compliance_status = 'compliant' THEN 1.0 ELSE 0.0 END) as compliance_rate
                FROM dsl_execution_traces t
                JOIN dsl_policy_packs p ON t.policy_pack_id = p.policy_pack_id
                WHERE t.workflow_id = $1
                AND ($2 IS NULL OR t.version_id = $2)
                AND t.created_at >= NOW() - INTERVAL '30 days'
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(policy_query, workflow_id, version_id)
                
                compliance_rate = row["compliance_rate"] or 0.8  # Default to good compliance
                total_checks = row["total_policy_checks"] or 0
                
                return TrustMetric(
                    factor=TrustFactor.GOVERNANCE_COMPLIANCE,
                    score=compliance_rate,
                    weight=self.config["weights"][TrustFactor.GOVERNANCE_COMPLIANCE],
                    evidence_count=total_checks,
                    last_updated=datetime.now(timezone.utc),
                    metadata={"compliance_rate": compliance_rate}
                )
        except Exception:
            return TrustMetric(
                factor=TrustFactor.GOVERNANCE_COMPLIANCE,
                score=0.7,
                weight=self.config["weights"][TrustFactor.GOVERNANCE_COMPLIANCE],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_security_posture(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate security posture score"""
        
        # This would integrate with security scanning results
        # For now, return a baseline score
        return TrustMetric(
            factor=TrustFactor.SECURITY_POSTURE,
            score=0.8,  # Assume good security by default
            weight=self.config["weights"][TrustFactor.SECURITY_POSTURE],
            evidence_count=1,
            last_updated=datetime.now(timezone.utc),
            metadata={"scan_status": "baseline"}
        )
    
    async def _calculate_usage_adoption(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate usage adoption score"""
        
        if not self.db_pool:
            return TrustMetric(
                factor=TrustFactor.USAGE_ADOPTION,
                score=0.3,
                weight=self.config["weights"][TrustFactor.USAGE_ADOPTION],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
        
        try:
            # Count unique users and tenants
            usage_query = """
                SELECT 
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(DISTINCT tenant_id) as unique_tenants,
                    COUNT(*) as total_executions
                FROM dsl_execution_traces 
                WHERE workflow_id = $1
                AND ($2 IS NULL OR version_id = $2)
                AND created_at >= NOW() - INTERVAL '30 days'
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(usage_query, workflow_id, version_id)
                
                unique_users = row["unique_users"] or 0
                unique_tenants = row["unique_tenants"] or 0
                total_executions = row["total_executions"] or 0
                
                # Calculate adoption score based on usage breadth and frequency
                user_score = min(1.0, unique_users / 10.0)  # Normalize to 10 users
                tenant_score = min(1.0, unique_tenants / 5.0)  # Normalize to 5 tenants
                frequency_score = min(1.0, total_executions / 100.0)  # Normalize to 100 executions
                
                adoption_score = (user_score + tenant_score + frequency_score) / 3.0
                
                return TrustMetric(
                    factor=TrustFactor.USAGE_ADOPTION,
                    score=adoption_score,
                    weight=self.config["weights"][TrustFactor.USAGE_ADOPTION],
                    evidence_count=total_executions,
                    last_updated=datetime.now(timezone.utc),
                    metadata={
                        "unique_users": unique_users,
                        "unique_tenants": unique_tenants,
                        "total_executions": total_executions
                    }
                )
        except Exception:
            return TrustMetric(
                factor=TrustFactor.USAGE_ADOPTION,
                score=0.3,
                weight=self.config["weights"][TrustFactor.USAGE_ADOPTION],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_incident_impact(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate incident history impact on trust"""
        
        # Start with perfect score
        base_score = 1.0
        incident_count = 0
        
        if self.db_pool:
            try:
                # Query recent incidents
                incident_query = """
                    SELECT severity, COUNT(*) as incident_count
                    FROM workflow_incidents 
                    WHERE workflow_id = $1
                    AND ($2 IS NULL OR version_id = $2)
                    AND occurred_at >= NOW() - INTERVAL '90 days'
                    GROUP BY severity
                """
                
                async with self.db_pool.acquire() as conn:
                    rows = await conn.fetch(incident_query, workflow_id, version_id)
                    
                    for row in rows:
                        severity = IncidentSeverity(row["severity"])
                        count = row["incident_count"]
                        impact = self.config["incident_impact"][severity]
                        
                        # Apply incident impact with decay
                        base_score += impact * count * 0.5  # 50% impact decay
                        incident_count += count
            except Exception:
                pass
        
        # Ensure score stays within bounds
        final_score = max(0.0, min(1.0, base_score))
        
        return TrustMetric(
            factor=TrustFactor.INCIDENT_HISTORY,
            score=final_score,
            weight=self.config["weights"][TrustFactor.INCIDENT_HISTORY],
            evidence_count=incident_count,
            last_updated=datetime.now(timezone.utc),
            metadata={"base_score": base_score, "incident_count": incident_count}
        )
    
    async def _calculate_evidence_quality(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate evidence pack quality score"""
        
        if not self.db_pool:
            return TrustMetric(
                factor=TrustFactor.EVIDENCE_QUALITY,
                score=0.6,
                weight=self.config["weights"][TrustFactor.EVIDENCE_QUALITY],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
        
        try:
            # Check evidence pack completeness and quality
            evidence_query = """
                SELECT 
                    COUNT(*) as total_evidence_packs,
                    AVG(CASE WHEN completeness_score >= 0.8 THEN 1.0 ELSE 0.0 END) as high_quality_rate,
                    AVG(completeness_score) as avg_completeness
                FROM dsl_evidence_packs 
                WHERE workflow_id = $1
                AND ($2 IS NULL OR version_id = $2)
                AND created_at >= NOW() - INTERVAL '30 days'
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(evidence_query, workflow_id, version_id)
                
                total_packs = row["total_evidence_packs"] or 0
                high_quality_rate = row["high_quality_rate"] or 0.6
                avg_completeness = row["avg_completeness"] or 0.6
                
                # Combine quality metrics
                quality_score = (high_quality_rate + avg_completeness) / 2.0
                
                return TrustMetric(
                    factor=TrustFactor.EVIDENCE_QUALITY,
                    score=quality_score,
                    weight=self.config["weights"][TrustFactor.EVIDENCE_QUALITY],
                    evidence_count=total_packs,
                    last_updated=datetime.now(timezone.utc),
                    metadata={
                        "high_quality_rate": high_quality_rate,
                        "avg_completeness": avg_completeness
                    }
                )
        except Exception:
            return TrustMetric(
                factor=TrustFactor.EVIDENCE_QUALITY,
                score=0.6,
                weight=self.config["weights"][TrustFactor.EVIDENCE_QUALITY],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_peer_review_score(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate peer review score"""
        
        # This would integrate with code review systems
        # For now, return a baseline score
        return TrustMetric(
            factor=TrustFactor.PEER_REVIEW_SCORE,
            score=0.7,
            weight=self.config["weights"][TrustFactor.PEER_REVIEW_SCORE],
            evidence_count=1,
            last_updated=datetime.now(timezone.utc),
            metadata={"review_status": "baseline"}
        )
    
    async def _calculate_maintenance_health(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate maintenance health score"""
        
        # This would check update frequency, bug fixes, etc.
        # For now, return a baseline score
        return TrustMetric(
            factor=TrustFactor.MAINTENANCE_HEALTH,
            score=0.8,
            weight=self.config["weights"][TrustFactor.MAINTENANCE_HEALTH],
            evidence_count=1,
            last_updated=datetime.now(timezone.utc),
            metadata={"maintenance_status": "active"}
        )
    
    async def _calculate_performance_metrics(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate performance metrics score"""
        
        if not self.db_pool:
            return TrustMetric(
                factor=TrustFactor.PERFORMANCE_METRICS,
                score=0.7,
                weight=self.config["weights"][TrustFactor.PERFORMANCE_METRICS],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
        
        try:
            # Check execution performance
            perf_query = """
                SELECT 
                    AVG(execution_duration_ms) as avg_duration,
                    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_duration_ms) as p95_duration,
                    COUNT(*) as execution_count
                FROM dsl_execution_traces 
                WHERE workflow_id = $1
                AND ($2 IS NULL OR version_id = $2)
                AND status = 'success'
                AND created_at >= NOW() - INTERVAL '30 days'
            """
            
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(perf_query, workflow_id, version_id)
                
                avg_duration = row["avg_duration"] or 5000  # 5 seconds default
                p95_duration = row["p95_duration"] or 10000  # 10 seconds default
                execution_count = row["execution_count"] or 0
                
                # Score based on performance (lower is better)
                # Assume 1 second is excellent, 10 seconds is poor
                avg_score = max(0.0, 1.0 - (avg_duration - 1000) / 9000)
                p95_score = max(0.0, 1.0 - (p95_duration - 1000) / 19000)
                
                performance_score = (avg_score + p95_score) / 2.0
                
                return TrustMetric(
                    factor=TrustFactor.PERFORMANCE_METRICS,
                    score=performance_score,
                    weight=self.config["weights"][TrustFactor.PERFORMANCE_METRICS],
                    evidence_count=execution_count,
                    last_updated=datetime.now(timezone.utc),
                    metadata={
                        "avg_duration_ms": avg_duration,
                        "p95_duration_ms": p95_duration
                    }
                )
        except Exception:
            return TrustMetric(
                factor=TrustFactor.PERFORMANCE_METRICS,
                score=0.7,
                weight=self.config["weights"][TrustFactor.PERFORMANCE_METRICS],
                evidence_count=0,
                last_updated=datetime.now(timezone.utc)
            )
    
    async def _calculate_documentation_completeness(self, workflow_id: str, version_id: Optional[str]) -> TrustMetric:
        """Calculate documentation completeness score"""
        
        # This would check for README, examples, API docs, etc.
        # For now, return a baseline score
        return TrustMetric(
            factor=TrustFactor.DOCUMENTATION_COMPLETENESS,
            score=0.6,
            weight=self.config["weights"][TrustFactor.DOCUMENTATION_COMPLETENESS],
            evidence_count=1,
            last_updated=datetime.now(timezone.utc),
            metadata={"doc_status": "basic"}
        )
    
    def _determine_trust_level(self, score: float) -> TrustLevel:
        """Determine trust level from score"""
        thresholds = self.config["thresholds"]
        
        if score >= thresholds[TrustLevel.EXCELLENT]:
            return TrustLevel.EXCELLENT
        elif score >= thresholds[TrustLevel.HIGH]:
            return TrustLevel.HIGH
        elif score >= thresholds[TrustLevel.MEDIUM]:
            return TrustLevel.MEDIUM
        elif score >= thresholds[TrustLevel.LOW]:
            return TrustLevel.LOW
        else:
            return TrustLevel.UNTRUSTED
    
    def _calculate_confidence(self, metrics: Dict[TrustFactor, TrustMetric]) -> float:
        """Calculate confidence in the trust score"""
        
        total_evidence = sum(metric.evidence_count for metric in metrics.values())
        
        # Confidence increases with evidence volume
        if total_evidence >= 100:
            return 1.0
        elif total_evidence >= 50:
            return 0.9
        elif total_evidence >= 20:
            return 0.8
        elif total_evidence >= 10:
            return 0.7
        elif total_evidence >= 5:
            return 0.6
        else:
            return 0.5
    
    async def _store_trust_score(self, trust_score: TrustScore) -> None:
        """Store trust score in database"""
        
        if not self.db_pool:
            return
        
        try:
            insert_query = """
                INSERT INTO workflow_trust_scores (
                    workflow_id, version_id, overall_score, trust_level,
                    confidence, metrics_data, calculated_at, expires_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (workflow_id, COALESCE(version_id, ''))
                DO UPDATE SET
                    overall_score = EXCLUDED.overall_score,
                    trust_level = EXCLUDED.trust_level,
                    confidence = EXCLUDED.confidence,
                    metrics_data = EXCLUDED.metrics_data,
                    calculated_at = EXCLUDED.calculated_at,
                    expires_at = EXCLUDED.expires_at
            """
            
            metrics_json = json.dumps({
                factor.value: metric.to_dict() 
                for factor, metric in trust_score.metrics.items()
            })
            
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    insert_query,
                    trust_score.workflow_id,
                    trust_score.version_id,
                    trust_score.overall_score,
                    trust_score.trust_level.value,
                    trust_score.confidence,
                    metrics_json,
                    trust_score.calculated_at,
                    trust_score.expires_at
                )
        except Exception as e:
            # Log error but don't fail the trust calculation
            pass
    
    async def record_incident(
        self,
        workflow_id: str,
        severity: IncidentSeverity,
        incident_type: str,
        description: str,
        version_id: Optional[str] = None
    ) -> TrustIncident:
        """Record a trust-affecting incident"""
        
        incident = TrustIncident(
            incident_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            version_id=version_id,
            severity=severity,
            incident_type=incident_type,
            description=description,
            occurred_at=datetime.now(timezone.utc),
            impact_score=abs(self.config["incident_impact"][severity])
        )
        
        # Store in database if available
        if self.db_pool:
            try:
                insert_query = """
                    INSERT INTO workflow_incidents (
                        incident_id, workflow_id, version_id, severity,
                        incident_type, description, occurred_at, impact_score
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """
                
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        insert_query,
                        incident.incident_id,
                        incident.workflow_id,
                        incident.version_id,
                        incident.severity.value,
                        incident.incident_type,
                        incident.description,
                        incident.occurred_at,
                        incident.impact_score
                    )
            except Exception:
                pass
        
        # Invalidate trust score cache
        cache_key = f"{workflow_id}:{version_id or 'latest'}"
        if cache_key in self.trust_cache:
            del self.trust_cache[cache_key]
        
        return incident
    
    async def get_trust_trends(
        self,
        workflow_id: str,
        days: int = 30,
        version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get trust score trends over time"""
        
        if not self.db_pool:
            return {"error": "Database not available"}
        
        try:
            trend_query = """
                SELECT 
                    DATE_TRUNC('day', calculated_at) as date,
                    AVG(overall_score) as avg_score,
                    AVG(confidence) as avg_confidence,
                    COUNT(*) as calculations
                FROM workflow_trust_scores 
                WHERE workflow_id = $1
                AND ($2 IS NULL OR version_id = $2)
                AND calculated_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE_TRUNC('day', calculated_at)
                ORDER BY date
            """ % days
            
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(trend_query, workflow_id, version_id)
                
                return {
                    "workflow_id": workflow_id,
                    "version_id": version_id,
                    "period_days": days,
                    "data_points": len(rows),
                    "trends": [
                        {
                            "date": row["date"].isoformat(),
                            "avg_score": float(row["avg_score"]),
                            "avg_confidence": float(row["avg_confidence"]),
                            "calculations": row["calculations"]
                        }
                        for row in rows
                    ]
                }
        except Exception as e:
            return {"error": str(e)}


# Database schema for trust scoring
TRUST_SCORING_SCHEMA_SQL = """
-- Workflow trust scores
CREATE TABLE IF NOT EXISTS workflow_trust_scores (
    score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    version_id UUID,
    overall_score DECIMAL(3,2) NOT NULL CHECK (overall_score >= 0 AND overall_score <= 1),
    trust_level VARCHAR(20) NOT NULL,
    confidence DECIMAL(3,2) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    metrics_data JSONB NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    
    CONSTRAINT uq_workflow_trust_score UNIQUE (workflow_id, COALESCE(version_id, '')),
    CONSTRAINT chk_trust_level CHECK (trust_level IN ('untrusted', 'low', 'medium', 'high', 'excellent'))
);

-- Workflow incidents
CREATE TABLE IF NOT EXISTS workflow_incidents (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id UUID NOT NULL,
    version_id UUID,
    severity VARCHAR(20) NOT NULL,
    incident_type VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    impact_score DECIMAL(3,2) NOT NULL DEFAULT 0,
    resolution_notes TEXT,
    
    CONSTRAINT chk_incident_severity CHECK (severity IN ('critical', 'high', 'medium', 'low', 'info'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_trust_scores_workflow ON workflow_trust_scores (workflow_id, calculated_at DESC);
CREATE INDEX IF NOT EXISTS idx_trust_scores_level ON workflow_trust_scores (trust_level, overall_score DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_workflow ON workflow_incidents (workflow_id, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON workflow_incidents (severity, occurred_at DESC);
"""
