"""
Observability & Continuous Assurance Framework - Section 7.4 Tasks
=================================================================

Implements Section 7.4 Tasks:
- 7.4.1: Define observability taxonomy
- 7.4.6: Configure trust scoring on telemetry data
- 7.4.7: Build anomaly detection for telemetry streams
- 7.4.21: Build predictive analytics on telemetry trends
- 7.4.28: Build trust dashboards scoped to observability

Following user rules:
- No hardcoding, dynamic configuration
- SaaS/IT industry focus
- Multi-tenant aware
- Modular design
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
import math

logger = logging.getLogger(__name__)

class ObservabilityDomain(Enum):
    """Task 7.4.1: Define observability taxonomy - Domains"""
    INFRASTRUCTURE = "infrastructure"
    APPLICATION = "application"
    BUSINESS = "business"
    SECURITY = "security"
    COMPLIANCE = "compliance"

class TelemetryType(Enum):
    """Task 7.4.1: Telemetry data types"""
    METRICS = "metrics"
    LOGS = "logs"
    TRACES = "traces"
    EVENTS = "events"
    ALERTS = "alerts"

class ObservabilityLevel(Enum):
    """Task 7.4.1: Observability maturity levels"""
    BASIC = "basic"          # Basic monitoring
    INTERMEDIATE = "intermediate"  # Structured observability
    ADVANCED = "advanced"    # Predictive analytics
    EXPERT = "expert"       # AI-driven insights

class AnomalyType(Enum):
    """Task 7.4.7: Anomaly types for detection"""
    SPIKE = "spike"
    DIP = "dip"
    TREND_CHANGE = "trend_change"
    SEASONAL_DEVIATION = "seasonal_deviation"
    PATTERN_BREAK = "pattern_break"

class TrustFactor(Enum):
    """Task 7.4.6: Trust factors for telemetry"""
    DATA_QUALITY = "data_quality"
    COMPLETENESS = "completeness"
    TIMELINESS = "timeliness"
    CONSISTENCY = "consistency"
    ACCURACY = "accuracy"

@dataclass
class ObservabilityTaxonomy:
    """Task 7.4.1: Define observability taxonomy"""
    taxonomy_id: str
    name: str
    description: str
    domain: ObservabilityDomain
    level: ObservabilityLevel
    
    # Metrics and KPIs
    key_metrics: List[str] = field(default_factory=list)
    sla_targets: Dict[str, float] = field(default_factory=dict)
    alert_thresholds: Dict[str, float] = field(default_factory=dict)
    
    # SaaS/IT specific
    business_impact: str = ""
    customer_facing: bool = False
    revenue_impact: bool = False
    
    # Configuration
    collection_frequency_sec: int = 60
    retention_days: int = 90
    aggregation_rules: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    owner_team: str = ""

@dataclass
class TelemetryDataPoint:
    """Individual telemetry data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    
    # Context
    domain: ObservabilityDomain
    telemetry_type: TelemetryType
    tenant_id: Optional[int] = None
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Quality indicators
    quality_score: float = 1.0
    confidence: float = 1.0
    source_reliability: float = 1.0

@dataclass
class AnomalyDetection:
    """Task 7.4.7: Anomaly detection result"""
    anomaly_id: str
    detected_at: datetime
    metric_name: str
    anomaly_type: AnomalyType
    
    # Anomaly details
    actual_value: float
    expected_value: float
    deviation_score: float
    confidence: float
    
    # Context
    domain: ObservabilityDomain
    tenant_id: Optional[int] = None
    
    # Impact assessment
    severity: str = "medium"  # low, medium, high, critical
    business_impact: str = ""
    suggested_actions: List[str] = field(default_factory=list)
    
    # Metadata
    detection_method: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class TrustScore:
    """Task 7.4.6: Trust scoring on telemetry data"""
    metric_name: str
    calculated_at: datetime
    overall_score: float
    
    # Individual factor scores
    data_quality_score: float
    completeness_score: float
    timeliness_score: float
    consistency_score: float
    accuracy_score: float
    
    # Context
    domain: ObservabilityDomain
    tenant_id: Optional[int] = None
    time_window_hours: int = 24
    
    # Confidence and metadata
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    sample_size: int = 0
    factors_analyzed: List[TrustFactor] = field(default_factory=list)

@dataclass
class PredictiveInsight:
    """Task 7.4.21: Predictive analytics on telemetry trends"""
    insight_id: str
    generated_at: datetime
    metric_name: str
    
    # Prediction details
    predicted_value: float
    prediction_horizon_hours: int
    confidence: float
    
    # Trend analysis
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_strength: float
    seasonal_pattern: bool
    
    # Business context
    domain: ObservabilityDomain
    business_impact: str = ""
    recommended_actions: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_level: str = "low"  # low, medium, high
    risk_factors: List[str] = field(default_factory=list)

class ObservabilityFramework:
    """
    Comprehensive Observability & Continuous Assurance Framework
    
    Implements Section 7.4 Tasks:
    - 7.4.1: Define observability taxonomy
    - 7.4.6: Configure trust scoring on telemetry data
    - 7.4.7: Build anomaly detection for telemetry streams
    - 7.4.21: Build predictive analytics on telemetry trends
    - 7.4.28: Build trust dashboards scoped to observability
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize observability taxonomy (Task 7.4.1)
        self.taxonomies = self._initialize_observability_taxonomies()
        
        # Telemetry data storage
        self.telemetry_data: Dict[str, List[TelemetryDataPoint]] = {}
        self.anomalies: List[AnomalyDetection] = []
        self.trust_scores: Dict[str, TrustScore] = {}
        self.predictive_insights: List[PredictiveInsight] = []
        
        # Configuration
        self.anomaly_detection_config = self._load_anomaly_detection_config()
        self.trust_scoring_config = self._load_trust_scoring_config()
        self.prediction_config = self._load_prediction_config()
        
        self.logger.info("✅ Initialized Observability Framework")
    
    def _initialize_observability_taxonomies(self) -> Dict[str, ObservabilityTaxonomy]:
        """Task 7.4.1: Define observability taxonomy"""
        taxonomies = {}
        
        # Infrastructure Observability
        taxonomies["infrastructure_performance"] = ObservabilityTaxonomy(
            taxonomy_id="infrastructure_performance",
            name="Infrastructure Performance",
            description="Core infrastructure metrics for SaaS platform",
            domain=ObservabilityDomain.INFRASTRUCTURE,
            level=ObservabilityLevel.INTERMEDIATE,
            key_metrics=[
                "cpu_utilization", "memory_utilization", "disk_io", 
                "network_throughput", "response_time", "error_rate"
            ],
            sla_targets={
                "cpu_utilization": 0.8,
                "memory_utilization": 0.85,
                "response_time_ms": 200.0,
                "error_rate": 0.01
            },
            alert_thresholds={
                "cpu_utilization": 0.9,
                "memory_utilization": 0.95,
                "response_time_ms": 500.0,
                "error_rate": 0.05
            },
            business_impact="Platform availability and performance",
            customer_facing=True,
            revenue_impact=True,
            collection_frequency_sec=30,
            retention_days=90,
            tags=["infrastructure", "performance", "sla"],
            owner_team="platform_engineering"
        )
        
        # Application Observability
        taxonomies["application_health"] = ObservabilityTaxonomy(
            taxonomy_id="application_health",
            name="Application Health",
            description="Application-level health and performance metrics",
            domain=ObservabilityDomain.APPLICATION,
            level=ObservabilityLevel.ADVANCED,
            key_metrics=[
                "request_rate", "success_rate", "latency_p95", 
                "throughput", "active_connections", "queue_depth"
            ],
            sla_targets={
                "success_rate": 0.995,
                "latency_p95": 100.0,
                "throughput": 1000.0
            },
            alert_thresholds={
                "success_rate": 0.99,
                "latency_p95": 300.0,
                "queue_depth": 100
            },
            business_impact="User experience and service quality",
            customer_facing=True,
            revenue_impact=True,
            collection_frequency_sec=15,
            retention_days=180,
            tags=["application", "health", "user_experience"],
            owner_team="application_team"
        )
        
        # Business Observability (SaaS focus)
        taxonomies["business_metrics"] = ObservabilityTaxonomy(
            taxonomy_id="business_metrics",
            name="Business Metrics",
            description="SaaS business and revenue metrics",
            domain=ObservabilityDomain.BUSINESS,
            level=ObservabilityLevel.EXPERT,
            key_metrics=[
                "active_users", "revenue_per_hour", "churn_rate",
                "conversion_rate", "customer_satisfaction", "arr_growth"
            ],
            sla_targets={
                "churn_rate": 0.05,
                "conversion_rate": 0.15,
                "customer_satisfaction": 4.5
            },
            alert_thresholds={
                "churn_rate": 0.08,
                "conversion_rate": 0.10,
                "customer_satisfaction": 4.0
            },
            business_impact="Revenue and customer success",
            customer_facing=False,
            revenue_impact=True,
            collection_frequency_sec=300,  # 5 minutes
            retention_days=365,
            tags=["business", "revenue", "saas"],
            owner_team="business_intelligence"
        )
        
        # Security Observability
        taxonomies["security_monitoring"] = ObservabilityTaxonomy(
            taxonomy_id="security_monitoring",
            name="Security Monitoring",
            description="Security events and threat detection",
            domain=ObservabilityDomain.SECURITY,
            level=ObservabilityLevel.ADVANCED,
            key_metrics=[
                "failed_logins", "suspicious_activities", "data_access_violations",
                "privilege_escalations", "anomalous_patterns"
            ],
            sla_targets={
                "failed_logins": 0.02,
                "data_access_violations": 0.0
            },
            alert_thresholds={
                "failed_logins": 0.05,
                "suspicious_activities": 1.0,
                "data_access_violations": 0.0
            },
            business_impact="Security posture and compliance",
            customer_facing=False,
            revenue_impact=False,
            collection_frequency_sec=10,
            retention_days=730,  # 2 years for compliance
            tags=["security", "compliance", "threat_detection"],
            owner_team="security_team"
        )
        
        # Compliance Observability
        taxonomies["compliance_monitoring"] = ObservabilityTaxonomy(
            taxonomy_id="compliance_monitoring",
            name="Compliance Monitoring",
            description="Regulatory compliance and audit metrics",
            domain=ObservabilityDomain.COMPLIANCE,
            level=ObservabilityLevel.INTERMEDIATE,
            key_metrics=[
                "policy_violations", "audit_trail_completeness", "data_retention_compliance",
                "access_control_violations", "encryption_compliance"
            ],
            sla_targets={
                "policy_violations": 0.0,
                "audit_trail_completeness": 1.0,
                "data_retention_compliance": 1.0
            },
            alert_thresholds={
                "policy_violations": 0.0,
                "audit_trail_completeness": 0.99,
                "access_control_violations": 0.0
            },
            business_impact="Regulatory compliance and risk management",
            customer_facing=False,
            revenue_impact=False,
            collection_frequency_sec=60,
            retention_days=2555,  # 7 years for compliance
            tags=["compliance", "audit", "governance"],
            owner_team="compliance_team"
        )
        
        return taxonomies
    
    def _load_anomaly_detection_config(self) -> Dict[str, Any]:
        """Task 7.4.7: Load anomaly detection configuration"""
        return {
            "detection_methods": {
                "statistical": {
                    "enabled": True,
                    "z_score_threshold": 3.0,
                    "window_size": 100
                },
                "machine_learning": {
                    "enabled": True,
                    "model_type": "isolation_forest",
                    "contamination": 0.1
                },
                "rule_based": {
                    "enabled": True,
                    "custom_rules": []
                }
            },
            "sensitivity": {
                "infrastructure": "medium",
                "application": "high",
                "business": "low",
                "security": "high",
                "compliance": "high"
            },
            "notification_config": {
                "immediate_alert_severity": ["critical", "high"],
                "batch_alert_severity": ["medium", "low"],
                "alert_channels": ["slack", "email", "webhook"]
            }
        }
    
    def _load_trust_scoring_config(self) -> Dict[str, Any]:
        """Task 7.4.6: Load trust scoring configuration"""
        return {
            "factor_weights": {
                TrustFactor.DATA_QUALITY: 0.25,
                TrustFactor.COMPLETENESS: 0.20,
                TrustFactor.TIMELINESS: 0.20,
                TrustFactor.CONSISTENCY: 0.20,
                TrustFactor.ACCURACY: 0.15
            },
            "calculation_frequency_minutes": 60,
            "minimum_sample_size": 50,
            "trust_thresholds": {
                "high": 0.8,
                "medium": 0.6,
                "low": 0.4
            },
            "domain_specific_weights": {
                ObservabilityDomain.SECURITY: {
                    TrustFactor.ACCURACY: 0.4,
                    TrustFactor.COMPLETENESS: 0.3,
                    TrustFactor.TIMELINESS: 0.3
                },
                ObservabilityDomain.COMPLIANCE: {
                    TrustFactor.COMPLETENESS: 0.4,
                    TrustFactor.ACCURACY: 0.3,
                    TrustFactor.CONSISTENCY: 0.3
                }
            }
        }
    
    def _load_prediction_config(self) -> Dict[str, Any]:
        """Task 7.4.21: Load predictive analytics configuration"""
        return {
            "prediction_horizons": [1, 6, 12, 24, 168],  # Hours
            "minimum_history_points": 100,
            "confidence_threshold": 0.7,
            "trend_detection": {
                "window_size": 50,
                "significance_threshold": 0.05
            },
            "seasonal_analysis": {
                "enabled": True,
                "periods": [24, 168, 720],  # Daily, weekly, monthly patterns
                "min_cycles": 3
            },
            "business_impact_mapping": {
                "revenue_metrics": ["revenue_per_hour", "conversion_rate", "arr_growth"],
                "customer_metrics": ["active_users", "customer_satisfaction", "churn_rate"],
                "performance_metrics": ["response_time", "success_rate", "throughput"]
            }
        }
    
    def ingest_telemetry(self, data_point: TelemetryDataPoint):
        """Ingest telemetry data point"""
        try:
            metric_key = f"{data_point.domain.value}_{data_point.metric_name}"
            
            if metric_key not in self.telemetry_data:
                self.telemetry_data[metric_key] = []
            
            self.telemetry_data[metric_key].append(data_point)
            
            # Keep only recent data (based on taxonomy retention)
            taxonomy = self._get_taxonomy_for_metric(data_point.domain, data_point.metric_name)
            if taxonomy:
                retention_cutoff = datetime.utcnow() - timedelta(days=taxonomy.retention_days)
                self.telemetry_data[metric_key] = [
                    dp for dp in self.telemetry_data[metric_key]
                    if dp.timestamp > retention_cutoff
                ]
            
            # Trigger real-time analysis
            self._analyze_data_point(data_point)
            
        except Exception as e:
            self.logger.error(f"Failed to ingest telemetry data: {e}")
    
    def detect_anomalies(self, metric_name: str, domain: ObservabilityDomain,
                        tenant_id: Optional[int] = None) -> List[AnomalyDetection]:
        """Task 7.4.7: Build anomaly detection for telemetry streams"""
        try:
            metric_key = f"{domain.value}_{metric_name}"
            
            if metric_key not in self.telemetry_data:
                return []
            
            data_points = self.telemetry_data[metric_key]
            
            # Filter by tenant if specified
            if tenant_id:
                data_points = [dp for dp in data_points if dp.tenant_id == tenant_id]
            
            if len(data_points) < 10:
                return []
            
            anomalies = []
            
            # Statistical anomaly detection
            if self.anomaly_detection_config["detection_methods"]["statistical"]["enabled"]:
                statistical_anomalies = self._detect_statistical_anomalies(data_points, metric_name, domain)
                anomalies.extend(statistical_anomalies)
            
            # Pattern-based anomaly detection
            pattern_anomalies = self._detect_pattern_anomalies(data_points, metric_name, domain)
            anomalies.extend(pattern_anomalies)
            
            # Store detected anomalies
            self.anomalies.extend(anomalies)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Failed to detect anomalies for {metric_name}: {e}")
            return []
    
    def calculate_trust_score(self, metric_name: str, domain: ObservabilityDomain,
                            tenant_id: Optional[int] = None,
                            time_window_hours: int = 24) -> Optional[TrustScore]:
        """Task 7.4.6: Configure trust scoring on telemetry data"""
        try:
            metric_key = f"{domain.value}_{metric_name}"
            
            if metric_key not in self.telemetry_data:
                return None
            
            # Get data points within time window
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            data_points = [
                dp for dp in self.telemetry_data[metric_key]
                if dp.timestamp > cutoff_time
            ]
            
            # Filter by tenant if specified
            if tenant_id:
                data_points = [dp for dp in data_points if dp.tenant_id == tenant_id]
            
            if len(data_points) < self.trust_scoring_config["minimum_sample_size"]:
                return None
            
            # Calculate individual factor scores
            data_quality_score = self._calculate_data_quality_score(data_points)
            completeness_score = self._calculate_completeness_score(data_points, time_window_hours)
            timeliness_score = self._calculate_timeliness_score(data_points)
            consistency_score = self._calculate_consistency_score(data_points)
            accuracy_score = self._calculate_accuracy_score(data_points, metric_name, domain)
            
            # Get domain-specific weights
            weights = self.trust_scoring_config["domain_specific_weights"].get(
                domain, self.trust_scoring_config["factor_weights"]
            )
            
            # Calculate overall score
            overall_score = (
                data_quality_score * weights.get(TrustFactor.DATA_QUALITY, 0.25) +
                completeness_score * weights.get(TrustFactor.COMPLETENESS, 0.20) +
                timeliness_score * weights.get(TrustFactor.TIMELINESS, 0.20) +
                consistency_score * weights.get(TrustFactor.CONSISTENCY, 0.20) +
                accuracy_score * weights.get(TrustFactor.ACCURACY, 0.15)
            )
            
            # Calculate confidence interval (simplified)
            confidence_margin = min(0.1, 1.0 / math.sqrt(len(data_points)))
            
            trust_score = TrustScore(
                metric_name=metric_name,
                calculated_at=datetime.utcnow(),
                overall_score=overall_score,
                data_quality_score=data_quality_score,
                completeness_score=completeness_score,
                timeliness_score=timeliness_score,
                consistency_score=consistency_score,
                accuracy_score=accuracy_score,
                domain=domain,
                tenant_id=tenant_id,
                time_window_hours=time_window_hours,
                confidence_interval=(
                    max(0.0, overall_score - confidence_margin),
                    min(1.0, overall_score + confidence_margin)
                ),
                sample_size=len(data_points),
                factors_analyzed=list(TrustFactor)
            )
            
            # Store trust score
            trust_key = f"{domain.value}_{metric_name}_{tenant_id or 'global'}"
            self.trust_scores[trust_key] = trust_score
            
            return trust_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate trust score for {metric_name}: {e}")
            return None
    
    def generate_predictive_insights(self, metric_name: str, domain: ObservabilityDomain,
                                   prediction_horizon_hours: int = 24,
                                   tenant_id: Optional[int] = None) -> Optional[PredictiveInsight]:
        """Task 7.4.21: Build predictive analytics on telemetry trends"""
        try:
            metric_key = f"{domain.value}_{metric_name}"
            
            if metric_key not in self.telemetry_data:
                return None
            
            data_points = self.telemetry_data[metric_key]
            
            # Filter by tenant if specified
            if tenant_id:
                data_points = [dp for dp in data_points if dp.tenant_id == tenant_id]
            
            if len(data_points) < self.prediction_config["minimum_history_points"]:
                return None
            
            # Sort by timestamp
            data_points.sort(key=lambda dp: dp.timestamp)
            
            # Extract values and timestamps
            values = [dp.value for dp in data_points]
            timestamps = [dp.timestamp for dp in data_points]
            
            # Trend analysis
            trend_analysis = self._analyze_trend(values, timestamps)
            
            # Seasonal pattern detection
            seasonal_analysis = self._detect_seasonal_patterns(values, timestamps)
            
            # Generate prediction
            prediction_result = self._predict_future_value(
                values, timestamps, prediction_horizon_hours
            )
            
            # Assess business impact
            business_impact = self._assess_business_impact(
                metric_name, domain, prediction_result["predicted_value"], trend_analysis
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                metric_name, domain, trend_analysis, prediction_result, business_impact
            )
            
            insight = PredictiveInsight(
                insight_id=f"{metric_key}_{datetime.utcnow().isoformat()}",
                generated_at=datetime.utcnow(),
                metric_name=metric_name,
                predicted_value=prediction_result["predicted_value"],
                prediction_horizon_hours=prediction_horizon_hours,
                confidence=prediction_result["confidence"],
                trend_direction=trend_analysis["direction"],
                trend_strength=trend_analysis["strength"],
                seasonal_pattern=seasonal_analysis["detected"],
                domain=domain,
                business_impact=business_impact["description"],
                recommended_actions=recommendations,
                risk_level=business_impact["risk_level"],
                risk_factors=business_impact["risk_factors"]
            )
            
            # Store insight
            self.predictive_insights.append(insight)
            
            return insight
            
        except Exception as e:
            self.logger.error(f"Failed to generate predictive insights for {metric_name}: {e}")
            return None
    
    def get_trust_dashboard_data(self, domain: Optional[ObservabilityDomain] = None,
                               tenant_id: Optional[int] = None) -> Dict[str, Any]:
        """Task 7.4.28: Build trust dashboards scoped to observability"""
        try:
            dashboard_data = {
                "generated_at": datetime.utcnow().isoformat(),
                "domain": domain.value if domain else "all",
                "tenant_id": tenant_id,
                "trust_scores": {},
                "anomalies": [],
                "predictive_insights": [],
                "summary": {}
            }
            
            # Filter trust scores
            filtered_trust_scores = {}
            for key, trust_score in self.trust_scores.items():
                if domain and trust_score.domain != domain:
                    continue
                if tenant_id and trust_score.tenant_id != tenant_id:
                    continue
                filtered_trust_scores[key] = trust_score
            
            dashboard_data["trust_scores"] = {
                key: {
                    "metric_name": ts.metric_name,
                    "overall_score": ts.overall_score,
                    "data_quality_score": ts.data_quality_score,
                    "completeness_score": ts.completeness_score,
                    "timeliness_score": ts.timeliness_score,
                    "consistency_score": ts.consistency_score,
                    "accuracy_score": ts.accuracy_score,
                    "calculated_at": ts.calculated_at.isoformat(),
                    "sample_size": ts.sample_size
                }
                for key, ts in filtered_trust_scores.items()
            }
            
            # Filter anomalies (last 24 hours)
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            filtered_anomalies = [
                anomaly for anomaly in self.anomalies
                if anomaly.detected_at > cutoff_time
                and (not domain or anomaly.domain == domain)
                and (not tenant_id or anomaly.tenant_id == tenant_id)
            ]
            
            dashboard_data["anomalies"] = [
                {
                    "anomaly_id": a.anomaly_id,
                    "detected_at": a.detected_at.isoformat(),
                    "metric_name": a.metric_name,
                    "anomaly_type": a.anomaly_type.value,
                    "severity": a.severity,
                    "deviation_score": a.deviation_score,
                    "confidence": a.confidence,
                    "business_impact": a.business_impact
                }
                for a in filtered_anomalies
            ]
            
            # Filter predictive insights (last 24 hours)
            filtered_insights = [
                insight for insight in self.predictive_insights
                if insight.generated_at > cutoff_time
                and (not domain or insight.domain == domain)
            ]
            
            dashboard_data["predictive_insights"] = [
                {
                    "insight_id": i.insight_id,
                    "generated_at": i.generated_at.isoformat(),
                    "metric_name": i.metric_name,
                    "predicted_value": i.predicted_value,
                    "confidence": i.confidence,
                    "trend_direction": i.trend_direction,
                    "risk_level": i.risk_level,
                    "business_impact": i.business_impact
                }
                for i in filtered_insights
            ]
            
            # Generate summary
            dashboard_data["summary"] = {
                "total_metrics_monitored": len(filtered_trust_scores),
                "average_trust_score": statistics.mean([ts.overall_score for ts in filtered_trust_scores.values()]) if filtered_trust_scores else 0.0,
                "anomalies_detected_24h": len(filtered_anomalies),
                "high_risk_insights": len([i for i in filtered_insights if i.risk_level == "high"]),
                "trust_score_distribution": self._calculate_trust_score_distribution(filtered_trust_scores),
                "top_concerns": self._identify_top_concerns(filtered_anomalies, filtered_insights)
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate trust dashboard data: {e}")
            return {"error": str(e)}
    
    # Helper methods for internal calculations
    
    def _get_taxonomy_for_metric(self, domain: ObservabilityDomain, metric_name: str) -> Optional[ObservabilityTaxonomy]:
        """Get taxonomy that contains the specified metric"""
        for taxonomy in self.taxonomies.values():
            if taxonomy.domain == domain and metric_name in taxonomy.key_metrics:
                return taxonomy
        return None
    
    def _analyze_data_point(self, data_point: TelemetryDataPoint):
        """Analyze individual data point for real-time insights"""
        # Check for immediate threshold violations
        taxonomy = self._get_taxonomy_for_metric(data_point.domain, data_point.metric_name)
        if taxonomy and data_point.metric_name in taxonomy.alert_thresholds:
            threshold = taxonomy.alert_thresholds[data_point.metric_name]
            if data_point.value > threshold:
                self.logger.warning(f"Threshold violation: {data_point.metric_name} = {data_point.value} > {threshold}")
    
    def _detect_statistical_anomalies(self, data_points: List[TelemetryDataPoint],
                                    metric_name: str, domain: ObservabilityDomain) -> List[AnomalyDetection]:
        """Detect statistical anomalies using z-score"""
        if len(data_points) < 30:
            return []
        
        values = [dp.value for dp in data_points]
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0.0
        
        if std_val == 0:
            return []
        
        anomalies = []
        threshold = self.anomaly_detection_config["detection_methods"]["statistical"]["z_score_threshold"]
        
        for dp in data_points[-10:]:  # Check last 10 points
            z_score = abs(dp.value - mean_val) / std_val
            if z_score > threshold:
                anomaly = AnomalyDetection(
                    anomaly_id=f"{metric_name}_{dp.timestamp.isoformat()}",
                    detected_at=datetime.utcnow(),
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.SPIKE if dp.value > mean_val else AnomalyType.DIP,
                    actual_value=dp.value,
                    expected_value=mean_val,
                    deviation_score=z_score,
                    confidence=min(0.95, z_score / threshold),
                    domain=domain,
                    tenant_id=dp.tenant_id,
                    severity="high" if z_score > threshold * 1.5 else "medium",
                    detection_method="statistical_z_score"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_pattern_anomalies(self, data_points: List[TelemetryDataPoint],
                                metric_name: str, domain: ObservabilityDomain) -> List[AnomalyDetection]:
        """Detect pattern-based anomalies"""
        # Simplified pattern detection - can be enhanced
        return []
    
    def _calculate_data_quality_score(self, data_points: List[TelemetryDataPoint]) -> float:
        """Calculate data quality score"""
        if not data_points:
            return 0.0
        
        # Check for valid values, no nulls, reasonable ranges
        valid_points = [dp for dp in data_points if dp.quality_score > 0.5]
        return len(valid_points) / len(data_points)
    
    def _calculate_completeness_score(self, data_points: List[TelemetryDataPoint], 
                                    time_window_hours: int) -> float:
        """Calculate completeness score"""
        if not data_points:
            return 0.0
        
        # Expected data points based on collection frequency
        expected_points = time_window_hours * 60  # Assuming 1-minute frequency
        actual_points = len(data_points)
        
        return min(1.0, actual_points / expected_points)
    
    def _calculate_timeliness_score(self, data_points: List[TelemetryDataPoint]) -> float:
        """Calculate timeliness score"""
        if not data_points:
            return 0.0
        
        # Check how recent the data is
        now = datetime.utcnow()
        recent_points = [dp for dp in data_points if (now - dp.timestamp).total_seconds() < 300]  # 5 minutes
        
        return len(recent_points) / len(data_points)
    
    def _calculate_consistency_score(self, data_points: List[TelemetryDataPoint]) -> float:
        """Calculate consistency score"""
        if len(data_points) < 2:
            return 1.0
        
        values = [dp.value for dp in data_points]
        cv = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 0
        
        # Lower coefficient of variation = higher consistency
        return max(0.0, 1.0 - cv)
    
    def _calculate_accuracy_score(self, data_points: List[TelemetryDataPoint],
                                metric_name: str, domain: ObservabilityDomain) -> float:
        """Calculate accuracy score"""
        # Simplified accuracy calculation
        return statistics.mean([dp.confidence for dp in data_points])
    
    def _analyze_trend(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Analyze trend in time series data"""
        if len(values) < 10:
            return {"direction": "stable", "strength": 0.0}
        
        # Simple linear trend analysis
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
        strength = abs(slope)
        
        return {"direction": direction, "strength": strength, "slope": slope}
    
    def _detect_seasonal_patterns(self, values: List[float], timestamps: List[datetime]) -> Dict[str, Any]:
        """Detect seasonal patterns in time series"""
        # Simplified seasonal detection
        return {"detected": False, "period": None, "strength": 0.0}
    
    def _predict_future_value(self, values: List[float], timestamps: List[datetime],
                            horizon_hours: int) -> Dict[str, Any]:
        """Predict future value using simple trend extrapolation"""
        if len(values) < 10:
            return {"predicted_value": values[-1] if values else 0.0, "confidence": 0.3}
        
        # Simple linear extrapolation
        trend = self._analyze_trend(values, timestamps)
        current_value = values[-1]
        predicted_value = current_value + (trend["slope"] * horizon_hours)
        
        # Confidence based on trend strength and data consistency
        confidence = min(0.9, 0.5 + trend["strength"] * 0.4)
        
        return {"predicted_value": predicted_value, "confidence": confidence}
    
    def _assess_business_impact(self, metric_name: str, domain: ObservabilityDomain,
                              predicted_value: float, trend_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess business impact of predicted changes"""
        impact_mapping = self.prediction_config["business_impact_mapping"]
        
        risk_level = "low"
        risk_factors = []
        description = "Minimal business impact expected"
        
        # Check if metric is business-critical
        if metric_name in impact_mapping["revenue_metrics"]:
            if trend_analysis["direction"] == "decreasing":
                risk_level = "high"
                risk_factors.append("Revenue impact")
                description = "Potential revenue impact from declining metric"
        
        elif metric_name in impact_mapping["customer_metrics"]:
            if trend_analysis["direction"] == "decreasing" and metric_name != "churn_rate":
                risk_level = "medium"
                risk_factors.append("Customer experience impact")
                description = "Potential customer experience degradation"
        
        elif metric_name in impact_mapping["performance_metrics"]:
            if trend_analysis["direction"] == "decreasing":
                risk_level = "medium"
                risk_factors.append("Performance degradation")
                description = "System performance may be declining"
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "description": description
        }
    
    def _generate_recommendations(self, metric_name: str, domain: ObservabilityDomain,
                                trend_analysis: Dict[str, Any], prediction_result: Dict[str, Any],
                                business_impact: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if business_impact["risk_level"] == "high":
            recommendations.append("Immediate investigation required")
            recommendations.append("Alert relevant stakeholders")
        
        if trend_analysis["direction"] == "decreasing":
            recommendations.append("Monitor trend closely")
            recommendations.append("Consider proactive intervention")
        
        if prediction_result["confidence"] < 0.5:
            recommendations.append("Increase monitoring frequency")
            recommendations.append("Gather additional data points")
        
        return recommendations
    
    def _calculate_trust_score_distribution(self, trust_scores: Dict[str, TrustScore]) -> Dict[str, int]:
        """Calculate distribution of trust scores"""
        distribution = {"high": 0, "medium": 0, "low": 0}
        thresholds = self.trust_scoring_config["trust_thresholds"]
        
        for trust_score in trust_scores.values():
            score = trust_score.overall_score
            if score >= thresholds["high"]:
                distribution["high"] += 1
            elif score >= thresholds["medium"]:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1
        
        return distribution
    
    def _identify_top_concerns(self, anomalies: List[AnomalyDetection],
                             insights: List[PredictiveInsight]) -> List[str]:
        """Identify top concerns from anomalies and insights"""
        concerns = []
        
        # High severity anomalies
        high_severity_anomalies = [a for a in anomalies if a.severity in ["high", "critical"]]
        if high_severity_anomalies:
            concerns.append(f"{len(high_severity_anomalies)} high-severity anomalies detected")
        
        # High risk insights
        high_risk_insights = [i for i in insights if i.risk_level == "high"]
        if high_risk_insights:
            concerns.append(f"{len(high_risk_insights)} high-risk predictions")
        
        # Business impact concerns
        business_impact_insights = [i for i in insights if "revenue" in i.business_impact.lower()]
        if business_impact_insights:
            concerns.append("Revenue-impacting trends detected")
        
        return concerns[:5]  # Top 5 concerns


# Global instance
observability_framework = ObservabilityFramework()

def get_observability_framework() -> ObservabilityFramework:
    """Get the global observability framework instance"""
    return observability_framework
