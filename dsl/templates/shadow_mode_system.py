"""
Shadow Mode System for Template Testing
Task 4.2.22: Build shadow mode for templates (run ML in background, compare vs RBA)
"""

import json
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import statistics
from enum import Enum
import sqlite3

from .industry_template_registry import IndustryTemplateRegistry, TemplateConfig
from .template_confidence_manager import TemplateConfidenceManager
from ..operators.base import OperatorContext, OperatorResult

logger = logging.getLogger(__name__)

class ShadowMode(Enum):
    """Shadow mode types"""
    DISABLED = "disabled"
    PASSIVE = "passive"  # Run ML in background, log results
    COMPARATIVE = "comparative"  # Compare ML vs RBA decisions
    CANARY = "canary"  # Route small percentage to ML
    CHAMPION_CHALLENGER = "champion_challenger"  # A/B test ML vs RBA

class ComparisonResult(Enum):
    """Comparison results between ML and RBA"""
    AGREEMENT = "agreement"
    ML_MORE_CONSERVATIVE = "ml_more_conservative"
    ML_MORE_AGGRESSIVE = "ml_more_aggressive"
    CONFLICTING = "conflicting"
    ERROR = "error"

@dataclass
class ShadowExecution:
    """Shadow execution record"""
    execution_id: str
    template_id: str
    workflow_id: str
    step_id: str
    shadow_mode: ShadowMode
    input_data: Dict[str, Any]
    rba_result: Dict[str, Any]
    ml_result: Dict[str, Any]
    comparison_result: ComparisonResult
    confidence_score: float
    execution_time_ml_ms: int
    execution_time_rba_ms: int
    tenant_id: str
    user_id: int
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ShadowModeConfig:
    """Configuration for shadow mode"""
    template_id: str
    shadow_mode: ShadowMode
    traffic_percentage: float = 100.0  # Percentage of traffic to shadow
    comparison_enabled: bool = True
    logging_enabled: bool = True
    alerting_enabled: bool = True
    performance_tracking: bool = True
    canary_percentage: float = 5.0  # For canary mode
    comparison_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "agreement_threshold": 0.8,
        "performance_threshold": 0.9,
        "confidence_threshold": 0.7
    })
    alert_conditions: List[str] = field(default_factory=lambda: [
        "disagreement_rate > 0.3",
        "ml_error_rate > 0.1",
        "performance_degradation > 0.2"
    ])

@dataclass
class ShadowModeMetrics:
    """Metrics for shadow mode performance"""
    template_id: str
    total_executions: int
    agreement_rate: float
    ml_error_rate: float
    rba_error_rate: float
    avg_ml_execution_time: float
    avg_rba_execution_time: float
    performance_ratio: float  # ML time / RBA time
    confidence_distribution: Dict[str, int]
    comparison_breakdown: Dict[ComparisonResult, int]
    recommendation: str
    analysis_period_days: int
    created_at: datetime = field(default_factory=datetime.utcnow)

class ShadowModeSystem:
    """
    Shadow mode system for template testing and validation
    
    Provides:
    - Passive ML execution alongside RBA
    - Comparative analysis of ML vs RBA decisions
    - Canary releases for gradual ML rollout
    - Champion/Challenger A/B testing
    - Performance monitoring and alerting
    - Recommendation engine for ML adoption
    """
    
    def __init__(self, db_path: str = "shadow_mode.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.template_registry = IndustryTemplateRegistry()
        self.confidence_manager = TemplateConfidenceManager()
        self.shadow_configs: Dict[str, ShadowModeConfig] = {}
        self.execution_cache: List[ShadowExecution] = []
        self._init_database()
        self._initialize_shadow_configs()
    
    def _init_database(self):
        """Initialize shadow mode database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create shadow executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shadow_executions (
                    execution_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    shadow_mode TEXT NOT NULL,
                    input_data TEXT NOT NULL,  -- JSON
                    rba_result TEXT NOT NULL,  -- JSON
                    ml_result TEXT NOT NULL,   -- JSON
                    comparison_result TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    execution_time_ml_ms INTEGER NOT NULL,
                    execution_time_rba_ms INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create shadow metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS shadow_metrics (
                    metric_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    total_executions INTEGER NOT NULL,
                    agreement_rate REAL NOT NULL,
                    ml_error_rate REAL NOT NULL,
                    rba_error_rate REAL NOT NULL,
                    avg_ml_execution_time REAL NOT NULL,
                    avg_rba_execution_time REAL NOT NULL,
                    performance_ratio REAL NOT NULL,
                    confidence_distribution TEXT NOT NULL,  -- JSON
                    comparison_breakdown TEXT NOT NULL,     -- JSON
                    recommendation TEXT NOT NULL,
                    analysis_period_days INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shadow_executions_template ON shadow_executions(template_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shadow_executions_tenant ON shadow_executions(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shadow_executions_created_at ON shadow_executions(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shadow_metrics_template ON shadow_metrics(template_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Shadow mode database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize shadow mode database: {e}")
            raise
    
    def _initialize_shadow_configs(self):
        """Initialize shadow mode configurations for templates"""
        
        templates = self.template_registry.list_all_templates()
        
        for template in templates:
            # Default configuration - start with passive mode
            self.shadow_configs[template.template_id] = ShadowModeConfig(
                template_id=template.template_id,
                shadow_mode=ShadowMode.PASSIVE,
                traffic_percentage=100.0,
                comparison_enabled=True,
                logging_enabled=True,
                alerting_enabled=True,
                performance_tracking=True,
                canary_percentage=5.0
            )
        
        self.logger.info(f"Initialized shadow configs for {len(self.shadow_configs)} templates")
    
    async def execute_shadow_mode(
        self,
        template_id: str,
        workflow_id: str,
        step_id: str,
        input_data: Dict[str, Any],
        context: OperatorContext,
        rba_logic: Callable[[Dict[str, Any], OperatorContext], Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute shadow mode for a template step"""
        
        try:
            config = self.shadow_configs.get(template_id)
            if not config or config.shadow_mode == ShadowMode.DISABLED:
                # Execute only RBA logic
                return await self._execute_rba_only(rba_logic, input_data, context)
            
            # Check traffic percentage
            if not self._should_execute_shadow(config.traffic_percentage):
                return await self._execute_rba_only(rba_logic, input_data, context)
            
            execution_id = str(uuid.uuid4())
            
            # Execute both RBA and ML in parallel
            rba_result, ml_result, execution_times = await self._execute_parallel(
                template_id, input_data, context, rba_logic
            )
            
            # Compare results
            comparison_result = await self._compare_results(
                template_id, rba_result, ml_result, input_data
            )
            
            # Determine which result to return based on shadow mode
            final_result = await self._determine_final_result(
                config, rba_result, ml_result, comparison_result
            )
            
            # Log execution
            shadow_execution = ShadowExecution(
                execution_id=execution_id,
                template_id=template_id,
                workflow_id=workflow_id,
                step_id=step_id,
                shadow_mode=config.shadow_mode,
                input_data=input_data,
                rba_result=rba_result,
                ml_result=ml_result,
                comparison_result=comparison_result,
                confidence_score=ml_result.get("confidence_score", 0.0),
                execution_time_ml_ms=execution_times["ml_ms"],
                execution_time_rba_ms=execution_times["rba_ms"],
                tenant_id=context.tenant_id,
                user_id=context.user_id
            )
            
            if config.logging_enabled:
                await self._log_shadow_execution(shadow_execution)
            
            # Cache for analysis
            self.execution_cache.append(shadow_execution)
            
            # Check for alerts
            if config.alerting_enabled:
                await self._check_alert_conditions(template_id, shadow_execution)
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Shadow mode execution failed for {template_id}: {e}")
            # Fallback to RBA only
            return await self._execute_rba_only(rba_logic, input_data, context)
    
    async def _execute_rba_only(
        self,
        rba_logic: Callable[[Dict[str, Any], OperatorContext], Dict[str, Any]],
        input_data: Dict[str, Any],
        context: OperatorContext
    ) -> Dict[str, Any]:
        """Execute only RBA logic"""
        
        start_time = datetime.utcnow()
        try:
            result = await rba_logic(input_data, context)
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return {
                "success": True,
                "result": result,
                "execution_time_ms": execution_time,
                "mode": "rba_only"
            }
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "mode": "rba_only"
            }
    
    async def _execute_parallel(
        self,
        template_id: str,
        input_data: Dict[str, Any],
        context: OperatorContext,
        rba_logic: Callable[[Dict[str, Any], OperatorContext], Dict[str, Any]]
    ) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, int]]:
        """Execute RBA and ML logic in parallel"""
        
        # Execute RBA and ML concurrently
        rba_task = asyncio.create_task(self._execute_rba_logic(rba_logic, input_data, context))
        ml_task = asyncio.create_task(self._execute_ml_logic(template_id, input_data, context))
        
        rba_result, ml_result = await asyncio.gather(rba_task, ml_task, return_exceptions=True)
        
        # Handle exceptions
        if isinstance(rba_result, Exception):
            rba_result = {
                "success": False,
                "error": str(rba_result),
                "execution_time_ms": 0
            }
        
        if isinstance(ml_result, Exception):
            ml_result = {
                "success": False,
                "error": str(ml_result),
                "execution_time_ms": 0,
                "confidence_score": 0.0
            }
        
        execution_times = {
            "rba_ms": rba_result.get("execution_time_ms", 0),
            "ml_ms": ml_result.get("execution_time_ms", 0)
        }
        
        return rba_result, ml_result, execution_times
    
    async def _execute_rba_logic(
        self,
        rba_logic: Callable[[Dict[str, Any], OperatorContext], Dict[str, Any]],
        input_data: Dict[str, Any],
        context: OperatorContext
    ) -> Dict[str, Any]:
        """Execute RBA logic with timing"""
        
        start_time = datetime.utcnow()
        try:
            result = await rba_logic(input_data, context)
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return {
                "success": True,
                "result": result,
                "execution_time_ms": execution_time
            }
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time
            }
    
    async def _execute_ml_logic(
        self,
        template_id: str,
        input_data: Dict[str, Any],
        context: OperatorContext
    ) -> Dict[str, Any]:
        """Execute ML logic with timing"""
        
        start_time = datetime.utcnow()
        try:
            template = self.template_registry.get_template(template_id)
            if not template:
                raise ValueError(f"Template {template_id} not found")
            
            # Simulate ML execution
            # In a real implementation, this would call the actual ML operators
            ml_result = await self._simulate_ml_execution(template, input_data, context)
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return {
                "success": True,
                "result": ml_result,
                "execution_time_ms": execution_time,
                "confidence_score": ml_result.get("confidence_score", 0.5)
            }
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return {
                "success": False,
                "error": str(e),
                "execution_time_ms": execution_time,
                "confidence_score": 0.0
            }
    
    async def _simulate_ml_execution(
        self,
        template: TemplateConfig,
        input_data: Dict[str, Any],
        context: OperatorContext
    ) -> Dict[str, Any]:
        """Simulate ML execution for shadow mode"""
        
        # Simulate different ML results based on template type
        if "churn" in template.template_id:
            confidence = min(0.95, max(0.1, 
                0.5 + (input_data.get("mrr", 1000) - 500) / 1000 * 0.3
            ))
            return {
                "churn_probability": confidence,
                "churn_segment": "high_risk" if confidence > 0.7 else "medium_risk" if confidence > 0.4 else "low_risk",
                "confidence_score": confidence
            }
        
        elif "fraud" in template.template_id:
            confidence = min(0.95, max(0.1,
                0.3 + (input_data.get("transaction_amount", 100) / 1000) * 0.4
            ))
            return {
                "fraud_probability": confidence,
                "fraud_score": confidence * 100,
                "confidence_score": confidence
            }
        
        elif "credit" in template.template_id:
            confidence = min(0.95, max(0.1,
                0.6 + (input_data.get("annual_income", 50000) / 100000) * 0.2
            ))
            return {
                "credit_score": int(300 + confidence * 550),
                "risk_category": "low" if confidence > 0.7 else "medium" if confidence > 0.4 else "high",
                "confidence_score": confidence
            }
        
        else:
            # Generic ML result
            confidence = 0.6 + (hash(str(input_data)) % 100) / 250  # Pseudo-random but deterministic
            return {
                "prediction": "positive" if confidence > 0.5 else "negative",
                "score": confidence,
                "confidence_score": confidence
            }
    
    async def _compare_results(
        self,
        template_id: str,
        rba_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        input_data: Dict[str, Any]
    ) -> ComparisonResult:
        """Compare RBA and ML results"""
        
        try:
            if not rba_result.get("success") or not ml_result.get("success"):
                return ComparisonResult.ERROR
            
            rba_data = rba_result.get("result", {})
            ml_data = ml_result.get("result", {})
            
            # Template-specific comparison logic
            if "churn" in template_id:
                return self._compare_churn_results(rba_data, ml_data)
            elif "fraud" in template_id:
                return self._compare_fraud_results(rba_data, ml_data)
            elif "credit" in template_id:
                return self._compare_credit_results(rba_data, ml_data)
            else:
                return self._compare_generic_results(rba_data, ml_data)
                
        except Exception as e:
            self.logger.error(f"Failed to compare results: {e}")
            return ComparisonResult.ERROR
    
    def _compare_churn_results(self, rba_data: Dict[str, Any], ml_data: Dict[str, Any]) -> ComparisonResult:
        """Compare churn prediction results"""
        
        rba_risk = rba_data.get("churn_segment", "low_risk")
        ml_risk = ml_data.get("churn_segment", "low_risk")
        
        risk_levels = {"low_risk": 1, "medium_risk": 2, "high_risk": 3}
        
        rba_level = risk_levels.get(rba_risk, 1)
        ml_level = risk_levels.get(ml_risk, 1)
        
        if rba_level == ml_level:
            return ComparisonResult.AGREEMENT
        elif ml_level > rba_level:
            return ComparisonResult.ML_MORE_CONSERVATIVE
        else:
            return ComparisonResult.ML_MORE_AGGRESSIVE
    
    def _compare_fraud_results(self, rba_data: Dict[str, Any], ml_data: Dict[str, Any]) -> ComparisonResult:
        """Compare fraud detection results"""
        
        rba_score = rba_data.get("fraud_score", 0)
        ml_score = ml_data.get("fraud_score", 0)
        
        # Normalize scores to 0-100 range
        rba_normalized = min(100, max(0, rba_score))
        ml_normalized = min(100, max(0, ml_score))
        
        diff = abs(rba_normalized - ml_normalized)
        
        if diff <= 10:  # Within 10 points
            return ComparisonResult.AGREEMENT
        elif ml_normalized > rba_normalized + 10:
            return ComparisonResult.ML_MORE_CONSERVATIVE
        elif ml_normalized < rba_normalized - 10:
            return ComparisonResult.ML_MORE_AGGRESSIVE
        else:
            return ComparisonResult.CONFLICTING
    
    def _compare_credit_results(self, rba_data: Dict[str, Any], ml_data: Dict[str, Any]) -> ComparisonResult:
        """Compare credit scoring results"""
        
        rba_score = rba_data.get("credit_score", 500)
        ml_score = ml_data.get("credit_score", 500)
        
        diff = abs(rba_score - ml_score)
        
        if diff <= 20:  # Within 20 points
            return ComparisonResult.AGREEMENT
        elif ml_score < rba_score - 20:
            return ComparisonResult.ML_MORE_CONSERVATIVE
        elif ml_score > rba_score + 20:
            return ComparisonResult.ML_MORE_AGGRESSIVE
        else:
            return ComparisonResult.CONFLICTING
    
    def _compare_generic_results(self, rba_data: Dict[str, Any], ml_data: Dict[str, Any]) -> ComparisonResult:
        """Compare generic results"""
        
        # Simple comparison based on primary prediction/score
        rba_prediction = rba_data.get("prediction", rba_data.get("score", 0))
        ml_prediction = ml_data.get("prediction", ml_data.get("score", 0))
        
        if str(rba_prediction).lower() == str(ml_prediction).lower():
            return ComparisonResult.AGREEMENT
        else:
            return ComparisonResult.CONFLICTING
    
    async def _determine_final_result(
        self,
        config: ShadowModeConfig,
        rba_result: Dict[str, Any],
        ml_result: Dict[str, Any],
        comparison_result: ComparisonResult
    ) -> Dict[str, Any]:
        """Determine which result to return based on shadow mode"""
        
        if config.shadow_mode == ShadowMode.PASSIVE:
            # Always return RBA result
            return rba_result
        
        elif config.shadow_mode == ShadowMode.COMPARATIVE:
            # Return RBA result but include comparison metadata
            result = rba_result.copy()
            result["shadow_mode"] = {
                "ml_result": ml_result,
                "comparison": comparison_result.value,
                "mode": "comparative"
            }
            return result
        
        elif config.shadow_mode == ShadowMode.CANARY:
            # Route small percentage to ML
            if self._should_use_ml_result(config.canary_percentage):
                result = ml_result.copy()
                result["shadow_mode"] = {
                    "source": "ml",
                    "mode": "canary"
                }
                return result
            else:
                result = rba_result.copy()
                result["shadow_mode"] = {
                    "source": "rba",
                    "mode": "canary"
                }
                return result
        
        elif config.shadow_mode == ShadowMode.CHAMPION_CHALLENGER:
            # A/B test - 50/50 split
            if self._should_use_ml_result(50.0):
                result = ml_result.copy()
                result["shadow_mode"] = {
                    "source": "ml",
                    "mode": "champion_challenger"
                }
                return result
            else:
                result = rba_result.copy()
                result["shadow_mode"] = {
                    "source": "rba",
                    "mode": "champion_challenger"
                }
                return result
        
        else:
            # Default to RBA
            return rba_result
    
    def _should_execute_shadow(self, traffic_percentage: float) -> bool:
        """Determine if shadow mode should execute based on traffic percentage"""
        import random
        return random.uniform(0, 100) < traffic_percentage
    
    def _should_use_ml_result(self, ml_percentage: float) -> bool:
        """Determine if ML result should be used"""
        import random
        return random.uniform(0, 100) < ml_percentage
    
    async def _log_shadow_execution(self, execution: ShadowExecution):
        """Log shadow execution to database"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO shadow_executions (
                    execution_id, template_id, workflow_id, step_id, shadow_mode,
                    input_data, rba_result, ml_result, comparison_result,
                    confidence_score, execution_time_ml_ms, execution_time_rba_ms,
                    tenant_id, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution.execution_id,
                execution.template_id,
                execution.workflow_id,
                execution.step_id,
                execution.shadow_mode.value,
                json.dumps(execution.input_data),
                json.dumps(execution.rba_result),
                json.dumps(execution.ml_result),
                execution.comparison_result.value,
                execution.confidence_score,
                execution.execution_time_ml_ms,
                execution.execution_time_rba_ms,
                execution.tenant_id,
                execution.user_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log shadow execution: {e}")
    
    async def _check_alert_conditions(self, template_id: str, execution: ShadowExecution):
        """Check alert conditions and trigger alerts if needed"""
        
        try:
            config = self.shadow_configs.get(template_id)
            if not config or not config.alert_conditions:
                return
            
            # Get recent executions for analysis
            recent_executions = [
                exec for exec in self.execution_cache[-100:]  # Last 100 executions
                if exec.template_id == template_id
            ]
            
            if len(recent_executions) < 10:
                return  # Not enough data
            
            # Calculate metrics
            disagreement_rate = sum(
                1 for exec in recent_executions 
                if exec.comparison_result != ComparisonResult.AGREEMENT
            ) / len(recent_executions)
            
            ml_error_rate = sum(
                1 for exec in recent_executions 
                if not exec.ml_result.get("success", True)
            ) / len(recent_executions)
            
            avg_ml_time = statistics.mean([exec.execution_time_ml_ms for exec in recent_executions])
            avg_rba_time = statistics.mean([exec.execution_time_rba_ms for exec in recent_executions])
            performance_degradation = (avg_ml_time - avg_rba_time) / avg_rba_time if avg_rba_time > 0 else 0
            
            # Check alert conditions
            alerts = []
            
            if disagreement_rate > 0.3:
                alerts.append(f"High disagreement rate: {disagreement_rate:.2%}")
            
            if ml_error_rate > 0.1:
                alerts.append(f"High ML error rate: {ml_error_rate:.2%}")
            
            if performance_degradation > 0.2:
                alerts.append(f"Performance degradation: {performance_degradation:.2%}")
            
            # Trigger alerts
            if alerts:
                await self._trigger_alert(template_id, alerts, recent_executions[-1])
            
        except Exception as e:
            self.logger.error(f"Failed to check alert conditions: {e}")
    
    async def _trigger_alert(
        self,
        template_id: str,
        alerts: List[str],
        latest_execution: ShadowExecution
    ):
        """Trigger alert for shadow mode issues"""
        
        alert_message = f"🚨 Shadow Mode Alert for {template_id}:\n" + "\n".join(f"• {alert}" for alert in alerts)
        
        self.logger.warning(alert_message)
        
        # In a real implementation, this would integrate with alerting systems
        # (Slack, PagerDuty, email, etc.)
    
    async def analyze_shadow_performance(
        self,
        template_id: str,
        days: int = 7
    ) -> ShadowModeMetrics:
        """Analyze shadow mode performance for a template"""
        
        try:
            # Get executions from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM shadow_executions 
                WHERE template_id = ? 
                AND created_at >= datetime('now', '-{} days')
                ORDER BY created_at DESC
            '''.format(days), (template_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return ShadowModeMetrics(
                    template_id=template_id,
                    total_executions=0,
                    agreement_rate=0.0,
                    ml_error_rate=0.0,
                    rba_error_rate=0.0,
                    avg_ml_execution_time=0.0,
                    avg_rba_execution_time=0.0,
                    performance_ratio=0.0,
                    confidence_distribution={},
                    comparison_breakdown={},
                    recommendation="Insufficient data",
                    analysis_period_days=days
                )
            
            # Parse execution data
            executions = []
            for row in rows:
                executions.append(ShadowExecution(
                    execution_id=row[0],
                    template_id=row[1],
                    workflow_id=row[2],
                    step_id=row[3],
                    shadow_mode=ShadowMode(row[4]),
                    input_data=json.loads(row[5]),
                    rba_result=json.loads(row[6]),
                    ml_result=json.loads(row[7]),
                    comparison_result=ComparisonResult(row[8]),
                    confidence_score=row[9],
                    execution_time_ml_ms=row[10],
                    execution_time_rba_ms=row[11],
                    tenant_id=row[12],
                    user_id=row[13],
                    created_at=datetime.fromisoformat(row[14])
                ))
            
            # Calculate metrics
            total_executions = len(executions)
            
            agreement_count = sum(1 for exec in executions if exec.comparison_result == ComparisonResult.AGREEMENT)
            agreement_rate = agreement_count / total_executions if total_executions > 0 else 0
            
            ml_error_count = sum(1 for exec in executions if not exec.ml_result.get("success", True))
            ml_error_rate = ml_error_count / total_executions if total_executions > 0 else 0
            
            rba_error_count = sum(1 for exec in executions if not exec.rba_result.get("success", True))
            rba_error_rate = rba_error_count / total_executions if total_executions > 0 else 0
            
            avg_ml_execution_time = statistics.mean([exec.execution_time_ml_ms for exec in executions])
            avg_rba_execution_time = statistics.mean([exec.execution_time_rba_ms for exec in executions])
            performance_ratio = avg_ml_execution_time / avg_rba_execution_time if avg_rba_execution_time > 0 else 0
            
            # Confidence distribution
            confidence_ranges = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, "0.9-1.0": 0}
            for exec in executions:
                confidence = exec.confidence_score
                if confidence < 0.3:
                    confidence_ranges["0.0-0.3"] += 1
                elif confidence < 0.5:
                    confidence_ranges["0.3-0.5"] += 1
                elif confidence < 0.7:
                    confidence_ranges["0.5-0.7"] += 1
                elif confidence < 0.9:
                    confidence_ranges["0.7-0.9"] += 1
                else:
                    confidence_ranges["0.9-1.0"] += 1
            
            # Comparison breakdown
            comparison_breakdown = {}
            for result_type in ComparisonResult:
                comparison_breakdown[result_type] = sum(
                    1 for exec in executions if exec.comparison_result == result_type
                )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(
                agreement_rate, ml_error_rate, performance_ratio, total_executions
            )
            
            metrics = ShadowModeMetrics(
                template_id=template_id,
                total_executions=total_executions,
                agreement_rate=agreement_rate,
                ml_error_rate=ml_error_rate,
                rba_error_rate=rba_error_rate,
                avg_ml_execution_time=avg_ml_execution_time,
                avg_rba_execution_time=avg_rba_execution_time,
                performance_ratio=performance_ratio,
                confidence_distribution=confidence_ranges,
                comparison_breakdown=comparison_breakdown,
                recommendation=recommendation,
                analysis_period_days=days
            )
            
            # Store metrics
            await self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to analyze shadow performance: {e}")
            return ShadowModeMetrics(
                template_id=template_id,
                total_executions=0,
                agreement_rate=0.0,
                ml_error_rate=0.0,
                rba_error_rate=0.0,
                avg_ml_execution_time=0.0,
                avg_rba_execution_time=0.0,
                performance_ratio=0.0,
                confidence_distribution={},
                comparison_breakdown={},
                recommendation="Analysis failed",
                analysis_period_days=days
            )
    
    def _generate_recommendation(
        self,
        agreement_rate: float,
        ml_error_rate: float,
        performance_ratio: float,
        total_executions: int
    ) -> str:
        """Generate recommendation based on shadow mode performance"""
        
        if total_executions < 100:
            return "Continue shadow mode - need more data for reliable analysis"
        
        if ml_error_rate > 0.1:
            return "Fix ML model issues before considering promotion"
        
        if performance_ratio > 3.0:
            return "Optimize ML performance before promotion - too slow compared to RBA"
        
        if agreement_rate > 0.85:
            return "Strong candidate for promotion - high agreement with RBA"
        elif agreement_rate > 0.7:
            return "Consider canary release - good agreement but monitor closely"
        elif agreement_rate > 0.5:
            return "Continue shadow mode - moderate agreement, investigate differences"
        else:
            return "Review ML model - low agreement with RBA, may need retraining"
    
    async def _store_metrics(self, metrics: ShadowModeMetrics):
        """Store shadow mode metrics"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metric_id = str(uuid.uuid4())
            
            cursor.execute('''
                INSERT INTO shadow_metrics (
                    metric_id, template_id, total_executions, agreement_rate,
                    ml_error_rate, rba_error_rate, avg_ml_execution_time,
                    avg_rba_execution_time, performance_ratio, confidence_distribution,
                    comparison_breakdown, recommendation, analysis_period_days
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric_id,
                metrics.template_id,
                metrics.total_executions,
                metrics.agreement_rate,
                metrics.ml_error_rate,
                metrics.rba_error_rate,
                metrics.avg_ml_execution_time,
                metrics.avg_rba_execution_time,
                metrics.performance_ratio,
                json.dumps(metrics.confidence_distribution),
                json.dumps({k.value: v for k, v in metrics.comparison_breakdown.items()}),
                metrics.recommendation,
                metrics.analysis_period_days
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store metrics: {e}")
    
    def configure_shadow_mode(
        self,
        template_id: str,
        shadow_mode: ShadowMode,
        **kwargs
    ):
        """Configure shadow mode for a template"""
        
        config = self.shadow_configs.get(template_id, ShadowModeConfig(
            template_id=template_id,
            shadow_mode=shadow_mode
        ))
        
        config.shadow_mode = shadow_mode
        
        # Update configuration with provided kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.shadow_configs[template_id] = config
        
        self.logger.info(f"Configured shadow mode for {template_id}: {shadow_mode.value}")
    
    def get_shadow_config(self, template_id: str) -> Optional[ShadowModeConfig]:
        """Get shadow mode configuration for a template"""
        return self.shadow_configs.get(template_id)
    
    async def get_shadow_summary(self) -> Dict[str, Any]:
        """Get summary of all shadow mode activities"""
        
        try:
            summary = {}
            
            for template_id, config in self.shadow_configs.items():
                if config.shadow_mode != ShadowMode.DISABLED:
                    metrics = await self.analyze_shadow_performance(template_id, days=7)
                    summary[template_id] = {
                        "shadow_mode": config.shadow_mode.value,
                        "total_executions": metrics.total_executions,
                        "agreement_rate": metrics.agreement_rate,
                        "ml_error_rate": metrics.ml_error_rate,
                        "performance_ratio": metrics.performance_ratio,
                        "recommendation": metrics.recommendation
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get shadow summary: {e}")
            return {"error": str(e)}
