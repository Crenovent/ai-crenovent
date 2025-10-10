"""
Fallback Service - Centralized fallback orchestration for ML operations
Task 4.1.5: Implement fallback logic (ML error → rule-only path)
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import sqlite3
from pathlib import Path

from .base import BaseOperator, OperatorContext, OperatorResult

logger = logging.getLogger(__name__)

@dataclass
class FallbackRule:
    """Fallback rule configuration"""
    rule_id: str
    rule_name: str
    rule_type: str  # 'threshold', 'field_check', 'custom', 'error_based'
    condition: str  # Expression to evaluate
    action: Dict[str, Any]  # Action to take when condition is met
    priority: int = 1  # Lower number = higher priority
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class FallbackExecution:
    """Fallback execution record"""
    execution_id: str
    workflow_id: str
    step_id: str
    model_id: str
    trigger_reason: str  # 'confidence_low', 'model_error', 'timeout', 'bias_detected'
    fallback_rule_id: str
    original_input: Dict[str, Any]
    fallback_output: Dict[str, Any]
    execution_time_ms: int
    success: bool
    tenant_id: str
    user_id: int
    created_at: datetime = field(default_factory=datetime.utcnow)

class FallbackService:
    """
    Centralized fallback orchestration service
    
    Provides:
    - Rule-based fallback execution
    - ML error handling and recovery
    - Fallback logging and monitoring
    - Integration with governance systems
    - Performance tracking and analytics
    """
    
    def __init__(self, db_path: str = "fallback_service.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        self._load_default_rules()
    
    def _init_database(self):
        """Initialize the fallback service database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create fallback rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fallback_rules (
                    rule_id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    rule_type TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    action TEXT NOT NULL,  -- JSON object
                    priority INTEGER DEFAULT 1,
                    enabled BOOLEAN DEFAULT 1,
                    model_id TEXT,
                    tenant_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create fallback executions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS fallback_executions (
                    execution_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    fallback_rule_id TEXT NOT NULL,
                    original_input TEXT NOT NULL,  -- JSON object
                    fallback_output TEXT NOT NULL,  -- JSON object
                    execution_time_ms INTEGER NOT NULL,
                    success BOOLEAN NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fallback_rules_model ON fallback_rules(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fallback_rules_tenant ON fallback_rules(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fallback_executions_workflow ON fallback_executions(workflow_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fallback_executions_tenant ON fallback_executions(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_fallback_executions_created_at ON fallback_executions(created_at)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Fallback service database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fallback database: {e}")
            raise
    
    def _load_default_rules(self):
        """Load default fallback rules"""
        try:
            # Check if rules already exist
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM fallback_rules')
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                self.logger.info("Default fallback rules already loaded")
                return
            
            # Load default rules
            default_rules = self._get_default_fallback_rules()
            
            for rule in default_rules:
                asyncio.create_task(self.create_fallback_rule(rule))
            
            self.logger.info(f"Loaded {len(default_rules)} default fallback rules")
            
        except Exception as e:
            self.logger.error(f"Failed to load default fallback rules: {e}")
    
    def _get_default_fallback_rules(self) -> List[FallbackRule]:
        """Get default fallback rules for common scenarios"""
        return [
            # Low confidence fallback
            FallbackRule(
                rule_id="low_confidence_fallback",
                rule_name="Low Confidence Fallback",
                rule_type="threshold",
                condition="confidence < 0.5",
                action={
                    "prediction": "unknown",
                    "confidence": 0.3,
                    "reason": "Low confidence fallback triggered",
                    "method": "default_fallback"
                },
                priority=1
            ),
            
            # Model error fallback
            FallbackRule(
                rule_id="model_error_fallback",
                rule_name="Model Error Fallback",
                rule_type="error_based",
                condition="model_error == true",
                action={
                    "prediction": "error_fallback",
                    "confidence": 0.1,
                    "reason": "Model error fallback triggered",
                    "method": "error_fallback"
                },
                priority=1
            ),
            
            # Timeout fallback
            FallbackRule(
                rule_id="timeout_fallback",
                rule_name="Timeout Fallback",
                rule_type="error_based",
                condition="timeout_exceeded == true",
                action={
                    "prediction": "timeout_fallback",
                    "confidence": 0.2,
                    "reason": "Execution timeout fallback triggered",
                    "method": "timeout_fallback"
                },
                priority=1
            ),
            
            # SaaS-specific fallback rules
            FallbackRule(
                rule_id="saas_low_mrr_fallback",
                rule_name="SaaS Low MRR Fallback",
                rule_type="field_check",
                condition="mrr < 100",
                action={
                    "prediction": "high_risk",
                    "confidence": 0.6,
                    "reason": "Low MRR fallback rule",
                    "method": "rule_based"
                },
                priority=2,
                model_id="saas_churn_predictor_v2"
            ),
            
            # Banking-specific fallback rules
            FallbackRule(
                rule_id="banking_low_income_fallback",
                rule_name="Banking Low Income Fallback",
                rule_type="field_check",
                condition="annual_income < 30000",
                action={
                    "score": 300,
                    "confidence": 0.5,
                    "reason": "Low income fallback rule",
                    "method": "rule_based"
                },
                priority=2,
                model_id="banking_credit_scorer_v2"
            ),
            
            # Insurance-specific fallback rules
            FallbackRule(
                rule_id="insurance_high_claim_fallback",
                rule_name="Insurance High Claim Fallback",
                rule_type="field_check",
                condition="claim_amount > 100000",
                action={
                    "fraud_score": 0.8,
                    "is_fraudulent": True,
                    "confidence": 0.7,
                    "reason": "High claim amount fallback rule",
                    "method": "rule_based"
                },
                priority=2,
                model_id="insurance_fraud_detector_v2"
            )
        ]
    
    async def create_fallback_rule(self, rule: FallbackRule) -> bool:
        """Create a new fallback rule"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fallback_rules (
                    rule_id, rule_name, rule_type, condition, action,
                    priority, enabled, model_id, tenant_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                rule.rule_id,
                rule.rule_name,
                rule.rule_type,
                rule.condition,
                json.dumps(rule.action),
                rule.priority,
                rule.enabled,
                getattr(rule, 'model_id', None),
                getattr(rule, 'tenant_id', None)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created fallback rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create fallback rule {rule.rule_id}: {e}")
            return False
    
    async def get_fallback_rules(
        self, 
        model_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        rule_type: Optional[str] = None
    ) -> List[FallbackRule]:
        """Get fallback rules with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT rule_id, rule_name, rule_type, condition, action,
                       priority, enabled, model_id, tenant_id, created_at
                FROM fallback_rules 
                WHERE enabled = 1
            '''
            params = []
            
            if model_id:
                query += ' AND (model_id = ? OR model_id IS NULL)'
                params.append(model_id)
            
            if tenant_id:
                query += ' AND (tenant_id = ? OR tenant_id IS NULL)'
                params.append(tenant_id)
            
            if rule_type:
                query += ' AND rule_type = ?'
                params.append(rule_type)
            
            query += ' ORDER BY priority ASC, created_at ASC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            rules = []
            for row in rows:
                rules.append(FallbackRule(
                    rule_id=row[0],
                    rule_name=row[1],
                    rule_type=row[2],
                    condition=row[3],
                    action=json.loads(row[4]),
                    priority=row[5],
                    enabled=bool(row[6]),
                    created_at=datetime.fromisoformat(row[9])
                ))
            
            return rules
            
        except Exception as e:
            self.logger.error(f"Failed to get fallback rules: {e}")
            return []
    
    async def execute_fallback(
        self,
        workflow_id: str,
        step_id: str,
        model_id: str,
        trigger_reason: str,
        original_input: Dict[str, Any],
        context: OperatorContext,
        custom_rules: Optional[List[FallbackRule]] = None
    ) -> OperatorResult:
        """Execute fallback logic for ML operation failure"""
        try:
            start_time = datetime.utcnow()
            
            # Get applicable fallback rules
            if custom_rules:
                applicable_rules = custom_rules
            else:
                applicable_rules = await self.get_fallback_rules(
                    model_id=model_id,
                    tenant_id=context.tenant_id
                )
            
            if not applicable_rules:
                return OperatorResult(
                    success=False,
                    error_message="No fallback rules available"
                )
            
            # Evaluate rules in priority order
            for rule in applicable_rules:
                if await self._evaluate_rule_condition(rule, original_input, trigger_reason):
                    # Execute fallback action
                    fallback_result = await self._execute_fallback_action(
                        rule, original_input, context
                    )
                    
                    # Log fallback execution
                    execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    await self._log_fallback_execution(
                        workflow_id, step_id, model_id, trigger_reason,
                        rule.rule_id, original_input, fallback_result.output_data,
                        execution_time, fallback_result.success, context
                    )
                    
                    return fallback_result
            
            # No applicable rules found
            return OperatorResult(
                success=False,
                error_message="No applicable fallback rules found"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute fallback: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Fallback execution failed: {e}"
            )
    
    async def _evaluate_rule_condition(
        self, 
        rule: FallbackRule, 
        input_data: Dict[str, Any], 
        trigger_reason: str
    ) -> bool:
        """Evaluate if a fallback rule condition is met"""
        try:
            if rule.rule_type == "error_based":
                # Handle error-based conditions
                if "model_error" in rule.condition and trigger_reason == "model_error":
                    return True
                elif "timeout" in rule.condition and trigger_reason == "timeout":
                    return True
                elif "bias_detected" in rule.condition and trigger_reason == "bias_detected":
                    return True
                return False
            
            elif rule.rule_type == "threshold":
                # Handle threshold-based conditions
                return self._evaluate_threshold_condition(rule.condition, input_data)
            
            elif rule.rule_type == "field_check":
                # Handle field-based conditions
                return self._evaluate_field_condition(rule.condition, input_data)
            
            elif rule.rule_type == "custom":
                # Handle custom expression conditions
                return self._evaluate_custom_condition(rule.condition, input_data)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate rule condition: {e}")
            return False
    
    def _evaluate_threshold_condition(self, condition: str, input_data: Dict[str, Any]) -> bool:
        """Evaluate threshold-based conditions"""
        try:
            # Simple threshold evaluation (e.g., "confidence < 0.5")
            if "<" in condition:
                field, threshold = condition.split("<")
                field = field.strip()
                threshold = float(threshold.strip())
                
                if field in input_data:
                    value = float(input_data[field])
                    return value < threshold
            
            elif ">" in condition:
                field, threshold = condition.split(">")
                field = field.strip()
                threshold = float(threshold.strip())
                
                if field in input_data:
                    value = float(input_data[field])
                    return value > threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate threshold condition: {e}")
            return False
    
    def _evaluate_field_condition(self, condition: str, input_data: Dict[str, Any]) -> bool:
        """Evaluate field-based conditions"""
        try:
            # Simple field evaluation (e.g., "mrr < 100")
            if "<" in condition:
                field, threshold = condition.split("<")
                field = field.strip()
                threshold = float(threshold.strip())
                
                if field in input_data:
                    value = float(input_data[field])
                    return value < threshold
            
            elif ">" in condition:
                field, threshold = condition.split(">")
                field = field.strip()
                threshold = float(threshold.strip())
                
                if field in input_data:
                    value = float(input_data[field])
                    return value > threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate field condition: {e}")
            return False
    
    def _evaluate_custom_condition(self, condition: str, input_data: Dict[str, Any]) -> bool:
        """Evaluate custom expression conditions"""
        try:
            # Create safe evaluation context
            eval_context = {
                **input_data,
                '__builtins__': {},
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool
            }
            
            # Evaluate the condition safely
            result = eval(condition, eval_context)
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate custom condition: {e}")
            return False
    
    async def _execute_fallback_action(
        self, 
        rule: FallbackRule, 
        input_data: Dict[str, Any], 
        context: OperatorContext
    ) -> OperatorResult:
        """Execute the fallback action for a rule"""
        try:
            action = rule.action
            
            # Create fallback output based on action
            fallback_output = action.copy()
            
            # Add metadata
            fallback_output.update({
                "fallback_rule_id": rule.rule_id,
                "fallback_rule_name": rule.rule_name,
                "fallback_method": action.get("method", "rule_based"),
                "fallback_reason": action.get("reason", "Fallback triggered"),
                "original_input": input_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Add evidence data
            evidence_data = {
                "fallback_executed": True,
                "rule_id": rule.rule_id,
                "rule_type": rule.rule_type,
                "trigger_reason": "rule_condition_met",
                "execution_time": datetime.utcnow().isoformat(),
                "tenant_id": context.tenant_id,
                "user_id": context.user_id
            }
            
            return OperatorResult(
                success=True,
                output_data=fallback_output,
                confidence_score=action.get("confidence", 0.5),
                evidence_data=evidence_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute fallback action: {e}")
            return OperatorResult(
                success=False,
                error_message=f"Fallback action execution failed: {e}"
            )
    
    async def _log_fallback_execution(
        self,
        workflow_id: str,
        step_id: str,
        model_id: str,
        trigger_reason: str,
        rule_id: str,
        original_input: Dict[str, Any],
        fallback_output: Dict[str, Any],
        execution_time_ms: int,
        success: bool,
        context: OperatorContext
    ):
        """Log fallback execution for monitoring and analytics"""
        try:
            execution_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO fallback_executions (
                    execution_id, workflow_id, step_id, model_id, trigger_reason,
                    fallback_rule_id, original_input, fallback_output,
                    execution_time_ms, success, tenant_id, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                execution_id,
                workflow_id,
                step_id,
                model_id,
                trigger_reason,
                rule_id,
                json.dumps(original_input),
                json.dumps(fallback_output),
                execution_time_ms,
                success,
                context.tenant_id,
                context.user_id
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged fallback execution: {execution_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log fallback execution: {e}")
    
    async def get_fallback_executions(
        self,
        workflow_id: Optional[str] = None,
        model_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[FallbackExecution]:
        """Get fallback execution history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT execution_id, workflow_id, step_id, model_id, trigger_reason,
                       fallback_rule_id, original_input, fallback_output,
                       execution_time_ms, success, tenant_id, user_id, created_at
                FROM fallback_executions 
                WHERE 1=1
            '''
            params = []
            
            if workflow_id:
                query += ' AND workflow_id = ?'
                params.append(workflow_id)
            
            if model_id:
                query += ' AND model_id = ?'
                params.append(model_id)
            
            if tenant_id:
                query += ' AND tenant_id = ?'
                params.append(tenant_id)
            
            if start_date:
                query += ' AND created_at >= ?'
                params.append(start_date.isoformat())
            
            if end_date:
                query += ' AND created_at <= ?'
                params.append(end_date.isoformat())
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            executions = []
            for row in rows:
                executions.append(FallbackExecution(
                    execution_id=row[0],
                    workflow_id=row[1],
                    step_id=row[2],
                    model_id=row[3],
                    trigger_reason=row[4],
                    fallback_rule_id=row[5],
                    original_input=json.loads(row[6]),
                    fallback_output=json.loads(row[7]),
                    execution_time_ms=row[8],
                    success=bool(row[9]),
                    tenant_id=row[10],
                    user_id=row[11],
                    created_at=datetime.fromisoformat(row[12])
                ))
            
            return executions
            
        except Exception as e:
            self.logger.error(f"Failed to get fallback executions: {e}")
            return []
    
    async def get_fallback_analytics(
        self,
        tenant_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get fallback analytics and metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get fallback statistics
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_fallbacks,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_fallbacks,
                    AVG(execution_time_ms) as avg_execution_time,
                    trigger_reason,
                    model_id
                FROM fallback_executions 
                WHERE tenant_id = ? AND created_at >= datetime('now', '-{} days')
                GROUP BY trigger_reason, model_id
            '''.format(days), (tenant_id,))
            
            rows = cursor.fetchall()
            conn.close()
            
            analytics = {
                "total_fallbacks": sum(row[0] for row in rows),
                "successful_fallbacks": sum(row[1] for row in rows),
                "success_rate": 0,
                "avg_execution_time_ms": 0,
                "by_trigger_reason": {},
                "by_model": {},
                "days_analyzed": days
            }
            
            if analytics["total_fallbacks"] > 0:
                analytics["success_rate"] = analytics["successful_fallbacks"] / analytics["total_fallbacks"]
                analytics["avg_execution_time_ms"] = sum(row[2] for row in rows) / len(rows)
            
            # Group by trigger reason
            for row in rows:
                trigger_reason = row[3]
                if trigger_reason not in analytics["by_trigger_reason"]:
                    analytics["by_trigger_reason"][trigger_reason] = {
                        "count": 0,
                        "success_rate": 0
                    }
                analytics["by_trigger_reason"][trigger_reason]["count"] += row[0]
                analytics["by_trigger_reason"][trigger_reason]["success_rate"] = row[1] / row[0] if row[0] > 0 else 0
            
            # Group by model
            for row in rows:
                model_id = row[4]
                if model_id not in analytics["by_model"]:
                    analytics["by_model"][model_id] = {
                        "count": 0,
                        "success_rate": 0
                    }
                analytics["by_model"][model_id]["count"] += row[0]
                analytics["by_model"][model_id]["success_rate"] = row[1] / row[0] if row[0] > 0 else 0
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Failed to get fallback analytics: {e}")
            return {}
    
    async def cleanup_old_executions(
        self, 
        days_to_keep: int = 90,
        tenant_id: Optional[str] = None
    ) -> int:
        """Clean up old fallback execution records"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "DELETE FROM fallback_executions WHERE created_at < datetime('now', '-{} days')".format(days_to_keep)
            params = []
            
            if tenant_id:
                query = "DELETE FROM fallback_executions WHERE tenant_id = ? AND created_at < datetime('now', '-{} days')".format(days_to_keep)
                params.append(tenant_id)
            
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old fallback execution records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old executions: {e}")
            return 0
