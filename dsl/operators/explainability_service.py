"""
Explainability Service - Extends explainability logs with SHAP/LIME outputs
Task 4.1.8: Extend explainability logs with SHAP/LIME outputs
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
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExplainabilityLog:
    """Log entry for explainability data"""
    log_id: str
    workflow_id: str
    step_id: str
    execution_id: str
    model_id: str
    model_version: str
    explanation_type: str  # 'shap', 'lime', 'gradient', 'attention'
    input_features: Dict[str, Any]
    prediction: Any
    confidence: float
    explanation_data: Dict[str, Any]
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    lime_values: Optional[Dict[str, float]] = None
    gradient_values: Optional[Dict[str, float]] = None
    attention_weights: Optional[Dict[str, float]] = None
    execution_time_ms: int = 0
    tenant_id: str = ""
    user_id: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

class ExplainabilityService:
    """
    Service for managing explainability logs with SHAP/LIME outputs
    
    Provides:
    - Logging of explainability data
    - SHAP/LIME value storage
    - Feature importance tracking
    - Query and retrieval of explanation data
    - Integration with governance and audit systems
    """
    
    def __init__(self, db_path: str = "explainability_logs.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the explainability logs database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create explainability logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS explainability_logs (
                    log_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    step_id TEXT NOT NULL,
                    execution_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    explanation_type TEXT NOT NULL,
                    input_features TEXT NOT NULL,  -- JSON object
                    prediction TEXT,  -- JSON serialized
                    confidence REAL NOT NULL,
                    explanation_data TEXT NOT NULL,  -- JSON object
                    feature_importance TEXT NOT NULL,  -- JSON object
                    shap_values TEXT,  -- JSON object
                    lime_values TEXT,  -- JSON object
                    gradient_values TEXT,  -- JSON object
                    attention_weights TEXT,  -- JSON object
                    execution_time_ms INTEGER DEFAULT 0,
                    tenant_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for efficient querying
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_workflow ON explainability_logs(workflow_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_step ON explainability_logs(step_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_model ON explainability_logs(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_tenant ON explainability_logs(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_created_at ON explainability_logs(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_type ON explainability_logs(explanation_type)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Explainability logs database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize explainability database: {e}")
            raise
    
    async def log_explanation(
        self,
        workflow_id: str,
        step_id: str,
        execution_id: str,
        model_id: str,
        model_version: str,
        explanation_type: str,
        input_features: Dict[str, Any],
        prediction: Any,
        confidence: float,
        explanation_data: Dict[str, Any],
        feature_importance: Dict[str, float],
        shap_values: Optional[Dict[str, float]] = None,
        lime_values: Optional[Dict[str, float]] = None,
        gradient_values: Optional[Dict[str, float]] = None,
        attention_weights: Optional[Dict[str, float]] = None,
        execution_time_ms: int = 0,
        tenant_id: str = "",
        user_id: int = 0
    ) -> str:
        """Log explainability data to the database"""
        try:
            log_id = str(uuid.uuid4())
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO explainability_logs (
                    log_id, workflow_id, step_id, execution_id, model_id, model_version,
                    explanation_type, input_features, prediction, confidence,
                    explanation_data, feature_importance, shap_values, lime_values,
                    gradient_values, attention_weights, execution_time_ms,
                    tenant_id, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_id,
                workflow_id,
                step_id,
                execution_id,
                model_id,
                model_version,
                explanation_type,
                json.dumps(input_features),
                json.dumps(prediction),
                confidence,
                json.dumps(explanation_data),
                json.dumps(feature_importance),
                json.dumps(shap_values) if shap_values else None,
                json.dumps(lime_values) if lime_values else None,
                json.dumps(gradient_values) if gradient_values else None,
                json.dumps(attention_weights) if attention_weights else None,
                execution_time_ms,
                tenant_id,
                user_id
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged explainability data: {log_id}")
            return log_id
            
        except Exception as e:
            self.logger.error(f"Failed to log explainability data: {e}")
            raise
    
    async def handle_explainability_failure(
        self,
        model_id: str,
        workflow_id: str,
        step_id: str,
        error_message: str,
        tenant_id: str = ""
    ):
        """Handle explainability failure and trigger fallback - Task 6.4.22"""
        try:
            logger.error(f"Explainability failure for model {model_id}: {error_message}")
            
            # Trigger fallback routing
            await self._trigger_explainability_fallback(
                model_id, workflow_id, step_id, error_message, tenant_id
            )
            
        except Exception as e:
            logger.error(f"Failed to handle explainability failure: {e}")
    
    async def _trigger_explainability_fallback(
        self,
        model_id: str,
        workflow_id: str,
        step_id: str,
        error_message: str,
        tenant_id: str
    ):
        """Trigger fallback when explainability generation fails"""
        try:
            # Create fallback trigger data
            fallback_data = {
                "request_id": f"explainability_failure_{uuid.uuid4()}",
                "tenant_id": tenant_id,
                "workflow_id": workflow_id,
                "current_system": "rbia",
                "error_type": "explainability_failure",
                "error_message": f"Explainability generation failed: {error_message}",
                "model_id": model_id,
                "step_id": step_id
            }
            
            logger.warning(f"Triggering explainability failure fallback: {json.dumps(fallback_data)}")
            
        except Exception as fallback_error:
            logger.error(f"Failed to trigger explainability fallback: {fallback_error}")
    
    async def get_explanation(
        self, 
        log_id: str
    ) -> Optional[ExplainabilityLog]:
        """Get explainability log by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT log_id, workflow_id, step_id, execution_id, model_id, model_version,
                       explanation_type, input_features, prediction, confidence,
                       explanation_data, feature_importance, shap_values, lime_values,
                       gradient_values, attention_weights, execution_time_ms,
                       tenant_id, user_id, created_at
                FROM explainability_logs WHERE log_id = ?
            ''', (log_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return ExplainabilityLog(
                log_id=row[0],
                workflow_id=row[1],
                step_id=row[2],
                execution_id=row[3],
                model_id=row[4],
                model_version=row[5],
                explanation_type=row[6],
                input_features=json.loads(row[7]),
                prediction=json.loads(row[8]) if row[8] else None,
                confidence=row[9],
                explanation_data=json.loads(row[10]),
                feature_importance=json.loads(row[11]),
                shap_values=json.loads(row[12]) if row[12] else None,
                lime_values=json.loads(row[13]) if row[13] else None,
                gradient_values=json.loads(row[14]) if row[14] else None,
                attention_weights=json.loads(row[15]) if row[15] else None,
                execution_time_ms=row[16],
                tenant_id=row[17],
                user_id=row[18],
                created_at=datetime.fromisoformat(row[19])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get explanation {log_id}: {e}")
            return None
    
    async def get_explanations_by_workflow(
        self, 
        workflow_id: str,
        tenant_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ExplainabilityLog]:
        """Get all explanations for a workflow"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT log_id, workflow_id, step_id, execution_id, model_id, model_version,
                       explanation_type, input_features, prediction, confidence,
                       explanation_data, feature_importance, shap_values, lime_values,
                       gradient_values, attention_weights, execution_time_ms,
                       tenant_id, user_id, created_at
                FROM explainability_logs 
                WHERE workflow_id = ?
            '''
            params = [workflow_id]
            
            if tenant_id:
                query += ' AND tenant_id = ?'
                params.append(tenant_id)
            
            query += ' ORDER BY created_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            logs = []
            for row in rows:
                logs.append(ExplainabilityLog(
                    log_id=row[0],
                    workflow_id=row[1],
                    step_id=row[2],
                    execution_id=row[3],
                    model_id=row[4],
                    model_version=row[5],
                    explanation_type=row[6],
                    input_features=json.loads(row[7]),
                    prediction=json.loads(row[8]) if row[8] else None,
                    confidence=row[9],
                    explanation_data=json.loads(row[10]),
                    feature_importance=json.loads(row[11]),
                    shap_values=json.loads(row[12]) if row[12] else None,
                    lime_values=json.loads(row[13]) if row[13] else None,
                    gradient_values=json.loads(row[14]) if row[14] else None,
                    attention_weights=json.loads(row[15]) if row[15] else None,
                    execution_time_ms=row[16],
                    tenant_id=row[17],
                    user_id=row[18],
                    created_at=datetime.fromisoformat(row[19])
                ))
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get explanations for workflow {workflow_id}: {e}")
            return []
    
    async def get_explanations_by_model(
        self, 
        model_id: str,
        tenant_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[ExplainabilityLog]:
        """Get explanations for a specific model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT log_id, workflow_id, step_id, execution_id, model_id, model_version,
                       explanation_type, input_features, prediction, confidence,
                       explanation_data, feature_importance, shap_values, lime_values,
                       gradient_values, attention_weights, execution_time_ms,
                       tenant_id, user_id, created_at
                FROM explainability_logs 
                WHERE model_id = ?
            '''
            params = [model_id]
            
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
            
            logs = []
            for row in rows:
                logs.append(ExplainabilityLog(
                    log_id=row[0],
                    workflow_id=row[1],
                    step_id=row[2],
                    execution_id=row[3],
                    model_id=row[4],
                    model_version=row[5],
                    explanation_type=row[6],
                    input_features=json.loads(row[7]),
                    prediction=json.loads(row[8]) if row[8] else None,
                    confidence=row[9],
                    explanation_data=json.loads(row[10]),
                    feature_importance=json.loads(row[11]),
                    shap_values=json.loads(row[12]) if row[12] else None,
                    lime_values=json.loads(row[13]) if row[13] else None,
                    gradient_values=json.loads(row[14]) if row[14] else None,
                    attention_weights=json.loads(row[15]) if row[15] else None,
                    execution_time_ms=row[16],
                    tenant_id=row[17],
                    user_id=row[18],
                    created_at=datetime.fromisoformat(row[19])
                ))
            
            return logs
            
        except Exception as e:
            self.logger.error(f"Failed to get explanations for model {model_id}: {e}")
            return []
    
    async def get_feature_importance_stats(
        self, 
        model_id: str,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get feature importance statistics for a model"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get feature importance data for the last N days
            cursor.execute('''
                SELECT feature_importance, created_at
                FROM explainability_logs 
                WHERE model_id = ? AND created_at >= datetime('now', '-{} days')
            '''.format(days), (model_id,))
            
            if tenant_id:
                cursor.execute('''
                    SELECT feature_importance, created_at
                    FROM explainability_logs 
                    WHERE model_id = ? AND tenant_id = ? AND created_at >= datetime('now', '-{} days')
                '''.format(days), (model_id, tenant_id))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return {}
            
            # Aggregate feature importance data
            feature_stats = {}
            total_explanations = len(rows)
            
            for row in rows:
                feature_importance = json.loads(row[0])
                for feature, importance in feature_importance.items():
                    if feature not in feature_stats:
                        feature_stats[feature] = {
                            'total_importance': 0.0,
                            'count': 0,
                            'min_importance': float('inf'),
                            'max_importance': 0.0,
                            'avg_importance': 0.0
                        }
                    
                    feature_stats[feature]['total_importance'] += importance
                    feature_stats[feature]['count'] += 1
                    feature_stats[feature]['min_importance'] = min(feature_stats[feature]['min_importance'], importance)
                    feature_stats[feature]['max_importance'] = max(feature_stats[feature]['max_importance'], importance)
            
            # Calculate averages
            for feature, stats in feature_stats.items():
                stats['avg_importance'] = stats['total_importance'] / stats['count']
                del stats['total_importance']  # Remove raw total
            
            # Sort by average importance
            sorted_features = sorted(
                feature_stats.items(), 
                key=lambda x: x[1]['avg_importance'], 
                reverse=True
            )
            
            return {
                'model_id': model_id,
                'total_explanations': total_explanations,
                'days_analyzed': days,
                'feature_importance_stats': dict(sorted_features),
                'top_features': [f[0] for f in sorted_features[:10]]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get feature importance stats for model {model_id}: {e}")
            return {}
    
    async def get_explanation_summary(
        self, 
        workflow_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of explanations for a workflow"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT explanation_type, COUNT(*) as count, AVG(confidence) as avg_confidence,
                       AVG(execution_time_ms) as avg_execution_time
                FROM explainability_logs 
                WHERE workflow_id = ?
            '''
            params = [workflow_id]
            
            if tenant_id:
                query += ' AND tenant_id = ?'
                params.append(tenant_id)
            
            query += ' GROUP BY explanation_type'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            summary = {
                'workflow_id': workflow_id,
                'total_explanations': sum(row[1] for row in rows),
                'explanation_types': {}
            }
            
            for row in rows:
                explanation_type, count, avg_confidence, avg_execution_time = row
                summary['explanation_types'][explanation_type] = {
                    'count': count,
                    'avg_confidence': round(avg_confidence, 3),
                    'avg_execution_time_ms': round(avg_execution_time, 2)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get explanation summary for workflow {workflow_id}: {e}")
            return {}
    
    async def export_explanations(
        self, 
        workflow_id: str,
        tenant_id: Optional[str] = None,
        format: str = 'json'
    ) -> Union[str, Dict[str, Any]]:
        """Export explanations in specified format"""
        try:
            explanations = await self.get_explanations_by_workflow(workflow_id, tenant_id)
            
            if format == 'json':
                return {
                    'workflow_id': workflow_id,
                    'exported_at': datetime.utcnow().isoformat(),
                    'total_explanations': len(explanations),
                    'explanations': [
                        {
                            'log_id': exp.log_id,
                            'step_id': exp.step_id,
                            'model_id': exp.model_id,
                            'explanation_type': exp.explanation_type,
                            'confidence': exp.confidence,
                            'feature_importance': exp.feature_importance,
                            'explanation_data': exp.explanation_data,
                            'created_at': exp.created_at.isoformat()
                        }
                        for exp in explanations
                    ]
                }
            elif format == 'csv':
                # Convert to CSV format (simplified)
                csv_data = "log_id,step_id,model_id,explanation_type,confidence,created_at\n"
                for exp in explanations:
                    csv_data += f"{exp.log_id},{exp.step_id},{exp.model_id},{exp.explanation_type},{exp.confidence},{exp.created_at.isoformat()}\n"
                return csv_data
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Failed to export explanations for workflow {workflow_id}: {e}")
            return {}
    
    async def cleanup_old_logs(
        self, 
        days_to_keep: int = 90,
        tenant_id: Optional[str] = None
    ) -> int:
        """Clean up old explainability logs"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "DELETE FROM explainability_logs WHERE created_at < datetime('now', '-{} days')".format(days_to_keep)
            params = []
            
            if tenant_id:
                query = "DELETE FROM explainability_logs WHERE tenant_id = ? AND created_at < datetime('now', '-{} days')".format(days_to_keep)
                params.append(tenant_id)
            
            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {deleted_count} old explainability logs")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
            return 0
