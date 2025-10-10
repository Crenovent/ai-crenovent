"""
Drift/Bias Monitor - Implements drift/bias check triggers at node level
Task 4.1.9: Implement drift/bias check triggers at node level
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import sqlite3
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    model_id: str
    drift_type: str  # 'data_drift', 'concept_drift', 'prediction_drift'
    severity: str  # 'low', 'medium', 'high', 'critical'
    drift_score: float
    threshold: float
    affected_features: List[str]
    description: str
    detected_at: datetime
    tenant_id: str
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None

@dataclass
class BiasAlert:
    """Bias detection alert"""
    alert_id: str
    model_id: str
    bias_type: str  # 'demographic_parity', 'equalized_odds', 'disparate_impact'
    severity: str  # 'low', 'medium', 'high', 'critical'
    bias_score: float
    threshold: float
    protected_attributes: List[str]
    affected_groups: List[str]
    description: str
    detected_at: datetime
    tenant_id: str
    workflow_id: Optional[str] = None
    step_id: Optional[str] = None
    is_resolved: bool = False
    resolved_at: Optional[datetime] = None

class DriftBiasMonitor:
    """
    Monitor for drift and bias detection at ML node level
    
    Provides:
    - Data drift detection (statistical tests)
    - Concept drift detection (performance degradation)
    - Prediction drift detection (output distribution changes)
    - Bias detection (fairness metrics)
    - Alert generation and management
    - Integration with governance systems
    """
    
    def __init__(self, db_path: str = "drift_bias_monitor.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        self._drift_thresholds = {
            'data_drift': 0.1,  # Kolmogorov-Smirnov test threshold
            'concept_drift': 0.05,  # Performance degradation threshold
            'prediction_drift': 0.15  # Prediction distribution change threshold
        }
        self._bias_thresholds = {
            'demographic_parity': 0.8,  # Minimum ratio for demographic parity
            'equalized_odds': 0.1,  # Maximum difference in TPR/FPR
            'disparate_impact': 0.8  # Minimum ratio for disparate impact
        }
    
    def _init_database(self):
        """Initialize the drift/bias monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create drift alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    alert_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    drift_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    affected_features TEXT NOT NULL,  -- JSON array
                    description TEXT NOT NULL,
                    detected_at TIMESTAMP NOT NULL,
                    tenant_id TEXT NOT NULL,
                    workflow_id TEXT,
                    step_id TEXT,
                    is_resolved BOOLEAN DEFAULT 0,
                    resolved_at TIMESTAMP
                )
            ''')
            
            # Create bias alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bias_alerts (
                    alert_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    bias_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    bias_score REAL NOT NULL,
                    threshold REAL NOT NULL,
                    protected_attributes TEXT NOT NULL,  -- JSON array
                    affected_groups TEXT NOT NULL,  -- JSON array
                    description TEXT NOT NULL,
                    detected_at TIMESTAMP NOT NULL,
                    tenant_id TEXT NOT NULL,
                    workflow_id TEXT,
                    step_id TEXT,
                    is_resolved BOOLEAN DEFAULT 0,
                    resolved_at TIMESTAMP
                )
            ''')
            
            # Create model performance history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    measured_at TIMESTAMP NOT NULL,
                    tenant_id TEXT NOT NULL,
                    workflow_id TEXT,
                    step_id TEXT
                )
            ''')
            
            # Create data distribution history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS data_distribution_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    distribution_stats TEXT NOT NULL,  -- JSON object
                    sample_size INTEGER NOT NULL,
                    measured_at TIMESTAMP NOT NULL,
                    tenant_id TEXT NOT NULL,
                    workflow_id TEXT,
                    step_id TEXT
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drift_model ON drift_alerts(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drift_tenant ON drift_alerts(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_drift_detected_at ON drift_alerts(detected_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_model ON bias_alerts(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_tenant ON bias_alerts(tenant_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_bias_detected_at ON bias_alerts(detected_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_model ON model_performance_history(model_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dist_model ON data_distribution_history(model_id)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Drift/bias monitoring database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize drift/bias database: {e}")
            raise
    
    async def check_data_drift(
        self,
        model_id: str,
        current_data: List[Dict[str, Any]],
        reference_data: Optional[List[Dict[str, Any]]] = None,
        tenant_id: str = "",
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None
    ) -> List[DriftAlert]:
        """Check for data drift using statistical tests"""
        try:
            alerts = []
            
            # Get reference data if not provided
            if reference_data is None:
                reference_data = await self._get_reference_data(model_id, tenant_id)
            
            if not reference_data:
                self.logger.warning(f"No reference data found for model {model_id}")
                return alerts
            
            # Extract numeric features
            numeric_features = self._extract_numeric_features(current_data[0] if current_data else {})
            
            for feature in numeric_features:
                # Extract feature values
                current_values = [row[feature] for row in current_data if feature in row and row[feature] is not None]
                reference_values = [row[feature] for row in reference_data if feature in row and row[feature] is not None]
                
                if len(current_values) < 10 or len(reference_values) < 10:
                    continue  # Skip if insufficient data
                
                # Perform Kolmogorov-Smirnov test
                ks_statistic, p_value = stats.ks_2samp(reference_values, current_values)
                
                # Check if drift is significant
                drift_score = 1 - p_value  # Convert p-value to drift score
                threshold = self._drift_thresholds['data_drift']
                
                if drift_score > threshold:
                    severity = self._get_drift_severity(drift_score, threshold)
                    
                    alert = DriftAlert(
                        alert_id=str(uuid.uuid4()),
                        model_id=model_id,
                        drift_type='data_drift',
                        severity=severity,
                        drift_score=drift_score,
                        threshold=threshold,
                        affected_features=[feature],
                        description=f"Data drift detected in feature '{feature}': KS statistic={ks_statistic:.4f}, p-value={p_value:.4f}",
                        detected_at=datetime.utcnow(),
                        tenant_id=tenant_id,
                        workflow_id=workflow_id,
                        step_id=step_id
                    )
                    
                    alerts.append(alert)
                    await self._log_drift_alert(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check data drift for model {model_id}: {e}")
            return []
    
    async def check_concept_drift(
        self,
        model_id: str,
        current_predictions: List[Any],
        current_labels: List[Any],
        reference_accuracy: Optional[float] = None,
        tenant_id: str = "",
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None
    ) -> List[DriftAlert]:
        """Check for concept drift using performance degradation"""
        try:
            alerts = []
            
            # Calculate current accuracy
            if not current_predictions or not current_labels:
                return alerts
            
            current_accuracy = accuracy_score(current_labels, current_predictions)
            
            # Get reference accuracy if not provided
            if reference_accuracy is None:
                reference_accuracy = await self._get_reference_accuracy(model_id, tenant_id)
            
            if reference_accuracy is None:
                # Store current accuracy as reference for future comparisons
                await self._store_performance_metric(
                    model_id, 'accuracy', current_accuracy, len(current_predictions),
                    tenant_id, workflow_id, step_id
                )
                return alerts
            
            # Calculate performance degradation
            performance_degradation = reference_accuracy - current_accuracy
            threshold = self._drift_thresholds['concept_drift']
            
            if performance_degradation > threshold:
                severity = self._get_drift_severity(performance_degradation, threshold)
                
                alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    model_id=model_id,
                    drift_type='concept_drift',
                    severity=severity,
                    drift_score=performance_degradation,
                    threshold=threshold,
                    affected_features=[],
                    description=f"Concept drift detected: accuracy dropped from {reference_accuracy:.4f} to {current_accuracy:.4f} (degradation: {performance_degradation:.4f})",
                    detected_at=datetime.utcnow(),
                    tenant_id=tenant_id,
                    workflow_id=workflow_id,
                    step_id=step_id
                )
                
                alerts.append(alert)
                await self._log_drift_alert(alert)
                
                # Task 4.3.10: Auto-quarantine if drift exceeds critical threshold
                if severity in ['high', 'critical']:
                    await self._trigger_auto_quarantine(alert)
            
            # Store current performance for future comparisons
            await self._store_performance_metric(
                model_id, 'accuracy', current_accuracy, len(current_predictions),
                tenant_id, workflow_id, step_id
            )
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check concept drift for model {model_id}: {e}")
            return []
    
    async def check_prediction_drift(
        self,
        model_id: str,
        current_predictions: List[Any],
        reference_predictions: Optional[List[Any]] = None,
        tenant_id: str = "",
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None
    ) -> List[DriftAlert]:
        """Check for prediction drift using output distribution changes"""
        try:
            alerts = []
            
            # Get reference predictions if not provided
            if reference_predictions is None:
                reference_predictions = await self._get_reference_predictions(model_id, tenant_id)
            
            if not reference_predictions:
                # Store current predictions as reference
                await self._store_prediction_distribution(
                    model_id, current_predictions, tenant_id, workflow_id, step_id
                )
                return alerts
            
            # Convert predictions to numeric for comparison
            current_numeric = self._convert_predictions_to_numeric(current_predictions)
            reference_numeric = self._convert_predictions_to_numeric(reference_predictions)
            
            if len(current_numeric) < 10 or len(reference_numeric) < 10:
                return alerts
            
            # Perform Kolmogorov-Smirnov test on predictions
            ks_statistic, p_value = stats.ks_2samp(reference_numeric, current_numeric)
            drift_score = 1 - p_value
            threshold = self._drift_thresholds['prediction_drift']
            
            if drift_score > threshold:
                severity = self._get_drift_severity(drift_score, threshold)
                
                alert = DriftAlert(
                    alert_id=str(uuid.uuid4()),
                    model_id=model_id,
                    drift_type='prediction_drift',
                    severity=severity,
                    drift_score=drift_score,
                    threshold=threshold,
                    affected_features=[],
                    description=f"Prediction drift detected: KS statistic={ks_statistic:.4f}, p-value={p_value:.4f}",
                    detected_at=datetime.utcnow(),
                    tenant_id=tenant_id,
                    workflow_id=workflow_id,
                    step_id=step_id
                )
                
                alerts.append(alert)
                await self._log_drift_alert(alert)
                
                # Task 4.3.10: Auto-quarantine if drift exceeds critical threshold
                if severity in ['high', 'critical']:
                    await self._trigger_auto_quarantine(alert)
            
            # Store current predictions for future comparisons
            await self._store_prediction_distribution(
                model_id, current_predictions, tenant_id, workflow_id, step_id
            )
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check prediction drift for model {model_id}: {e}")
            return []
    
    async def check_bias(
        self,
        model_id: str,
        predictions: List[Any],
        labels: List[Any],
        protected_attributes: Dict[str, List[Any]],
        tenant_id: str = "",
        workflow_id: Optional[str] = None,
        step_id: Optional[str] = None
    ) -> List[BiasAlert]:
        """Check for bias using fairness metrics"""
        try:
            alerts = []
            
            if not predictions or not labels or not protected_attributes:
                return alerts
            
            # Check demographic parity
            dp_alerts = await self._check_demographic_parity(
                model_id, predictions, protected_attributes, tenant_id, workflow_id, step_id
            )
            alerts.extend(dp_alerts)
            
            # Check equalized odds
            eo_alerts = await self._check_equalized_odds(
                model_id, predictions, labels, protected_attributes, tenant_id, workflow_id, step_id
            )
            alerts.extend(eo_alerts)
            
            # Check disparate impact
            di_alerts = await self._check_disparate_impact(
                model_id, predictions, protected_attributes, tenant_id, workflow_id, step_id
            )
            alerts.extend(di_alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check bias for model {model_id}: {e}")
            return []
    
    async def _check_demographic_parity(
        self,
        model_id: str,
        predictions: List[Any],
        protected_attributes: Dict[str, List[Any]],
        tenant_id: str,
        workflow_id: Optional[str],
        step_id: Optional[str]
    ) -> List[BiasAlert]:
        """Check demographic parity bias"""
        alerts = []
        
        for attr_name, attr_values in protected_attributes.items():
            if len(set(attr_values)) < 2:
                continue  # Need at least 2 groups
            
            # Calculate positive prediction rates for each group
            group_rates = {}
            for group in set(attr_values):
                group_indices = [i for i, val in enumerate(attr_values) if val == group]
                group_predictions = [predictions[i] for i in group_indices]
                
                if group_predictions:
                    # Convert predictions to binary (assuming 1 is positive)
                    binary_predictions = [1 if str(p).lower() in ['1', 'true', 'positive', 'high_risk'] else 0 for p in group_predictions]
                    group_rates[group] = sum(binary_predictions) / len(binary_predictions)
            
            if len(group_rates) < 2:
                continue
            
            # Calculate demographic parity ratio
            rates = list(group_rates.values())
            min_rate = min(rates)
            max_rate = max(rates)
            
            if max_rate > 0:
                dp_ratio = min_rate / max_rate
                threshold = self._bias_thresholds['demographic_parity']
                
                if dp_ratio < threshold:
                    severity = self._get_bias_severity(dp_ratio, threshold)
                    
                    alert = BiasAlert(
                        alert_id=str(uuid.uuid4()),
                        model_id=model_id,
                        bias_type='demographic_parity',
                        severity=severity,
                        bias_score=dp_ratio,
                        threshold=threshold,
                        protected_attributes=[attr_name],
                        affected_groups=list(group_rates.keys()),
                        description=f"Demographic parity violation in {attr_name}: ratio={dp_ratio:.4f} (threshold={threshold})",
                        detected_at=datetime.utcnow(),
                        tenant_id=tenant_id,
                        workflow_id=workflow_id,
                        step_id=step_id
                    )
                    
                    alerts.append(alert)
                    await self._log_bias_alert(alert)
                
                # Task 4.3.10: Auto-quarantine if bias exceeds critical threshold  
                if alert.severity in ['high', 'critical']:
                    await self._trigger_auto_quarantine(alert)
        
        return alerts
    
    async def _check_equalized_odds(
        self,
        model_id: str,
        predictions: List[Any],
        labels: List[Any],
        protected_attributes: Dict[str, List[Any]],
        tenant_id: str,
        workflow_id: Optional[str],
        step_id: Optional[str]
    ) -> List[BiasAlert]:
        """Check equalized odds bias"""
        alerts = []
        
        for attr_name, attr_values in protected_attributes.items():
            if len(set(attr_values)) < 2:
                continue
            
            # Calculate TPR and FPR for each group
            group_metrics = {}
            for group in set(attr_values):
                group_indices = [i for i, val in enumerate(attr_values) if val == group]
                group_predictions = [predictions[i] for i in group_indices]
                group_labels = [labels[i] for i in group_indices]
                
                if not group_predictions or not group_labels:
                    continue
                
                # Convert to binary
                binary_predictions = [1 if str(p).lower() in ['1', 'true', 'positive', 'high_risk'] else 0 for p in group_predictions]
                binary_labels = [1 if str(l).lower() in ['1', 'true', 'positive', 'high_risk'] else 0 for l in group_labels]
                
                # Calculate TPR and FPR
                tp = sum(1 for p, l in zip(binary_predictions, binary_labels) if p == 1 and l == 1)
                fn = sum(1 for p, l in zip(binary_predictions, binary_labels) if p == 0 and l == 1)
                fp = sum(1 for p, l in zip(binary_predictions, binary_labels) if p == 1 and l == 0)
                tn = sum(1 for p, l in zip(binary_predictions, binary_labels) if p == 0 and l == 0)
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                group_metrics[group] = {'tpr': tpr, 'fpr': fpr}
            
            if len(group_metrics) < 2:
                continue
            
            # Calculate maximum differences in TPR and FPR
            tprs = [metrics['tpr'] for metrics in group_metrics.values()]
            fprs = [metrics['fpr'] for metrics in group_metrics.values()]
            
            max_tpr_diff = max(tprs) - min(tprs)
            max_fpr_diff = max(fprs) - min(fprs)
            max_diff = max(max_tpr_diff, max_fpr_diff)
            
            threshold = self._bias_thresholds['equalized_odds']
            
            if max_diff > threshold:
                severity = self._get_bias_severity(max_diff, threshold)
                
                alert = BiasAlert(
                    alert_id=str(uuid.uuid4()),
                    model_id=model_id,
                    bias_type='equalized_odds',
                    severity=severity,
                    bias_score=max_diff,
                    threshold=threshold,
                    protected_attributes=[attr_name],
                    affected_groups=list(group_metrics.keys()),
                    description=f"Equalized odds violation in {attr_name}: max difference={max_diff:.4f} (threshold={threshold})",
                    detected_at=datetime.utcnow(),
                    tenant_id=tenant_id,
                    workflow_id=workflow_id,
                    step_id=step_id
                )
                
                alerts.append(alert)
                await self._log_bias_alert(alert)
                
                # Task 4.3.10: Auto-quarantine if bias exceeds critical threshold  
                if alert.severity in ['high', 'critical']:
                    await self._trigger_auto_quarantine(alert)
        
        return alerts
    
    async def _check_disparate_impact(
        self,
        model_id: str,
        predictions: List[Any],
        protected_attributes: Dict[str, List[Any]],
        tenant_id: str,
        workflow_id: Optional[str],
        step_id: Optional[str]
    ) -> List[BiasAlert]:
        """Check disparate impact bias"""
        alerts = []
        
        for attr_name, attr_values in protected_attributes.items():
            if len(set(attr_values)) < 2:
                continue
            
            # Calculate positive prediction rates for each group
            group_rates = {}
            for group in set(attr_values):
                group_indices = [i for i, val in enumerate(attr_values) if val == group]
                group_predictions = [predictions[i] for i in group_indices]
                
                if group_predictions:
                    binary_predictions = [1 if str(p).lower() in ['1', 'true', 'positive', 'high_risk'] else 0 for p in group_predictions]
                    group_rates[group] = sum(binary_predictions) / len(binary_predictions)
            
            if len(group_rates) < 2:
                continue
            
            # Calculate disparate impact ratio
            rates = list(group_rates.values())
            min_rate = min(rates)
            max_rate = max(rates)
            
            if max_rate > 0:
                di_ratio = min_rate / max_rate
                threshold = self._bias_thresholds['disparate_impact']
                
                if di_ratio < threshold:
                    severity = self._get_bias_severity(di_ratio, threshold)
                    
                    alert = BiasAlert(
                        alert_id=str(uuid.uuid4()),
                        model_id=model_id,
                        bias_type='disparate_impact',
                        severity=severity,
                        bias_score=di_ratio,
                        threshold=threshold,
                        protected_attributes=[attr_name],
                        affected_groups=list(group_rates.keys()),
                        description=f"Disparate impact violation in {attr_name}: ratio={di_ratio:.4f} (threshold={threshold})",
                        detected_at=datetime.utcnow(),
                        tenant_id=tenant_id,
                        workflow_id=workflow_id,
                        step_id=step_id
                    )
                    
                    alerts.append(alert)
                    await self._log_bias_alert(alert)
                
                # Task 4.3.10: Auto-quarantine if bias exceeds critical threshold  
                if alert.severity in ['high', 'critical']:
                    await self._trigger_auto_quarantine(alert)
        
        return alerts
    
    def _get_drift_severity(self, drift_score: float, threshold: float) -> str:
        """Determine drift severity based on score and threshold"""
        ratio = drift_score / threshold
        if ratio >= 3.0:
            return 'critical'
        elif ratio >= 2.0:
            return 'high'
        elif ratio >= 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _get_bias_severity(self, bias_score: float, threshold: float) -> str:
        """Determine bias severity based on score and threshold"""
        ratio = bias_score / threshold
        if ratio <= 0.3:
            return 'critical'
        elif ratio <= 0.5:
            return 'high'
        elif ratio <= 0.7:
            return 'medium'
        else:
            return 'low'
    
    def _extract_numeric_features(self, sample_data: Dict[str, Any]) -> List[str]:
        """Extract numeric feature names from sample data"""
        numeric_features = []
        for key, value in sample_data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_features.append(key)
        return numeric_features
    
    def _convert_predictions_to_numeric(self, predictions: List[Any]) -> List[float]:
        """Convert predictions to numeric values for statistical tests"""
        numeric_predictions = []
        for pred in predictions:
            if isinstance(pred, (int, float)):
                numeric_predictions.append(float(pred))
            elif isinstance(pred, str):
                # Try to convert string predictions to numeric
                if pred.lower() in ['high_risk', 'true', 'positive', '1']:
                    numeric_predictions.append(1.0)
                elif pred.lower() in ['low_risk', 'false', 'negative', '0']:
                    numeric_predictions.append(0.0)
                else:
                    numeric_predictions.append(0.5)  # Default middle value
            else:
                numeric_predictions.append(0.5)  # Default middle value
        return numeric_predictions
    
    async def _get_reference_data(self, model_id: str, tenant_id: str) -> List[Dict[str, Any]]:
        """Get reference data for drift comparison"""
        # This would typically fetch from a reference data store
        # For now, return empty list
        return []
    
    async def _get_reference_accuracy(self, model_id: str, tenant_id: str) -> Optional[float]:
        """Get reference accuracy for concept drift detection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT AVG(metric_value) as avg_accuracy
                FROM model_performance_history 
                WHERE model_id = ? AND metric_name = 'accuracy'
                ORDER BY measured_at DESC LIMIT 10
            ''', (model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            return row[0] if row and row[0] is not None else None
            
        except Exception as e:
            self.logger.error(f"Failed to get reference accuracy for model {model_id}: {e}")
            return None
    
    async def _get_reference_predictions(self, model_id: str, tenant_id: str) -> List[Any]:
        """Get reference predictions for prediction drift detection"""
        # This would typically fetch from a prediction history store
        # For now, return empty list
        return []
    
    async def _store_performance_metric(
        self,
        model_id: str,
        metric_name: str,
        metric_value: float,
        sample_size: int,
        tenant_id: str,
        workflow_id: Optional[str],
        step_id: Optional[str]
    ):
        """Store performance metric for future drift detection"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_performance_history (
                    model_id, metric_name, metric_value, sample_size,
                    measured_at, tenant_id, workflow_id, step_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id, metric_name, metric_value, sample_size,
                datetime.utcnow().isoformat(), tenant_id, workflow_id, step_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store performance metric: {e}")
    
    async def _store_prediction_distribution(
        self,
        model_id: str,
        predictions: List[Any],
        tenant_id: str,
        workflow_id: Optional[str],
        step_id: Optional[str]
    ):
        """Store prediction distribution for future drift detection"""
        try:
            # Convert predictions to distribution statistics
            numeric_predictions = self._convert_predictions_to_numeric(predictions)
            
            if not numeric_predictions:
                return
            
            distribution_stats = {
                'mean': np.mean(numeric_predictions),
                'std': np.std(numeric_predictions),
                'min': np.min(numeric_predictions),
                'max': np.max(numeric_predictions),
                'median': np.median(numeric_predictions)
            }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO data_distribution_history (
                    model_id, feature_name, distribution_stats, sample_size,
                    measured_at, tenant_id, workflow_id, step_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_id, 'predictions', json.dumps(distribution_stats),
                len(numeric_predictions), datetime.utcnow().isoformat(),
                tenant_id, workflow_id, step_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store prediction distribution: {e}")
    
    async def _log_drift_alert(self, alert: DriftAlert):
        """Log drift alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO drift_alerts (
                    alert_id, model_id, drift_type, severity, drift_score, threshold,
                    affected_features, description, detected_at, tenant_id,
                    workflow_id, step_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.model_id, alert.drift_type, alert.severity,
                alert.drift_score, alert.threshold, json.dumps(alert.affected_features),
                alert.description, alert.detected_at.isoformat(), alert.tenant_id,
                alert.workflow_id, alert.step_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log drift alert: {e}")
    
    async def _log_bias_alert(self, alert: BiasAlert):
        """Log bias alert to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO bias_alerts (
                    alert_id, model_id, bias_type, severity, bias_score, threshold,
                    protected_attributes, affected_groups, description, detected_at,
                    tenant_id, workflow_id, step_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id, alert.model_id, alert.bias_type, alert.severity,
                alert.bias_score, alert.threshold, json.dumps(alert.protected_attributes),
                json.dumps(alert.affected_groups), alert.description, alert.detected_at.isoformat(),
                alert.tenant_id, alert.workflow_id, alert.step_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to log bias alert: {e}")
    
    async def get_active_alerts(
        self,
        tenant_id: str,
        alert_type: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Union[DriftAlert, BiasAlert]]:
        """Get active (unresolved) alerts"""
        try:
            alerts = []
            
            # Get drift alerts
            if alert_type is None or alert_type == 'drift':
                drift_alerts = await self._get_drift_alerts(tenant_id, severity, limit)
                alerts.extend(drift_alerts)
            
            # Get bias alerts
            if alert_type is None or alert_type == 'bias':
                bias_alerts = await self._get_bias_alerts(tenant_id, severity, limit)
                alerts.extend(bias_alerts)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get active alerts: {e}")
            return []
    
    async def _get_drift_alerts(
        self,
        tenant_id: str,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[DriftAlert]:
        """Get drift alerts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT alert_id, model_id, drift_type, severity, drift_score, threshold,
                       affected_features, description, detected_at, tenant_id,
                       workflow_id, step_id, is_resolved, resolved_at
                FROM drift_alerts 
                WHERE tenant_id = ? AND is_resolved = 0
            '''
            params = [tenant_id]
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity)
            
            query += ' ORDER BY detected_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alerts.append(DriftAlert(
                    alert_id=row[0],
                    model_id=row[1],
                    drift_type=row[2],
                    severity=row[3],
                    drift_score=row[4],
                    threshold=row[5],
                    affected_features=json.loads(row[6]),
                    description=row[7],
                    detected_at=datetime.fromisoformat(row[8]),
                    tenant_id=row[9],
                    workflow_id=row[10],
                    step_id=row[11],
                    is_resolved=bool(row[12]),
                    resolved_at=datetime.fromisoformat(row[13]) if row[13] else None
                ))
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get drift alerts: {e}")
            return []
    
    async def _get_bias_alerts(
        self,
        tenant_id: str,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[BiasAlert]:
        """Get bias alerts from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT alert_id, model_id, bias_type, severity, bias_score, threshold,
                       protected_attributes, affected_groups, description, detected_at,
                       tenant_id, workflow_id, step_id, is_resolved, resolved_at
                FROM bias_alerts 
                WHERE tenant_id = ? AND is_resolved = 0
            '''
            params = [tenant_id]
            
            if severity:
                query += ' AND severity = ?'
                params.append(severity)
            
            query += ' ORDER BY detected_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alerts.append(BiasAlert(
                    alert_id=row[0],
                    model_id=row[1],
                    bias_type=row[2],
                    severity=row[3],
                    bias_score=row[4],
                    threshold=row[5],
                    protected_attributes=json.loads(row[6]),
                    affected_groups=json.loads(row[7]),
                    description=row[8],
                    detected_at=datetime.fromisoformat(row[9]),
                    tenant_id=row[10],
                    workflow_id=row[11],
                    step_id=row[12],
                    is_resolved=bool(row[13]),
                    resolved_at=datetime.fromisoformat(row[14]) if row[14] else None
                ))
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get bias alerts: {e}")
            return []
    
    async def _trigger_auto_quarantine(self, alert: Union[DriftAlert, BiasAlert]):
        """
        Task 4.3.10: Auto-quarantine rule - disable model and trigger RBA fallback
        when drift/bias exceeds critical threshold
        """
        try:
            # Import quarantine function
            from ...rbia_drift_monitor.quarantine import auto_quarantine
            
            # Determine quarantine reason
            if isinstance(alert, DriftAlert):
                reason = f"Drift threshold exceeded: {alert.drift_type} score {alert.drift_score:.4f} > {alert.threshold:.4f}"
            else:  # BiasAlert
                reason = f"Bias threshold exceeded: {alert.bias_type} score {alert.bias_score:.4f} > {alert.threshold:.4f}"
            
            # Trigger auto-quarantine
            auto_quarantine(
                model_name=alert.model_id,
                version="current",
                tenant_id=alert.tenant_id,
                reason=reason
            )
            
            # Log quarantine event
            self.logger.critical(f"AUTO-QUARANTINE TRIGGERED: Model {alert.model_id} quarantined due to {reason}")
            
            # Update alert to indicate quarantine was triggered
            await self._log_quarantine_event(alert, reason)
            
        except Exception as e:
            self.logger.error(f"Failed to trigger auto-quarantine for alert {alert.alert_id}: {e}")
    
    async def _log_quarantine_event(self, alert: Union[DriftAlert, BiasAlert], reason: str):
        """Log quarantine event to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create quarantine events table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quarantine_events (
                    event_id TEXT PRIMARY KEY,
                    alert_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    quarantine_reason TEXT NOT NULL,
                    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL
                )
            ''')
            
            # Insert quarantine event
            event_id = str(uuid.uuid4())
            cursor.execute('''
                INSERT INTO quarantine_events (
                    event_id, alert_id, model_id, tenant_id, quarantine_reason,
                    alert_type, severity
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                event_id,
                alert.alert_id,
                alert.model_id,
                alert.tenant_id,
                reason,
                "drift" if isinstance(alert, DriftAlert) else "bias",
                alert.severity
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Logged quarantine event {event_id} for alert {alert.alert_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to log quarantine event: {e}")
    
    async def get_quarantine_events(self, tenant_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get quarantine events for a tenant"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT event_id, alert_id, model_id, tenant_id, quarantine_reason,
                       triggered_at, alert_type, severity
                FROM quarantine_events 
                WHERE tenant_id = ?
                ORDER BY triggered_at DESC 
                LIMIT ?
            ''', (tenant_id, limit))
            
            events = []
            for row in cursor.fetchall():
                events.append({
                    'event_id': row[0],
                    'alert_id': row[1],
                    'model_id': row[2],
                    'tenant_id': row[3],
                    'quarantine_reason': row[4],
                    'triggered_at': row[5],
                    'alert_type': row[6],
                    'severity': row[7]
                })
            
            conn.close()
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to get quarantine events: {e}")
            return []