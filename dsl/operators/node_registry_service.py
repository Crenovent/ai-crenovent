"""
Node Registry Service - Central catalog of reusable ML blocks
Task 4.1.4: Create Node Registry service for reusable ML blocks (churn, fraud, anomaly)
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

logger = logging.getLogger(__name__)

@dataclass
class MLModelConfig:
    """Configuration for ML model in registry"""
    model_id: str
    model_name: str
    model_type: str  # 'classification', 'regression', 'anomaly_detection'
    model_version: str
    industry: str
    description: str
    input_features: List[str]
    output_schema: Dict[str, Any]
    confidence_threshold: float = 0.7
    fallback_enabled: bool = True
    explainability_enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NodeRegistryEntry:
    """Entry in the node registry"""
    node_id: str
    node_name: str
    node_type: str  # 'predict', 'score', 'classify', 'explain'
    model_config: MLModelConfig
    template_config: Dict[str, Any]
    governance_config: Dict[str, Any]
    performance_config: Dict[str, Any]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

class NodeRegistryService:
    """
    Central service for managing reusable ML blocks
    
    Provides:
    - Model registration and discovery
    - Industry-specific node templates
    - Model configuration management
    - Version control and lifecycle management
    - Integration with workflow DSL
    """
    
    def __init__(self, db_path: str = "node_registry.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        self._load_default_models()
    
    def _init_database(self):
        """Initialize the node registry database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create models table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    industry TEXT NOT NULL,
                    description TEXT,
                    input_features TEXT NOT NULL,  -- JSON array
                    output_schema TEXT NOT NULL,  -- JSON object
                    confidence_threshold REAL DEFAULT 0.7,
                    fallback_enabled BOOLEAN DEFAULT 1,
                    explainability_enabled BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by TEXT DEFAULT 'system',
                    tags TEXT,  -- JSON array
                    metadata TEXT  -- JSON object
                )
            ''')
            
            # Create nodes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_name TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    template_config TEXT NOT NULL,  -- JSON object
                    governance_config TEXT NOT NULL,  -- JSON object
                    performance_config TEXT NOT NULL,  -- JSON object
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models (model_id)
                )
            ''')
            
            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_industry ON models(industry)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_nodes_active ON nodes(is_active)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Node registry database initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _load_default_models(self):
        """Load default ML models for common use cases"""
        try:
            # Check if models already exist
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM models')
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                self.logger.info("Default models already loaded")
                return
            
            # Load default models
            default_models = self._get_default_models()
            
            for model in default_models:
                await self.register_model(model)
            
            self.logger.info(f"Loaded {len(default_models)} default models")
            
        except Exception as e:
            self.logger.error(f"Failed to load default models: {e}")
    
    def _get_default_models(self) -> List[MLModelConfig]:
        """Get default ML models for common use cases"""
        return [
            # SaaS Churn Prediction Model
            MLModelConfig(
                model_id="saas_churn_predictor_v1",
                model_name="SaaS Churn Predictor",
                model_type="classification",
                model_version="1.0.0",
                industry="saas",
                description="Predicts customer churn risk for SaaS companies",
                input_features=[
                    "days_since_last_login",
                    "monthly_recurring_revenue",
                    "support_tickets_count",
                    "feature_usage_score",
                    "contract_length_months",
                    "payment_history_score"
                ],
                output_schema={
                    "prediction": {"type": "string", "enum": ["low_risk", "medium_risk", "high_risk"]},
                    "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                    "probability": {"type": "object", "properties": {
                        "low_risk": {"type": "float"},
                        "medium_risk": {"type": "float"},
                        "high_risk": {"type": "float"}
                    }}
                },
                confidence_threshold=0.75,
                tags=["churn", "saas", "retention", "customer_success"]
            ),
            
            # Banking Credit Scoring Model
            MLModelConfig(
                model_id="banking_credit_scorer_v1",
                model_name="Banking Credit Scorer",
                model_type="regression",
                model_version="1.0.0",
                industry="banking",
                description="Scores credit risk for banking applications",
                input_features=[
                    "credit_score",
                    "annual_income",
                    "debt_to_income_ratio",
                    "employment_length_years",
                    "loan_amount",
                    "collateral_value"
                ],
                output_schema={
                    "score": {"type": "float", "min": 300, "max": 850},
                    "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                    "risk_tier": {"type": "string", "enum": ["excellent", "good", "fair", "poor"]}
                },
                confidence_threshold=0.8,
                tags=["credit", "banking", "risk", "scoring"]
            ),
            
            # Insurance Fraud Detection Model
            MLModelConfig(
                model_id="insurance_fraud_detector_v1",
                model_name="Insurance Fraud Detector",
                model_type="anomaly_detection",
                model_version="1.0.0",
                industry="insurance",
                description="Detects fraudulent insurance claims",
                input_features=[
                    "claim_amount",
                    "policy_age_days",
                    "claim_frequency",
                    "geographic_risk_score",
                    "time_to_report_hours",
                    "previous_claims_count"
                ],
                output_schema={
                    "anomaly_score": {"type": "float", "min": 0.0, "max": 1.0},
                    "is_fraudulent": {"type": "boolean"},
                    "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                    "risk_factors": {"type": "array", "items": {"type": "string"}}
                },
                confidence_threshold=0.85,
                tags=["fraud", "insurance", "anomaly", "detection"]
            ),
            
            # E-commerce Recommendation Model
            MLModelConfig(
                model_id="ecommerce_recommender_v1",
                model_name="E-commerce Recommender",
                model_type="classification",
                model_version="1.0.0",
                industry="ecommerce",
                description="Recommends products to customers",
                input_features=[
                    "user_id",
                    "product_category",
                    "purchase_history_score",
                    "browsing_behavior_score",
                    "seasonal_factor",
                    "price_sensitivity"
                ],
                output_schema={
                    "recommended_products": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                    "recommendation_reason": {"type": "string"}
                },
                confidence_threshold=0.7,
                tags=["recommendation", "ecommerce", "personalization"]
            ),
            
            # Financial Services Risk Assessment Model
            MLModelConfig(
                model_id="fs_risk_assessor_v1",
                model_name="Financial Services Risk Assessor",
                model_type="regression",
                model_version="1.0.0",
                industry="financial_services",
                description="Assesses risk for financial services transactions",
                input_features=[
                    "transaction_amount",
                    "customer_risk_score",
                    "transaction_frequency",
                    "geographic_risk",
                    "time_of_day",
                    "merchant_category_risk"
                ],
                output_schema={
                    "risk_score": {"type": "float", "min": 0.0, "max": 100.0},
                    "confidence": {"type": "float", "min": 0.0, "max": 1.0},
                    "action_required": {"type": "string", "enum": ["approve", "review", "decline"]}
                },
                confidence_threshold=0.8,
                tags=["risk", "financial_services", "compliance", "transaction"]
            )
        ]
    
    async def register_model(self, model_config: MLModelConfig) -> bool:
        """Register a new ML model in the registry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if model already exists
            cursor.execute('SELECT model_id FROM models WHERE model_id = ?', (model_config.model_id,))
            if cursor.fetchone():
                self.logger.warning(f"Model {model_config.model_id} already exists")
                conn.close()
                return False
            
            # Insert model
            cursor.execute('''
                INSERT INTO models (
                    model_id, model_name, model_type, model_version, industry,
                    description, input_features, output_schema, confidence_threshold,
                    fallback_enabled, explainability_enabled, created_by, tags, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                model_config.model_id,
                model_config.model_name,
                model_config.model_type,
                model_config.model_version,
                model_config.industry,
                model_config.description,
                json.dumps(model_config.input_features),
                json.dumps(model_config.output_schema),
                model_config.confidence_threshold,
                model_config.fallback_enabled,
                model_config.explainability_enabled,
                model_config.created_by,
                json.dumps(model_config.tags),
                json.dumps(model_config.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Registered model: {model_config.model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_config.model_id}: {e}")
            return False
    
    async def get_model(self, model_id: str) -> Optional[MLModelConfig]:
        """Get model configuration by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT model_id, model_name, model_type, model_version, industry,
                       description, input_features, output_schema, confidence_threshold,
                       fallback_enabled, explainability_enabled, created_by, tags, metadata
                FROM models WHERE model_id = ?
            ''', (model_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            return MLModelConfig(
                model_id=row[0],
                model_name=row[1],
                model_type=row[2],
                model_version=row[3],
                industry=row[4],
                description=row[5],
                input_features=json.loads(row[6]),
                output_schema=json.loads(row[7]),
                confidence_threshold=row[8],
                fallback_enabled=bool(row[9]),
                explainability_enabled=bool(row[10]),
                created_by=row[11],
                tags=json.loads(row[12]) if row[12] else [],
                metadata=json.loads(row[13]) if row[13] else {}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get model {model_id}: {e}")
            return None
    
    async def list_models(
        self, 
        industry: Optional[str] = None,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[MLModelConfig]:
        """List models with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT model_id, model_name, model_type, model_version, industry,
                       description, input_features, output_schema, confidence_threshold,
                       fallback_enabled, explainability_enabled, created_by, tags, metadata
                FROM models WHERE 1=1
            '''
            params = []
            
            if industry:
                query += ' AND industry = ?'
                params.append(industry)
            
            if model_type:
                query += ' AND model_type = ?'
                params.append(model_type)
            
            if tags:
                # Filter by tags (simplified - in production would use proper JSON querying)
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append('tags LIKE ?')
                    params.append(f'%"{tag}"%')
                query += ' AND (' + ' OR '.join(tag_conditions) + ')'
            
            query += ' ORDER BY created_at DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            models = []
            for row in rows:
                models.append(MLModelConfig(
                    model_id=row[0],
                    model_name=row[1],
                    model_type=row[2],
                    model_version=row[3],
                    industry=row[4],
                    description=row[5],
                    input_features=json.loads(row[6]),
                    output_schema=json.loads(row[7]),
                    confidence_threshold=row[8],
                    fallback_enabled=bool(row[9]),
                    explainability_enabled=bool(row[10]),
                    created_by=row[11],
                    tags=json.loads(row[12]) if row[12] else [],
                    metadata=json.loads(row[13]) if row[13] else {}
                ))
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return []
    
    async def create_node(
        self, 
        node_name: str,
        node_type: str,
        model_id: str,
        template_config: Dict[str, Any],
        governance_config: Optional[Dict[str, Any]] = None,
        performance_config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a new node in the registry"""
        try:
            # Validate model exists
            model = await self.get_model(model_id)
            if not model:
                self.logger.error(f"Model {model_id} not found")
                return None
            
            node_id = f"{node_type}_{model_id}_{uuid.uuid4().hex[:8]}"
            
            # Default configurations
            if governance_config is None:
                governance_config = {
                    "evidence_required": True,
                    "audit_logging": True,
                    "bias_monitoring": True,
                    "drift_monitoring": True
                }
            
            if performance_config is None:
                performance_config = {
                    "timeout_seconds": 30,
                    "max_retries": 3,
                    "cache_enabled": False
                }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO nodes (
                    node_id, node_name, node_type, model_id,
                    template_config, governance_config, performance_config
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                node_id,
                node_name,
                node_type,
                model_id,
                json.dumps(template_config),
                json.dumps(governance_config),
                json.dumps(performance_config)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Created node: {node_id}")
            return node_id
            
        except Exception as e:
            self.logger.error(f"Failed to create node: {e}")
            return None
    
    async def get_node(self, node_id: str) -> Optional[NodeRegistryEntry]:
        """Get node configuration by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT n.node_id, n.node_name, n.node_type, n.model_id,
                       n.template_config, n.governance_config, n.performance_config,
                       n.is_active, n.created_at, n.updated_at,
                       m.model_name, m.model_type, m.model_version, m.industry,
                       m.description, m.input_features, m.output_schema,
                       m.confidence_threshold, m.fallback_enabled, m.explainability_enabled
                FROM nodes n
                JOIN models m ON n.model_id = m.model_id
                WHERE n.node_id = ?
            ''', (node_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # Create model config
            model_config = MLModelConfig(
                model_id=row[3],
                model_name=row[10],
                model_type=row[11],
                model_version=row[12],
                industry=row[13],
                description=row[14],
                input_features=json.loads(row[15]),
                output_schema=json.loads(row[16]),
                confidence_threshold=row[17],
                fallback_enabled=bool(row[18]),
                explainability_enabled=bool(row[19])
            )
            
            return NodeRegistryEntry(
                node_id=row[0],
                node_name=row[1],
                node_type=row[2],
                model_config=model_config,
                template_config=json.loads(row[4]),
                governance_config=json.loads(row[5]),
                performance_config=json.loads(row[6]),
                is_active=bool(row[7]),
                created_at=datetime.fromisoformat(row[8]),
                updated_at=datetime.fromisoformat(row[9])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get node {node_id}: {e}")
            return None
    
    async def list_nodes(
        self, 
        node_type: Optional[str] = None,
        industry: Optional[str] = None,
        active_only: bool = True
    ) -> List[NodeRegistryEntry]:
        """List nodes with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = '''
                SELECT n.node_id, n.node_name, n.node_type, n.model_id,
                       n.template_config, n.governance_config, n.performance_config,
                       n.is_active, n.created_at, n.updated_at,
                       m.model_name, m.model_type, m.model_version, m.industry,
                       m.description, m.input_features, m.output_schema,
                       m.confidence_threshold, m.fallback_enabled, m.explainability_enabled
                FROM nodes n
                JOIN models m ON n.model_id = m.model_id
                WHERE 1=1
            '''
            params = []
            
            if node_type:
                query += ' AND n.node_type = ?'
                params.append(node_type)
            
            if industry:
                query += ' AND m.industry = ?'
                params.append(industry)
            
            if active_only:
                query += ' AND n.is_active = 1'
            
            query += ' ORDER BY n.created_at DESC'
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            nodes = []
            for row in rows:
                model_config = MLModelConfig(
                    model_id=row[3],
                    model_name=row[10],
                    model_type=row[11],
                    model_version=row[12],
                    industry=row[13],
                    description=row[14],
                    input_features=json.loads(row[15]),
                    output_schema=json.loads(row[16]),
                    confidence_threshold=row[17],
                    fallback_enabled=bool(row[18]),
                    explainability_enabled=bool(row[19])
                )
                
                nodes.append(NodeRegistryEntry(
                    node_id=row[0],
                    node_name=row[1],
                    node_type=row[2],
                    model_config=model_config,
                    template_config=json.loads(row[4]),
                    governance_config=json.loads(row[5]),
                    performance_config=json.loads(row[6]),
                    is_active=bool(row[7]),
                    created_at=datetime.fromisoformat(row[8]),
                    updated_at=datetime.fromisoformat(row[9])
                ))
            
            return nodes
            
        except Exception as e:
            self.logger.error(f"Failed to list nodes: {e}")
            return []
    
    async def get_model_registry_dict(self) -> Dict[str, MLModelConfig]:
        """Get all models as a dictionary for operator injection"""
        try:
            models = await self.list_models()
            return {model.model_id: model for model in models}
        except Exception as e:
            self.logger.error(f"Failed to get model registry dict: {e}")
            return {}
    
    async def update_model(self, model_id: str, updates: Dict[str, Any]) -> bool:
        """Update model configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build update query dynamically
            set_clauses = []
            params = []
            
            for key, value in updates.items():
                if key in ['input_features', 'output_schema', 'tags', 'metadata']:
                    set_clauses.append(f"{key} = ?")
                    params.append(json.dumps(value))
                else:
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                conn.close()
                return False
            
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            params.append(model_id)
            
            query = f"UPDATE models SET {', '.join(set_clauses)} WHERE model_id = ?"
            cursor.execute(query, params)
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Updated model: {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update model {model_id}: {e}")
            return False
    
    async def deactivate_node(self, node_id: str) -> bool:
        """Deactivate a node (soft delete)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE nodes 
                SET is_active = 0, updated_at = CURRENT_TIMESTAMP 
                WHERE node_id = ?
            ''', (node_id,))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Deactivated node: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to deactivate node {node_id}: {e}")
            return False
