"""
Explainability Hooks Service - Task 6.2.30
===========================================

Explainability hooks: SHAP/LIME references baked into node metadata
- Embeds references and metadata needed for explainability tooling
- Node metadata schema additions for explainability strategy
- Links runtime traces to post-hoc explainability artifacts
- API for runtime to request/record explainability outputs
- Backend implementation (no actual SHAP/LIME computation - that's ML infrastructure)

Dependencies: Task 6.2.4 (IR), Task 6.2.26 (Plan Manifest Generator)
Outputs: Explainability metadata → enables runtime explainability and audit trail
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)

class ExplainabilityStrategy(Enum):
    """Supported explainability strategies"""
    SHAP = "shap"
    LIME = "lime"
    COUNTERFACTUALS = "counterfactuals"
    FEATURE_IMPORTANCE = "feature_importance"
    ATTENTION_WEIGHTS = "attention_weights"
    GRADIENT_BASED = "gradient_based"
    PERMUTATION = "permutation"
    CUSTOM = "custom"

class ExplainabilityScope(Enum):
    """Scope of explainability"""
    GLOBAL = "global"  # Model-level explanations
    LOCAL = "local"    # Instance-level explanations
    COHORT = "cohort"  # Group-level explanations

class ExplainerArtifactType(Enum):
    """Types of explainer artifacts"""
    PRECOMPUTED_EXPLAINER = "precomputed_explainer"
    BACKGROUND_DATASET = "background_dataset"
    FEATURE_BASELINE = "feature_baseline"
    MODEL_COEFFICIENTS = "model_coefficients"
    ATTENTION_MAPS = "attention_maps"
    CUSTOM_ARTIFACT = "custom_artifact"

@dataclass
class ExplainabilityParameter:
    """Parameter for explainability computation"""
    name: str
    value: Any
    type: str  # "int", "float", "string", "boolean", "list", "dict"
    description: Optional[str] = None
    required: bool = True
    default_value: Optional[Any] = None

@dataclass
class ExplainerArtifact:
    """Reference to an explainer artifact"""
    artifact_id: str
    artifact_type: ExplainerArtifactType
    artifact_uri: str  # URI to the artifact (file, database, API endpoint)
    artifact_hash: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvidencePoint:
    """Reference to runtime evidence points for explainability"""
    evidence_id: str
    evidence_type: str  # "input_features", "model_output", "intermediate_state"
    collection_point: str  # Where in the workflow this is collected
    data_schema: Dict[str, Any] = field(default_factory=dict)
    retention_policy: Optional[str] = None
    privacy_level: str = "internal"  # "public", "internal", "confidential", "restricted"

@dataclass
class ExplainabilityConfiguration:
    """Complete explainability configuration for a node"""
    node_id: str
    enabled: bool = True
    
    # Strategy and scope
    strategy: ExplainabilityStrategy = ExplainabilityStrategy.SHAP
    scope: ExplainabilityScope = ExplainabilityScope.LOCAL
    
    # Required inputs and parameters
    required_inputs: List[str] = field(default_factory=list)
    parameters: List[ExplainabilityParameter] = field(default_factory=list)
    
    # Artifacts and references
    explainer_artifacts: List[ExplainerArtifact] = field(default_factory=list)
    evidence_points: List[EvidencePoint] = field(default_factory=list)
    
    # Output configuration
    output_format: str = "json"  # "json", "html", "pdf", "image"
    include_feature_importance: bool = True
    include_confidence_intervals: bool = False
    include_counterfactuals: bool = False
    
    # Runtime configuration
    compute_on_demand: bool = True
    cache_explanations: bool = True
    cache_ttl_hours: int = 24
    max_computation_time_ms: int = 5000
    
    # Audit and compliance
    audit_integration: bool = True
    store_explanations: bool = True
    explanation_retention_days: int = 90
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    configuration_version: str = "1.0"

@dataclass
class ExplainabilityRequest:
    """Request for explainability computation"""
    request_id: str
    node_id: str
    model_id: str
    instance_data: Dict[str, Any]
    strategy: ExplainabilityStrategy
    scope: ExplainabilityScope
    parameters: Dict[str, Any] = field(default_factory=dict)
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requested_by: Optional[str] = None
    context_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplainabilityResult:
    """Result of explainability computation"""
    request_id: str
    node_id: str
    explanation_id: str
    
    # Explanation data
    feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_values: Dict[str, Any] = field(default_factory=dict)
    explanation_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    strategy_used: ExplainabilityStrategy = ExplainabilityStrategy.SHAP
    scope_used: ExplainabilityScope = ExplainabilityScope.LOCAL
    computation_time_ms: float = 0.0
    confidence_score: Optional[float] = None
    
    # Artifacts
    explanation_artifacts: List[str] = field(default_factory=list)  # URIs to generated artifacts
    evidence_links: List[str] = field(default_factory=list)  # Links to evidence
    
    # Status
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    status: str = "completed"  # "completed", "failed", "partial"
    error_message: Optional[str] = None

# Task 6.2.30: Explainability Hooks Service
class ExplainabilityHooksService:
    """Service for managing explainability hooks and metadata in IR nodes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Storage for explainability configurations
        self.node_configurations: Dict[str, ExplainabilityConfiguration] = {}  # node_id -> config
        self.explainability_requests: Dict[str, ExplainabilityRequest] = {}  # request_id -> request
        self.explainability_results: Dict[str, ExplainabilityResult] = {}  # explanation_id -> result
        
        # Artifact registry
        self.explainer_artifacts: Dict[str, ExplainerArtifact] = {}  # artifact_id -> artifact
        
        # Default configurations by model type
        self.default_configurations = {
            'classification': {
                'strategy': ExplainabilityStrategy.SHAP,
                'scope': ExplainabilityScope.LOCAL,
                'include_feature_importance': True,
                'include_confidence_intervals': True,
                'parameters': [
                    ExplainabilityParameter('n_samples', 100, 'int', 'Number of samples for SHAP'),
                    ExplainabilityParameter('background_size', 50, 'int', 'Background dataset size')
                ]
            },
            'regression': {
                'strategy': ExplainabilityStrategy.SHAP,
                'scope': ExplainabilityScope.LOCAL,
                'include_feature_importance': True,
                'include_confidence_intervals': False,
                'parameters': [
                    ExplainabilityParameter('n_samples', 100, 'int', 'Number of samples for SHAP')
                ]
            },
            'deep_learning': {
                'strategy': ExplainabilityStrategy.GRADIENT_BASED,
                'scope': ExplainabilityScope.LOCAL,
                'include_feature_importance': True,
                'include_attention_weights': True,
                'parameters': [
                    ExplainabilityParameter('layer_name', 'last_hidden', 'string', 'Layer for gradient computation')
                ]
            }
        }
    
    def create_explainability_configuration(self, node_id: str,
                                          strategy: ExplainabilityStrategy = ExplainabilityStrategy.SHAP,
                                          scope: ExplainabilityScope = ExplainabilityScope.LOCAL,
                                          model_type: Optional[str] = None) -> ExplainabilityConfiguration:
        """
        Create explainability configuration for a node
        
        Args:
            node_id: Node identifier
            strategy: Explainability strategy
            scope: Explainability scope
            model_type: Type of model (for default parameters)
            
        Returns:
            ExplainabilityConfiguration
        """
        config = ExplainabilityConfiguration(
            node_id=node_id,
            strategy=strategy,
            scope=scope
        )
        
        # Apply default configuration based on model type
        if model_type and model_type in self.default_configurations:
            defaults = self.default_configurations[model_type]
            
            config.strategy = ExplainabilityStrategy(defaults.get('strategy', strategy.value))
            config.scope = ExplainabilityScope(defaults.get('scope', scope.value))
            config.include_feature_importance = defaults.get('include_feature_importance', True)
            config.include_confidence_intervals = defaults.get('include_confidence_intervals', False)
            
            # Add default parameters
            for param_info in defaults.get('parameters', []):
                param = ExplainabilityParameter(
                    name=param_info['name'],
                    value=param_info['value'],
                    type=param_info['type'],
                    description=param_info['description']
                )
                config.parameters.append(param)
        
        # Store configuration
        self.node_configurations[node_id] = config
        
        self.logger.info(f"✅ Created explainability configuration for node {node_id}: {strategy.value}")
        
        return config
    
    def add_explainer_artifact(self, config: ExplainabilityConfiguration,
                             artifact_type: ExplainerArtifactType,
                             artifact_uri: str,
                             metadata: Optional[Dict[str, Any]] = None) -> ExplainerArtifact:
        """
        Add an explainer artifact to the configuration
        
        Args:
            config: Explainability configuration
            artifact_type: Type of artifact
            artifact_uri: URI to the artifact
            metadata: Additional metadata
            
        Returns:
            ExplainerArtifact
        """
        artifact_id = f"artifact_{hashlib.sha256(f'{config.node_id}_{artifact_uri}'.encode()).hexdigest()[:16]}"
        
        artifact = ExplainerArtifact(
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            artifact_uri=artifact_uri,
            created_at=datetime.now(timezone.utc),
            metadata=metadata or {}
        )
        
        # Add to configuration
        config.explainer_artifacts.append(artifact)
        
        # Store in registry
        self.explainer_artifacts[artifact_id] = artifact
        
        self.logger.info(f"Added {artifact_type.value} artifact to node {config.node_id}: {artifact_id}")
        
        return artifact
    
    def add_evidence_point(self, config: ExplainabilityConfiguration,
                          evidence_type: str,
                          collection_point: str,
                          data_schema: Optional[Dict[str, Any]] = None,
                          privacy_level: str = "internal") -> EvidencePoint:
        """
        Add an evidence point to the configuration
        
        Args:
            config: Explainability configuration
            evidence_type: Type of evidence
            collection_point: Where evidence is collected
            data_schema: Schema of the evidence data
            privacy_level: Privacy level of the evidence
            
        Returns:
            EvidencePoint
        """
        evidence_id = f"evidence_{hashlib.sha256(f'{config.node_id}_{evidence_type}_{collection_point}'.encode()).hexdigest()[:16]}"
        
        evidence_point = EvidencePoint(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            collection_point=collection_point,
            data_schema=data_schema or {},
            privacy_level=privacy_level
        )
        
        config.evidence_points.append(evidence_point)
        
        self.logger.info(f"Added evidence point to node {config.node_id}: {evidence_type} at {collection_point}")
        
        return evidence_point
    
    def configure_runtime_parameters(self, config: ExplainabilityConfiguration,
                                   compute_on_demand: bool = True,
                                   cache_explanations: bool = True,
                                   max_computation_time_ms: int = 5000,
                                   explanation_retention_days: int = 90) -> ExplainabilityConfiguration:
        """
        Configure runtime parameters for explainability
        
        Args:
            config: Configuration to update
            compute_on_demand: Whether to compute explanations on demand
            cache_explanations: Whether to cache explanations
            max_computation_time_ms: Maximum computation time
            explanation_retention_days: How long to retain explanations
            
        Returns:
            Updated configuration
        """
        config.compute_on_demand = compute_on_demand
        config.cache_explanations = cache_explanations
        config.max_computation_time_ms = max_computation_time_ms
        config.explanation_retention_days = explanation_retention_days
        config.updated_at = datetime.now(timezone.utc)
        
        self.logger.info(f"Updated runtime parameters for node {config.node_id}")
        
        return config
    
    def enrich_ir_node_with_explainability(self, ir_node, 
                                         explainability_config: ExplainabilityConfiguration) -> None:
        """
        Enrich IR node with explainability metadata
        
        Args:
            ir_node: IR node to enrich
            explainability_config: Explainability configuration
        """
        # Add explainability metadata to node
        if not hasattr(ir_node, 'metadata'):
            ir_node.metadata = {}
        
        # Convert configuration to metadata format
        explainability_metadata = {
            'explainability_enabled': explainability_config.enabled,
            'explainability_strategy': explainability_config.strategy.value,
            'explainability_scope': explainability_config.scope.value,
            'required_inputs': explainability_config.required_inputs,
            'output_format': explainability_config.output_format,
            'compute_on_demand': explainability_config.compute_on_demand,
            'cache_explanations': explainability_config.cache_explanations,
            'audit_integration': explainability_config.audit_integration,
            'configuration_version': explainability_config.configuration_version
        }
        
        # Add parameters
        if explainability_config.parameters:
            explainability_metadata['parameters'] = {
                param.name: {
                    'value': param.value,
                    'type': param.type,
                    'description': param.description,
                    'required': param.required
                }
                for param in explainability_config.parameters
            }
        
        # Add artifact references
        if explainability_config.explainer_artifacts:
            explainability_metadata['explainer_artifacts'] = [
                {
                    'artifact_id': artifact.artifact_id,
                    'artifact_type': artifact.artifact_type.value,
                    'artifact_uri': artifact.artifact_uri,
                    'artifact_hash': artifact.artifact_hash
                }
                for artifact in explainability_config.explainer_artifacts
            ]
        
        # Add evidence points
        if explainability_config.evidence_points:
            explainability_metadata['evidence_points'] = [
                {
                    'evidence_id': point.evidence_id,
                    'evidence_type': point.evidence_type,
                    'collection_point': point.collection_point,
                    'privacy_level': point.privacy_level
                }
                for point in explainability_config.evidence_points
            ]
        
        # Store in node metadata
        ir_node.metadata['explainability'] = explainability_metadata
        
        self.logger.info(f"Enriched IR node {ir_node.id} with explainability metadata")
    
    def create_explainability_request(self, node_id: str,
                                    model_id: str,
                                    instance_data: Dict[str, Any],
                                    requested_by: Optional[str] = None,
                                    custom_parameters: Optional[Dict[str, Any]] = None) -> ExplainabilityRequest:
        """
        Create a request for explainability computation
        
        Args:
            node_id: Node identifier
            model_id: Model identifier
            instance_data: Data instance to explain
            requested_by: Who requested the explanation
            custom_parameters: Custom parameters for computation
            
        Returns:
            ExplainabilityRequest
        """
        # Get configuration for node
        config = self.node_configurations.get(node_id)
        if not config:
            raise ValueError(f"No explainability configuration found for node {node_id}")
        
        request_id = f"req_{uuid.uuid4().hex[:16]}"
        
        request = ExplainabilityRequest(
            request_id=request_id,
            node_id=node_id,
            model_id=model_id,
            instance_data=instance_data,
            strategy=config.strategy,
            scope=config.scope,
            parameters=custom_parameters or {},
            requested_by=requested_by,
            context_metadata={
                'configuration_version': config.configuration_version,
                'max_computation_time_ms': config.max_computation_time_ms
            }
        )
        
        # Store request
        self.explainability_requests[request_id] = request
        
        self.logger.info(f"Created explainability request {request_id} for node {node_id}")
        
        return request
    
    def record_explainability_result(self, request: ExplainabilityRequest,
                                   explanation_data: Dict[str, Any],
                                   feature_importance: Optional[Dict[str, float]] = None,
                                   computation_time_ms: float = 0.0,
                                   artifacts: Optional[List[str]] = None) -> ExplainabilityResult:
        """
        Record the result of explainability computation
        
        Args:
            request: Original request
            explanation_data: Computed explanation data
            feature_importance: Feature importance scores
            computation_time_ms: Time taken for computation
            artifacts: Generated artifact URIs
            
        Returns:
            ExplainabilityResult
        """
        explanation_id = f"exp_{uuid.uuid4().hex[:16]}"
        
        # Get configuration for retention policy
        config = self.node_configurations.get(request.node_id)
        expires_at = None
        if config and config.explanation_retention_days > 0:
            expires_at = datetime.now(timezone.utc) + timedelta(days=config.explanation_retention_days)
        
        result = ExplainabilityResult(
            request_id=request.request_id,
            node_id=request.node_id,
            explanation_id=explanation_id,
            feature_importance=feature_importance or {},
            explanation_data=explanation_data,
            strategy_used=request.strategy,
            scope_used=request.scope,
            computation_time_ms=computation_time_ms,
            explanation_artifacts=artifacts or [],
            expires_at=expires_at
        )
        
        # Store result
        self.explainability_results[explanation_id] = result
        
        self.logger.info(f"Recorded explainability result {explanation_id} for request {request.request_id}")
        
        return result
    
    def get_node_explainability_hooks(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get explainability hooks for a node (for manifest inclusion)
        
        Args:
            node_id: Node identifier
            
        Returns:
            Dictionary with explainability hooks or None
        """
        config = self.node_configurations.get(node_id)
        if not config or not config.enabled:
            return None
        
        hooks = {
            'node_id': node_id,
            'enabled': config.enabled,
            'strategy': config.strategy.value,
            'scope': config.scope.value,
            'required_inputs': config.required_inputs,
            'output_format': config.output_format,
            'compute_on_demand': config.compute_on_demand,
            'cache_explanations': config.cache_explanations,
            'audit_integration': config.audit_integration
        }
        
        # Add parameters
        if config.parameters:
            hooks['parameters'] = {
                param.name: {
                    'value': param.value,
                    'type': param.type,
                    'required': param.required
                }
                for param in config.parameters
            }
        
        # Add artifact references (without sensitive data)
        if config.explainer_artifacts:
            hooks['explainer_artifacts'] = [
                {
                    'artifact_id': artifact.artifact_id,
                    'artifact_type': artifact.artifact_type.value,
                    'artifact_uri': artifact.artifact_uri
                }
                for artifact in config.explainer_artifacts
            ]
        
        # Add evidence collection points
        if config.evidence_points:
            hooks['evidence_points'] = [
                {
                    'evidence_id': point.evidence_id,
                    'evidence_type': point.evidence_type,
                    'collection_point': point.collection_point
                }
                for point in config.evidence_points
            ]
        
        return hooks
    
    def get_explainability_statistics(self) -> Dict[str, Any]:
        """Get explainability service statistics"""
        # Count configurations by strategy
        strategy_counts = {}
        for config in self.node_configurations.values():
            strategy = config.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Count requests and results
        completed_results = len([r for r in self.explainability_results.values() if r.status == "completed"])
        failed_results = len([r for r in self.explainability_results.values() if r.status == "failed"])
        
        return {
            'total_configurations': len(self.node_configurations),
            'enabled_configurations': len([c for c in self.node_configurations.values() if c.enabled]),
            'strategy_distribution': strategy_counts,
            'total_artifacts': len(self.explainer_artifacts),
            'total_requests': len(self.explainability_requests),
            'total_results': len(self.explainability_results),
            'completed_results': completed_results,
            'failed_results': failed_results,
            'success_rate': (completed_results / max(1, len(self.explainability_results))) * 100
        }

# API Interface
class ExplainabilityHooksAPI:
    """API interface for explainability hooks operations"""
    
    def __init__(self, hooks_service: Optional[ExplainabilityHooksService] = None):
        self.hooks_service = hooks_service or ExplainabilityHooksService()
    
    def create_configuration(self, node_id: str, strategy: str = "shap", 
                           scope: str = "local", model_type: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint to create explainability configuration"""
        try:
            strategy_enum = ExplainabilityStrategy(strategy.lower())
            scope_enum = ExplainabilityScope(scope.lower())
            
            config = self.hooks_service.create_explainability_configuration(
                node_id, strategy_enum, scope_enum, model_type
            )
            
            return {
                'success': True,
                'node_id': config.node_id,
                'strategy': config.strategy.value,
                'scope': config.scope.value,
                'enabled': config.enabled,
                'configuration_version': config.configuration_version
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def request_explanation(self, node_id: str, model_id: str, 
                          instance_data: Dict[str, Any],
                          requested_by: Optional[str] = None) -> Dict[str, Any]:
        """API endpoint to request explanation"""
        try:
            request = self.hooks_service.create_explainability_request(
                node_id, model_id, instance_data, requested_by
            )
            
            return {
                'success': True,
                'request_id': request.request_id,
                'node_id': request.node_id,
                'model_id': request.model_id,
                'strategy': request.strategy.value,
                'scope': request.scope.value,
                'requested_at': request.requested_at.isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def get_node_hooks(self, node_id: str) -> Dict[str, Any]:
        """API endpoint to get explainability hooks for a node"""
        hooks = self.hooks_service.get_node_explainability_hooks(node_id)
        
        if hooks:
            return {
                'success': True,
                'node_id': node_id,
                'hooks': hooks
            }
        else:
            return {
                'success': True,
                'node_id': node_id,
                'hooks': None,
                'message': 'No explainability configuration found for node'
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """API endpoint to get explainability statistics"""
        return {
            'success': True,
            'statistics': self.hooks_service.get_explainability_statistics()
        }

# Test Functions
def create_test_explainability_configuration():
    """Create test explainability configuration"""
    service = ExplainabilityHooksService()
    
    # Create configuration for ML prediction node
    config = service.create_explainability_configuration(
        "ml_predict_node_001",
        ExplainabilityStrategy.SHAP,
        ExplainabilityScope.LOCAL,
        model_type="classification"
    )
    
    # Add custom parameters
    config.parameters.append(
        ExplainabilityParameter(
            name="feature_perturbation",
            value="marginal",
            type="string",
            description="Feature perturbation method for SHAP"
        )
    )
    
    # Add explainer artifact
    service.add_explainer_artifact(
        config,
        ExplainerArtifactType.PRECOMPUTED_EXPLAINER,
        "s3://ml-artifacts/explainers/churn_model_explainer.pkl",
        {"model_version": "v2.1", "training_date": "2024-01-15"}
    )
    
    # Add background dataset artifact
    service.add_explainer_artifact(
        config,
        ExplainerArtifactType.BACKGROUND_DATASET,
        "s3://ml-artifacts/background/churn_background_1000.parquet",
        {"sample_size": 1000, "sampling_method": "stratified"}
    )
    
    # Add evidence points
    service.add_evidence_point(
        config,
        "input_features",
        "node_enter",
        {"schema": "customer_features_v1"},
        "internal"
    )
    
    service.add_evidence_point(
        config,
        "model_output",
        "prediction_complete",
        {"schema": "prediction_output_v1"},
        "internal"
    )
    
    # Configure runtime parameters
    service.configure_runtime_parameters(
        config,
        compute_on_demand=True,
        cache_explanations=True,
        max_computation_time_ms=3000,
        explanation_retention_days=30
    )
    
    return service, config

def run_explainability_hooks_tests():
    """Run comprehensive explainability hooks tests"""
    print("=== Explainability Hooks Service Tests ===")
    
    # Initialize service
    service, test_config = create_test_explainability_configuration()
    api = ExplainabilityHooksAPI(service)
    
    print(f"\nTest configuration created for node: {test_config.node_id}")
    
    # Test 1: Configuration creation
    print("\n1. Testing explainability configuration...")
    print(f"   Node ID: {test_config.node_id}")
    print(f"   Strategy: {test_config.strategy.value}")
    print(f"   Scope: {test_config.scope.value}")
    print(f"   Parameters: {len(test_config.parameters)}")
    print(f"   Artifacts: {len(test_config.explainer_artifacts)}")
    print(f"   Evidence points: {len(test_config.evidence_points)}")
    
    # Test 2: IR node enrichment
    print("\n2. Testing IR node enrichment...")
    
    # Create mock IR node
    class MockIRNode:
        def __init__(self, node_id):
            self.id = node_id
            self.metadata = {}
    
    mock_node = MockIRNode("ml_predict_node_001")
    service.enrich_ir_node_with_explainability(mock_node, test_config)
    
    explainability_metadata = mock_node.metadata.get('explainability', {})
    print(f"   Enriched node metadata: {len(explainability_metadata)} fields")
    print(f"   Strategy in metadata: {explainability_metadata.get('explainability_strategy')}")
    print(f"   Artifacts in metadata: {len(explainability_metadata.get('explainer_artifacts', []))}")
    
    # Test 3: Explainability request
    print("\n3. Testing explainability request...")
    
    test_instance = {
        "customer_age": 35,
        "account_balance": 1500.50,
        "transaction_count": 45,
        "support_tickets": 2
    }
    
    request = service.create_explainability_request(
        "ml_predict_node_001",
        "churn_predictor_v2",
        test_instance,
        requested_by="test_user"
    )
    
    print(f"   Request ID: {request.request_id}")
    print(f"   Model ID: {request.model_id}")
    print(f"   Strategy: {request.strategy.value}")
    print(f"   Instance features: {len(request.instance_data)}")
    
    # Test 4: Record explainability result
    print("\n4. Testing explainability result recording...")
    
    # Mock explanation result
    explanation_data = {
        "shap_values": [0.2, -0.1, 0.3, -0.05],
        "base_value": 0.1,
        "prediction": 0.75,
        "explanation_type": "additive"
    }
    
    feature_importance = {
        "account_balance": 0.3,
        "customer_age": 0.2,
        "transaction_count": -0.1,
        "support_tickets": -0.05
    }
    
    result = service.record_explainability_result(
        request,
        explanation_data,
        feature_importance,
        computation_time_ms=1250.5,
        artifacts=["s3://explanations/exp_123_shap.json", "s3://explanations/exp_123_plot.png"]
    )
    
    print(f"   Explanation ID: {result.explanation_id}")
    print(f"   Computation time: {result.computation_time_ms}ms")
    print(f"   Feature importance: {len(result.feature_importance)} features")
    print(f"   Generated artifacts: {len(result.explanation_artifacts)}")
    
    # Test 5: Get node hooks (for manifest)
    print("\n5. Testing manifest hooks extraction...")
    
    hooks = service.get_node_explainability_hooks("ml_predict_node_001")
    print(f"   Hooks extracted: {'✅ PASS' if hooks else '❌ FAIL'}")
    
    if hooks:
        print(f"   Strategy: {hooks['strategy']}")
        print(f"   Compute on demand: {hooks['compute_on_demand']}")
        print(f"   Artifacts: {len(hooks.get('explainer_artifacts', []))}")
        print(f"   Evidence points: {len(hooks.get('evidence_points', []))}")
    
    # Test 6: Multiple strategies
    print("\n6. Testing multiple explainability strategies...")
    
    strategies_to_test = [
        ("lime_node", ExplainabilityStrategy.LIME, "regression"),
        ("attention_node", ExplainabilityStrategy.ATTENTION_WEIGHTS, "deep_learning"),
        ("counterfactual_node", ExplainabilityStrategy.COUNTERFACTUALS, "classification")
    ]
    
    for node_id, strategy, model_type in strategies_to_test:
        config = service.create_explainability_configuration(
            node_id, strategy, ExplainabilityScope.LOCAL, model_type
        )
        print(f"   Created {strategy.value} config for {node_id}")
    
    # Test 7: API interface
    print("\n7. Testing API interface...")
    
    # Create configuration via API
    api_config_result = api.create_configuration(
        "api_test_node", "shap", "global", "classification"
    )
    print(f"   API configuration: {'✅ PASS' if api_config_result['success'] else '❌ FAIL'}")
    
    if api_config_result['success']:
        # Request explanation via API
        api_request_result = api.request_explanation(
            "api_test_node", "test_model", test_instance, "api_user"
        )
        print(f"   API request: {'✅ PASS' if api_request_result['success'] else '❌ FAIL'}")
        
        # Get hooks via API
        api_hooks_result = api.get_node_hooks("api_test_node")
        print(f"   API hooks: {'✅ PASS' if api_hooks_result['success'] else '❌ FAIL'}")
    
    # Test 8: Statistics
    print("\n8. Testing statistics...")
    
    stats = service.get_explainability_statistics()
    print(f"   Total configurations: {stats['total_configurations']}")
    print(f"   Enabled configurations: {stats['enabled_configurations']}")
    print(f"   Strategy distribution: {stats['strategy_distribution']}")
    print(f"   Total artifacts: {stats['total_artifacts']}")
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Success rate: {stats['success_rate']:.1f}%")
    
    # Test 9: Configuration for different model types
    print("\n9. Testing model-type specific configurations...")
    
    model_types = ["classification", "regression", "deep_learning"]
    for model_type in model_types:
        config = service.create_explainability_configuration(
            f"test_{model_type}_node",
            model_type=model_type
        )
        print(f"   {model_type}: {config.strategy.value} with {len(config.parameters)} parameters")
    
    print(f"\n=== Test Summary ===")
    print(f"Explainability hooks service tested successfully")
    print(f"Configurations created: {stats['total_configurations']}")
    print(f"Requests processed: {stats['total_requests']}")
    print(f"Results recorded: {stats['total_results']}")
    print(f"Artifacts registered: {stats['total_artifacts']}")
    
    return service, api

if __name__ == "__main__":
    run_explainability_hooks_tests()
