#!/usr/bin/env python3
"""
RevOps Ontology Definition
=========================

Implements the RevOps-specific ontology as defined in Vision Doc Ch.8.2
- Entity types (Opportunity, Account, Quota, etc.)
- Relationship types (acted_on_by, produced, governed_by, etc.)
- Industry overlays (SaaS, Banking, Insurance)
- Governance metadata
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

class EntityType(Enum):
    """Core entity types in RevOps ontology"""
    
    # Core Business Entities
    OPPORTUNITY = "opportunity"
    ACCOUNT = "account" 
    CUSTOMER = "customer"
    QUOTA = "quota"
    FORECAST_CYCLE = "forecast_cycle"
    COMPENSATION_PLAN = "compensation_plan"
    TERRITORY = "territory"
    
    # Workflow Entities
    WORKFLOW = "workflow"
    EXECUTION_RUN = "execution_run"
    WORKFLOW_STEP = "workflow_step"
    AGENT = "agent"
    
    # Governance Entities  
    POLICY_PACK = "policy_pack"
    EVIDENCE_PACK = "evidence_pack"
    OVERRIDE = "override"
    USER = "user"
    TENANT = "tenant"
    
    # Industry-Specific Entities
    # Banking
    LOAN_APPLICATION = "loan_application"
    SANCTION = "sanction" 
    DISBURSAL = "disbursal"
    
    # Insurance
    CLAIM = "claim"
    BROKER = "broker"
    UNDERWRITING_DECISION = "underwriting_decision"
    
    # SaaS
    ARR = "arr"
    RENEWAL = "renewal"
    EXPANSION_OPPORTUNITY = "expansion_opportunity"
    
    # E-Commerce
    CAMPAIGN = "campaign"
    ORDER = "order"
    ROI = "roi"

class RelationshipType(Enum):
    """Relationship types between entities"""
    
    # Workflow Relationships
    ACTED_ON_BY = "acted_on_by"           # Opportunity -[acted_on_by]-> Workflow
    PRODUCED = "produced"                 # Workflow -[produced]-> Alert/Result
    EXECUTED = "executed"                 # User -[executed]-> Workflow
    CONTAINS_STEP = "contains_step"       # Workflow -[contains_step]-> Step
    
    # Governance Relationships
    GOVERNED_BY = "governed_by"           # Workflow -[governed_by]-> PolicyPack
    EVIDENCE = "evidence"                 # Workflow -[evidence]-> EvidencePack
    OVERRIDDEN_BY = "overridden_by"       # Workflow -[overridden_by]-> Override
    APPROVED_BY = "approved_by"           # Override -[approved_by]-> User
    
    # Business Relationships
    OWNS = "owns"                        # Account -[owns]-> Opportunity
    ASSIGNED_TO = "assigned_to"          # Opportunity -[assigned_to]-> User
    PART_OF = "part_of"                  # Opportunity -[part_of]-> ForecastCycle
    AFFECTS = "affects"                  # Workflow -[affects]-> Quota
    
    # Data Relationships
    READS_FROM = "reads_from"            # Workflow -[reads_from]-> DataSource
    DERIVES = "derives"                  # Entity -[derives]-> Entity
    UPDATES = "updates"                  # Workflow -[updates]-> Entity
    
    # Temporal Relationships
    FOLLOWS = "follows"                  # Run -[follows]-> Run
    TRIGGERED_BY = "triggered_by"        # Workflow -[triggered_by]-> Event

@dataclass
class EntityNode:
    """Represents a node in the Knowledge Graph"""
    id: str
    entity_type: EntityType
    properties: Dict[str, Any]
    tenant_id: str
    created_at: datetime
    updated_at: datetime
    
    # Governance metadata
    region: Optional[str] = None
    policy_pack_id: Optional[str] = None
    evidence_pack_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'entity_type': self.entity_type.value,
            'properties': self.properties,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'region': self.region,
            'policy_pack_id': self.policy_pack_id,
            'evidence_pack_id': self.evidence_pack_id
        }

@dataclass  
class RelationshipEdge:
    """Represents an edge in the Knowledge Graph"""
    id: str
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    properties: Dict[str, Any]
    tenant_id: str
    created_at: datetime
    
    # Governance metadata
    confidence_score: Optional[float] = None
    evidence_pack_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'source_id': self.source_id,
            'target_id': self.target_id,
            'relationship_type': self.relationship_type.value,
            'properties': self.properties,
            'tenant_id': self.tenant_id,
            'created_at': self.created_at.isoformat(),
            'confidence_score': self.confidence_score,
            'evidence_pack_id': self.evidence_pack_id
        }

class RevOpsOntology:
    """
    Central ontology manager for RevOps Knowledge Graph
    
    Defines the schema, validation rules, and entity/relationship patterns
    as specified in Vision Doc Ch.8.2
    """
    
    def __init__(self):
        self.entity_types = EntityType
        self.relationship_types = RelationshipType
        self._setup_industry_overlays()
        self._setup_validation_rules()
    
    def _setup_industry_overlays(self):
        """Setup industry-specific entity mappings"""
        self.industry_entities = {
            'SaaS': [
                EntityType.OPPORTUNITY, EntityType.ACCOUNT, EntityType.ARR,
                EntityType.RENEWAL, EntityType.EXPANSION_OPPORTUNITY
            ],
            'Banking': [
                EntityType.LOAN_APPLICATION, EntityType.SANCTION, 
                EntityType.DISBURSAL, EntityType.CUSTOMER
            ],
            'Insurance': [
                EntityType.CLAIM, EntityType.BROKER, 
                EntityType.UNDERWRITING_DECISION, EntityType.CUSTOMER
            ],
            'E-Commerce': [
                EntityType.CAMPAIGN, EntityType.ORDER, EntityType.ROI
            ]
        }
    
    def _setup_validation_rules(self):
        """Setup validation rules for relationships"""
        self.valid_relationships = {
            EntityType.OPPORTUNITY: [
                RelationshipType.ACTED_ON_BY,
                RelationshipType.ASSIGNED_TO,
                RelationshipType.PART_OF,
                RelationshipType.OWNS
            ],
            EntityType.WORKFLOW: [
                RelationshipType.PRODUCED,
                RelationshipType.GOVERNED_BY,
                RelationshipType.EVIDENCE,
                RelationshipType.OVERRIDDEN_BY,
                RelationshipType.CONTAINS_STEP
            ],
            EntityType.POLICY_PACK: [
                RelationshipType.GOVERNED_BY
            ]
        }
    
    def validate_relationship(self, source_type: EntityType, 
                            relationship: RelationshipType,
                            target_type: EntityType) -> bool:
        """Validate if a relationship is valid in the ontology"""
        return relationship in self.valid_relationships.get(source_type, [])
    
    def get_entity_schema(self, entity_type: EntityType) -> Dict[str, Any]:
        """Get the schema definition for an entity type"""
        schemas = {
            EntityType.OPPORTUNITY: {
                'external_id': 'string',
                'name': 'string', 
                'amount': 'number',
                'stage': 'string',
                'close_date': 'date',
                'probability': 'number',
                'owner_id': 'string'
            },
            EntityType.WORKFLOW: {
                'workflow_id': 'string',
                'name': 'string',
                'automation_type': 'string',
                'module': 'string',
                'version': 'string',
                'status': 'string'
            },
            EntityType.EXECUTION_RUN: {
                'run_id': 'string',
                'workflow_id': 'string',
                'started_at': 'datetime',
                'ended_at': 'datetime',
                'status': 'string',
                'cost_tokens': 'number',
                'latency_ms': 'number',
                'trust_score': 'number'
            },
            EntityType.EVIDENCE_PACK: {
                'evidence_id': 'string',
                'workflow_id': 'string',
                'inputs_hash': 'string',
                'outputs_hash': 'string',
                'policy_checks': 'array',
                'compliance_status': 'string'
            }
        }
        return schemas.get(entity_type, {})
    
    def get_industry_entities(self, industry: str) -> List[EntityType]:
        """Get entity types relevant for a specific industry"""
        core_entities = [
            EntityType.WORKFLOW, EntityType.EXECUTION_RUN,
            EntityType.POLICY_PACK, EntityType.EVIDENCE_PACK,
            EntityType.USER, EntityType.TENANT
        ]
        industry_specific = self.industry_entities.get(industry, [])
        return core_entities + industry_specific
    
    def create_entity_node(self, entity_id: str, entity_type: EntityType,
                          properties: Dict[str, Any], tenant_id: str,
                          **metadata) -> EntityNode:
        """Create a new entity node with proper validation"""
        now = datetime.utcnow()
        
        return EntityNode(
            id=entity_id,
            entity_type=entity_type,
            properties=properties,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
            **metadata
        )
    
    def create_relationship_edge(self, edge_id: str, source_id: str, target_id: str,
                               relationship_type: RelationshipType,
                               properties: Dict[str, Any], tenant_id: str,
                               **metadata) -> RelationshipEdge:
        """Create a new relationship edge"""
        now = datetime.utcnow()
        
        return RelationshipEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            properties=properties,
            tenant_id=tenant_id,
            created_at=now,
            **metadata
        )

# Example usage patterns from Vision Doc Ch.8.2
EXAMPLE_KG_PATTERNS = {
    "forecast_workflow_pattern": {
        "description": "Forecast rebalancing workflow with governance",
        "nodes": [
            {"type": "forecast_cycle", "id": "Q3_NA"},
            {"type": "workflow", "id": "wf_forecast_rebalance"},  
            {"type": "policy_pack", "id": "sox_forecast"},
            {"type": "evidence_pack", "id": "ep2025"},
            {"type": "override", "id": "CRO_override_2025"}
        ],
        "relationships": [
            {"from": "Q3_NA", "to": "wf_forecast_rebalance", "type": "acted_on_by"},
            {"from": "wf_forecast_rebalance", "to": "sox_forecast", "type": "governed_by"},
            {"from": "wf_forecast_rebalance", "to": "ep2025", "type": "evidence"},
            {"from": "wf_forecast_rebalance", "to": "CRO_override_2025", "type": "overridden_by"}
        ]
    },
    "pipeline_hygiene_pattern": {
        "description": "Pipeline hygiene workflow acting on opportunities",
        "nodes": [
            {"type": "opportunity", "id": "opp_456"},
            {"type": "workflow", "id": "wf_pipeline_hygiene"},
            {"type": "alert", "id": "stalled_opps_alert"},
            {"type": "policy_pack", "id": "sox_pipeline"}
        ],
        "relationships": [
            {"from": "opp_456", "to": "wf_pipeline_hygiene", "type": "acted_on_by"},
            {"from": "wf_pipeline_hygiene", "to": "stalled_opps_alert", "type": "produced"},
            {"from": "wf_pipeline_hygiene", "to": "sox_pipeline", "type": "governed_by"}
        ]
    }
}
