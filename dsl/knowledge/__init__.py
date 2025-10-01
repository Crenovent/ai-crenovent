"""
Knowledge Graph Foundation
========================

Implements the RevAI Pro Knowledge Graph as specified in Vision Doc Ch.8.2

Core Components:
- KG Schema with RevOps ontology
- Execution trace ingestion
- Entity linking and relationship mapping  
- Governance metadata integration
- Multi-tenant isolation
"""

from .ontology import EntityType, RelationshipType
from .kg_store import KnowledgeGraphStore
from .trace_ingestion import TraceIngestionEngine
from .kg_query import KnowledgeGraphQuery
from .ontology import RevOpsOntology

__all__ = [
    'EntityType', 
    'RelationshipType',
    'KnowledgeGraphStore',
    'TraceIngestionEngine', 
    'KnowledgeGraphQuery',
    'RevOpsOntology'
]
